"""澄清质量评估：把 gt.json 的「考点」当作 blocker 集合，用 LLM judge 判定覆盖。

指标口径对齐澄清类基准（依据 paper/1.html 汇总的 23 篇论文）：

  Recall    = |B_addr| / |B|      HiL-Bench：被至少一个相关提问解决的考点占比
  Precision = |Q_rel|  / |Q|      HiL-Bench：提问中命中考点的比例（惩罚滥问）
  Ask-F1    = 2PR/(P+R)           两者调和平均，防"提问轰炸"刷 recall
  KQC_1     首轮提问的考点覆盖率   ClarEval：零样本推断能力
  KQC_all   全对话累计覆盖率       ClarEval 多轮版 = Recall
  Redundant 无关/重复提问次数      CLAMBER：越低越好
  CE-A      问了的考点里问对方向的比例  Clarify-Then-Search：|C_right|/|C_asked|
  CE-B      问对且真正推进任务的比例    UserBench：|C_right ∩ C_adv|/|C_asked|
  by_tag    按信息缺口标签拆开的 recall / CE-A（本文扩展）

by_tag 是我们自己加的一维：总 recall 相同的两个模型，可能一个是各类均匀漏、
另一个是「数据语义」全问到而「评分口径」全没问——后者才是论文要讲的失败模式。
标签取自 gt.json 每条澄清项的 `标签` 字段，judge 不看它。

CE-A / CE-B 的统计单位是**考点**而非提问：分母只含"agent 确实就该考点发起过
提问"的考点。CE-A 衡量"一旦决定问，方向对不对"；CE-B 更严格，还要求回收到的
答案真正转化为任务推进（本实现以最终 validity>0 作为推进判据）。

judge 采用二元判定（命中/未命中）而非打分，理由同 HiL-Bench：二元信号可复现、
可跨模型比较；打分会随 judge 模型漂移。judge 模型需固定版本并记入结果。
"""

from __future__ import annotations

import json
import re
from dataclasses import dataclass, field
from pathlib import Path

_SYSTEM = (
    "You grade whether an agent's clarification questions uncovered a specific hidden "
    "requirement. Judge strictly and reply with JSON only."
)

_ARBITRATION_SYSTEM = (
    "You arbitrate clarification-judge votes. Review the hidden requirement, "
    "the agent questions, and the independent judge ballots. Reply with JSON only."
)

# 判据统一为"客户会不会因此说出该信息"——单一、可操作，且与实际 answerer 行为一致。
# 早先版本并列了两条判据（"回答会否揭示" vs "是否只点大话题"），二者在边界样本上
# 给出相反结论，judge 因而在同一输入上反复横跳（实测 5 次 3 比 2 分裂，
# temperature=0 与换模型均无效）。
#
# 【为什么按考点循环、而不按提问循环】2026-08-25 试过改成 HiL-Bench
# （arXiv 2604.09408）的 ask_human() 形状——按提问循环、一条提问最多认领一个
# 考点（其 Appendix D.1 "Multi-blocker questions. Matched to at most one blocker"）
# ——已撤回。它的 at-most-one 成立的前提是 blocker **被构造成互相独立**
# （Appendix B 准则 5："Resolving one blocker must not reveal the resolution of
# any other blocker in the same task"，7 条硬准则任一不满足即拒收，靠 5-6 轮
# 人工审计保证）。我们的考点从真实 case 提取、天然成组（「距离度量」与
# 「时长速度分段」共同决定行驶时长），套用会系统性惩罚"成组澄清相关约定"这一
# 专业行为——实测把分组标注用于限制归属，recall 0.67→0.33。
# 详见 docs/FRAMEWORK.md「不采用 HiL-Bench 的一问最多兑一个考点」。
#
# 代价：多归属下"一句话打包三件事"在 recall 侧不吃亏，防线只剩把 precision
# 分母撑大 —— 这就是 split_atomic 存在的理由，两者是一套，不能只留一个。
_PROMPT = """A client holds the hidden requirement below. An agent asked the questions
below to clarify the task. Decide what the client would reveal in reply.

HIDDEN REQUIREMENT (id: {kid})
{desc}
{triggers}

AGENT'S QUESTIONS (each on its own line, numbered)
{questions}

The client answers cooperatively and concretely: when a question points at a
topic they hold information about, they state the actual values or rules rather
than a generic description. They do NOT volunteer information the agent never
asked about.

Judge two things:

1. ATTEMPTED — does any question point at this requirement's topic, however
   loosely phrased?
2. COVERED — answering that question, would the client state the specific
   requirement above?
   - Yes if the question names the specific quantity, rule, field or entity this
     requirement is about, so the client knows exactly which convention to state.
     The agent need not already know the value, nor use the same wording; asking
     what the convention is counts.
   - No if the question merely names a topic, module or data source and asks what
     to "watch out for", what the "rules" are, or what is "special" — even when
     that topic is the right one. Such a question does not tell the client which
     of the many conventions in that area is wanted, so a cooperative client
     would ask back rather than recite this requirement.
     Examples that must be judged No: "anything special about the data?",
     "any rules I should know about billing?", "any pitfalls when reading these
     sheets?". Naming the right domain does not make a question specific.

Reply with JSON only:
{{"attempted": true|false, "covered": true|false,
  "question_indices": [<1-based indices that cover it>]}}"""

_ARBITRATION_PROMPT = """Resolve a disagreement among independent clarification judges.

HIDDEN REQUIREMENT (id: {kid})
{desc}
{triggers}

AGENT'S QUESTIONS (each on its own line, numbered)
{questions}

INDEPENDENT JUDGE BALLOTS
{ballots}

Apply the same strict rule as the original judge: ATTEMPTED means a question points
at the requirement's topic; COVERED means a cooperative client would state the
specific requirement in reply. Do not infer information the agent did not ask for.
Choose the best-supported decision from the evidence, not the majority merely because
it is the majority. Return JSON only:
{{"attempted": true|false, "covered": true|false,
  "question_indices": [<1-based indices that cover it>]}}"""


def _redact_judge_error(message: str, limit: int = 500) -> str:
    """Return a bounded, credential-safe error summary for the audit log.

    Judge exceptions can contain a provider URL, an Authorization header, or a
    snippet of a provider response.  Keep enough text to diagnose transport and
    parsing failures, but never persist a bearer/API key in ``clarify_score``.
    """
    text = str(message or "")
    text = re.sub(r"(?i)bearer\s+[^\s,;]+", "Bearer [REDACTED]", text)
    text = re.sub(
        r"(?i)((?:api[_-]?key|authorization|token))\s*[:=]\s*[^\s,;]+",
        lambda match: f"{match.group(1)}=[REDACTED]",
        text,
    )
    text = re.sub(r"\bsk-[A-Za-z0-9_-]{8,}\b", "[REDACTED]", text)
    return text[:limit]


@dataclass
class ClarifyScore:
    n_blockers: int = 0
    n_questions: int = 0
    covered: list[str] = field(default_factory=list)
    relevant_qs: set[int] = field(default_factory=set)
    # 判定明细：每个考点判了什么、依据哪几条提问。落盘供人工抽检——
    # 澄清指标全靠 judge，不留痕就无法核对"这个分是怎么来的"。
    audit: list = field(default_factory=list)
    # 实际送给 judge 的提问全文（切分后），precision 的分母就是它的长度
    questions_judged: list = field(default_factory=list)
    first_turn_covered: list[str] = field(default_factory=list)
    # 每个被覆盖考点「首次被问到」的轮次，用于 ATC
    covered_at_turn: dict = field(default_factory=dict)
    # CE-A/CE-B 用：agent 就该考点发起过提问（无论问对与否）
    asked_blockers: list[str] = field(default_factory=list)
    # 澄清项 → 信息缺口标签。只用于把覆盖率按信息类型拆开，judge 不看。
    tags: dict = field(default_factory=dict)
    task_advanced: bool | None = None      # 最终是否产出有效解，CE-B 的推进判据
    # 兼容旧结果的单值字段；多模型判定时另存实际模型列表。
    judge_model: str | None = None
    judge_models: list[str] = field(default_factory=list)
    judge_errors: int = 0
    # 每个失败投票的可诊断信息。与 judge_errors（考点级计数）分开，便于定位
    # 到具体模型/考点；错误摘要经过脱敏且有长度上限。
    judge_error_details: list[dict] = field(default_factory=list)
    # 每个考点判定的多数派占比（1.0 = 5 次全同），用于报告 judge 自一致性
    agreements: list[float] = field(default_factory=list)
    # 逐模型投票计数，用于跨 run 的一致性/系统偏差审计。
    # 只保存布尔结果和计数，不保存模型的长文本回复。
    judge_vote_stats: dict[str, dict[str, int]] = field(default_factory=dict)
    # 仲裁仅在显式开启且五票出现分歧/失败时调用；默认关闭，不改变历史口径。
    arbitration_model: str | None = None
    arbitration_calls: int = 0
    arbitration_used: int = 0
    arbitration_errors: int = 0
    arbitration_error_details: list[dict] = field(default_factory=list)

    @property
    def judge_failed(self) -> bool:
        """judge 是否全军覆没。

        全部考点判定失败时，recall 会算成 0.0——与「一个都没问到」无法区分，
        基础设施故障就此伪装成 agent 表现差（实测 kimi-k3 作 judge 时
        judge_errors=3、三个考点全失败，却报 recall=0.0）。故显式区分。

        分母是**考点数**，因为判定按考点循环、每条考点判一次（见 _PROMPT 上方
        关于为何不按提问循环的说明）。
        """
        return self.n_blockers > 0 and self.judge_errors >= self.n_blockers

    @property
    def recall(self) -> float | None:
        if self.judge_failed:
            return None
        return len(self.covered) / self.n_blockers if self.n_blockers else None

    @property
    def precision(self) -> float | None:
        if self.judge_failed:
            return None
        return len(self.relevant_qs) / self.n_questions if self.n_questions else None

    @property
    def ask_f1(self) -> float | None:
        p, r = self.precision, self.recall
        if p is None or r is None or (p + r) == 0:
            return 0.0 if (p is not None and r is not None) else None
        return 2 * p * r / (p + r)

    @property
    def atc(self) -> float | None:
        """Average Turns to Clarify（ClarEval）：平均问到第几轮才问出关键信息。

        只对**已覆盖**的考点求平均——没问出来的考点没有"第几轮"可言，
        把它记成 max_rounds 会让"少问但问得准"和"问了一堆也没问到"混为一谈，
        前者的 ATC 应当低。覆盖率本身由 recall 承担。

        1.0 = 首轮就问到了（最好）。
        """
        if not self.covered_at_turn:
            return None
        return round(sum(self.covered_at_turn.values()) / len(self.covered_at_turn), 3)

    @property
    def kqc_first_turn(self) -> float | None:
        return len(self.first_turn_covered) / self.n_blockers if self.n_blockers else None

    @property
    def redundant(self) -> int:
        return self.n_questions - len(self.relevant_qs)

    @property
    def ce_a(self) -> float | None:
        """|C_right| / |C_asked|：问了的考点里，问对方向的比例。"""
        if not self.asked_blockers:
            return None                     # 一次都没问，该指标无定义
        return len(self.covered) / len(self.asked_blockers)

    @property
    def ce_b(self) -> float | None:
        """|C_right ∩ C_adv| / |C_asked|：问对且任务真正被推进的比例。"""
        if not self.asked_blockers or self.task_advanced is None:
            return None
        advanced = len(self.covered) if self.task_advanced else 0
        return advanced / len(self.asked_blockers)

    @property
    def by_tag(self) -> dict:
        """按信息缺口标签拆开的覆盖率。

        回答「模型是在哪一类信息上瞎」——总 recall 0.4 可能是各类均匀漏，
        也可能是「数据语义」全问到、「评分口径」全没问，两者的结论完全不同。

        每档给三个数：
          n        该标签下的澄清项总数（分母，与 gt.json 一致）
          covered  问到且问对的条数
          asked    发起过提问的条数（含问偏了的）
          recall   covered / n
          ce_a     covered / asked，问了的里面问对的比例；asked=0 时为 None

        judge 全军覆没时返回 {}——此时每一档都会是 0，与「真的一条没问到」
        无法区分，跟 recall 的处理保持一致。
        """
        if self.judge_failed or not self.tags:
            return {}
        cov, ask = set(self.covered), set(self.asked_blockers)
        out: dict[str, dict] = {}
        for kid, tag in self.tags.items():
            d = out.setdefault(tag or "未标注",
                               {"n": 0, "covered": 0, "asked": 0})
            d["n"] += 1
            d["covered"] += kid in cov
            d["asked"] += kid in ask
        for d in out.values():
            d["recall"] = round(d["covered"] / d["n"], 4) if d["n"] else None
            d["ce_a"] = round(d["covered"] / d["asked"], 4) if d["asked"] else None
        return dict(sorted(out.items()))

    def to_dict(self) -> dict:
        return {
            "n_blockers": self.n_blockers,
            "n_questions": self.n_questions,
            "recall": self.recall,
            "precision": self.precision,
            "ask_f1": self.ask_f1,
            "kqc_first_turn": self.kqc_first_turn,
            "atc": self.atc,
            "covered_at_turn": dict(self.covered_at_turn),
            "kqc_all": self.recall,          # 多轮 KQC 与 Recall 同义
            "redundant_questions": self.redundant,
            "ce_a": self.ce_a,
            "ce_b": self.ce_b,
            "n_asked_blockers": len(self.asked_blockers),
            # 名单也落盘：只留个数的话，事后拿 gt.json 反查也补不出
            # 「问了但问偏了」那一档的标签分布（历史数据就是这么丢的）。
            "asked_blockers": self.asked_blockers,
            "by_tag": self.by_tag,
            "task_advanced": self.task_advanced,
            "covered_blockers": self.covered,
            "judge_model": self.judge_model,
            "judge_models": list(self.judge_models),
            "judge_errors": self.judge_errors,
            "judge_failed": self.judge_failed,
            "judge_error_details": list(self.judge_error_details),
            "judge_decisions": len(self.audit),
            "judge_total_decisions": len(self.audit) + max(0, self.judge_errors),
            # 审计用：判定过程留痕，可人工抽检"这个分怎么来的"
            "questions_judged": list(self.questions_judged),
            "audit": list(self.audit),
            "judge_votes": JUDGE_VOTES,
            "judge_agreement_mean": (
                round(sum(self.agreements) / len(self.agreements), 3)
                if self.agreements else None),
            "judge_unanimous_rate": (
                round(sum(1 for a in self.agreements if a == 1.0) / len(self.agreements), 3)
                if self.agreements else None),
            "judge_vote_stats": self._vote_stats_dict(),
            "arbitration_model": self.arbitration_model,
            "arbitration_calls": self.arbitration_calls,
            "arbitration_used": self.arbitration_used,
            "arbitration_errors": self.arbitration_errors,
            "arbitration_error_details": list(self.arbitration_error_details),
        }

    def _vote_stats_dict(self) -> dict[str, dict[str, int | float | None]]:
        """把逐模型原始计数转成可跨 run 聚合的审计字段。"""
        out: dict[str, dict[str, int | float | None]] = {}
        for model, raw in sorted(self.judge_vote_stats.items()):
            votes = int(raw.get("votes", 0))
            errors = int(raw.get("errors", 0))
            successful = max(0, votes - errors)
            out[model] = {
                "votes": votes,
                "errors": errors,
                "covered_yes": int(raw.get("covered_yes", 0)),
                "covered_disagree": int(raw.get("covered_disagree", 0)),
                "attempted_yes": int(raw.get("attempted_yes", 0)),
                "attempted_disagree": int(raw.get("attempted_disagree", 0)),
                "error_rate": round(errors / votes, 4) if votes else None,
                "covered_yes_rate": (
                    round(int(raw.get("covered_yes", 0)) / successful, 4)
                    if successful else None),
                "covered_disagreement_rate": (
                    round(int(raw.get("covered_disagree", 0)) / successful, 4)
                    if successful else None),
                "attempted_yes_rate": (
                    round(int(raw.get("attempted_yes", 0)) / successful, 4)
                    if successful else None),
                "attempted_disagreement_rate": (
                    round(int(raw.get("attempted_disagree", 0)) / successful, 4)
                    if successful else None),
            }
        return out


def load_blockers(case_dir: Path) -> dict[str, str]:
    """从 gt.json 取「考点」作为 blocker 集合。

    三种书写形态，都支持：

        "澄清项·距离度量": {"澄清问题": "距离按 Haversine 算", "标签": "计算口径"}  # 现行
        "澄清项·距离度量": "距离按 Haversine 算"                    # 旧式纯描述
        "澄清项·距离度量": {"answer": "...", "triggers": ["...", ...]}  # 带参照问句

    现行形态（2026-08-25 起）是嵌套两层：考点名做键，下层 `澄清问题` 是答案正文、
    `标签` 是信息缺口分类。标签只做论文统计，judge 不看。

    triggers 是「一个好 agent 会怎么问」的示例，给 judge 一个**同型参照**——
    否则它只能拿提问去比答案（异型比对），判定全靠自由心证，这也是当初必须
    5 票投票才勉强稳住的原因（见模块头部关于稳定性的说明）。
    字段可选，缺失时行为与旧格式完全一致。
    """
    gt = case_dir / "gt.json"
    if not gt.is_file():
        return {}
    data = json.loads(gt.read_text(encoding="utf-8"))
    out: dict[str, str] = {}
    for k, v in data.items():
        if not k.startswith(("澄清项", "考点")):
            continue
        if isinstance(v, dict):
            out[k] = str(v.get("澄清问题") or v.get("answer") or "")
        else:
            out[k] = str(v)
    return out


def load_triggers(case_dir: Path) -> dict[str, list[str]]:
    """考点 → 参照问句列表。没标注的考点返回空列表。"""
    gt = case_dir / "gt.json"
    if not gt.is_file():
        return {}
    data = json.loads(gt.read_text(encoding="utf-8"))
    out: dict[str, list[str]] = {}
    for k, v in data.items():
        if not k.startswith(("澄清项", "考点")):
            continue
        raw = v.get("triggers") or [] if isinstance(v, dict) else []
        out[k] = [str(x).strip() for x in raw if str(x).strip()]
    return out


def load_tags(case_dir: Path) -> dict[str, str]:
    """澄清项 → 信息缺口标签（数据语义 / 评分口径 / 计算口径 / …）。

    标签是 gt.json 嵌套形态里的 `标签` 字段，judge 不看它——它只用来把覆盖率
    按信息类型拆开，回答「模型是在哪一类缺口上瞎」。未标注的返回空串，
    统计时归入 `未标注` 一档，不静默丢弃（丢了会让分母对不上 n_blockers）。
    """
    gt = case_dir / "gt.json"
    if not gt.is_file():
        return {}
    data = json.loads(gt.read_text(encoding="utf-8"))
    out: dict[str, str] = {}
    for k, v in data.items():
        if not k.startswith(("澄清项", "考点")):
            continue
        out[k] = str(v.get("标签", "")).strip() if isinstance(v, dict) else ""
    return out


def _triggers_block(items: list[str]) -> str:
    """渲染进 judge prompt 的参照段。无标注时返回空串（prompt 与旧版逐字相同）。"""
    if not items:
        return ""
    lines = "\n".join(f"  - {x}" for x in items)
    return ("\nQUESTIONS THAT WOULD SURFACE IT (reference examples, not exhaustive —\n"
            "a differently-worded question that targets the same fact also counts)\n"
            + lines + "\n")


def load_groups(case_dir: Path) -> list[list[str]]:
    """读 gt.json 的 `_考点分组`：哪些考点会被一个专业问题自然一起问到。

    考点粒度在各 case 间不齐——有的是彼此独立的数据坑，有的是同一套业务规则
    的几个面（如「距离度量」与「时长速度分段」共同决定行驶时长）。判定时若
    一律"一问只兑一个考点"，会系统性惩罚"成组澄清相关约定"这一专业行为。
    实测该标注用于判定时会适得其反：judge 依据它把跨组提问只记一个考点，
    反而低于不标注时的常识判断（recall 0.67→0.33）。故判定不再读取，
    此函数仅供分析阶段统计「agent 是否倾向成组澄清」。
    """
    gt = case_dir / "gt.json"
    if not gt.is_file():
        return []
    data = json.loads(gt.read_text(encoding="utf-8"))
    raw = data.get("_考点分组") or []
    return [[str(x) for x in g] for g in raw if isinstance(g, list) and len(g) > 1]


_QMARK = re.compile(r"[?？]")
_ATOMIC_SPLIT = re.compile(r"(?<=[?？])\s*")


def split_atomic(text: str) -> list[str]:
    """把一条 label 拆成原子问句。

    「一个问题」的边界本来由 agent 自己定，而 precision 的分母和 judge 的
    粒度判断都挂在这个单位上：三件事塞进一个 label，分母就是 1（稳拿高
    precision），judge 还要纠结"这算一个还是三个"，实测同一输入 recall 在
    0.667/1.0 之间摇摆。按问号切成原子问句后，分母与判定单位都变得客观。
    """
    parts = [p.strip() for p in _ATOMIC_SPLIT.split(text) if p.strip()]
    # 只保留真正的问句：按问号切分会把问号**之后**的陈述也切成独立条目
    # （实测一条 CF 记录里 15 条"原子问句"有 3 条是陈述——"该点纬度最高"、
    # "总客户重量 60903 kg"、"因为数据中只有经纬度坐标"）。这些进了 precision
    # 的分母却永远命中不了任何考点，等于凭空压低分数，还让 judge 去判一堆
    # 本不是问题的东西——三个 judge 的分歧正出在这里（recall 全 1.0，
    # 差异全在 precision）。
    asked = [p for p in parts if _QMARK.search(p)]
    if asked:
        return asked
    return parts or ([text.strip()] if text.strip() else [])


def load_questions(run_dir: Path) -> list[tuple[int, str]]:
    """从 clarify.json 取提问，返回 [(轮次, 原子问句)]。"""
    log = run_dir / "clarify.json"
    if not log.is_file():
        return []
    data = json.loads(log.read_text(encoding="utf-8"))
    out: list[tuple[int, str]] = []
    for rnd in data.get("rounds", []):
        turn = rnd.get("turn", 0)
        for q in rnd.get("questions", []):
            text = (q.get("label") or q.get("question") or "").strip()
            for atom in split_atomic(text):
                out.append((turn, atom))
    return out


# 澄清 judge 与产物抽取使用同一组冻结模型；从抽取器注册表直接复用，
# 避免两条链路日后增删模型时悄悄漂移。每个模型对每个考点判定一次。
from ..scoring.triple import _MODELS as JUDGE_MODELS

JUDGE_VOTES = len(JUDGE_MODELS)


def _judge_max_tokens() -> int:
    """judge 的输出预算。默认 4096，可由 config.toml [judge] max_tokens 覆盖。

    ⚠️ 不要往下调。判定必须以一段 JSON 结尾，而始终思考的模型（glm-5.3）会把
    分析过程写进 **content 正文**，预算不够就在写到 JSON 之前被截断，
    `re.search(r"{.*}")` 找不到东西 → ValueError → 该考点 5 票全废 →
    judge_errors 累加，而 recall 记 0。

    实测（003 案例，6 个考点 / 7 个提问，2026-08-25）：
        512   6 个考点里 5 个截断，recall=0.0     ← 事故现场
        1024  仍截断（回复 4321 字全是分析，无 JSON）
        2048  正常（回复 64 字纯 JSON，判定正确）
        4096  正常
    512 之所以曾经够用，是因为当时的辅助模型关得掉思维链；glm-5.3 关不掉。
    取 4096 而非 2048 是留余量——考点多、提问多的 case 分析更长。

    这个数只影响 judge 能不能把话说完，不影响判定口径，所以调它不破坏
    跨批次可比性（判定模型组仍需锁定）。
    """
    from ..config import get
    return int(get("judge", "max_tokens", 4096) or 4096)


def _parse_judge_json(reply: object, actor: str) -> dict:
    """Extract one JSON object from a judge response.

    Models sometimes wrap the answer in Markdown, add leading analysis, or put
    prose after the object.  A greedy ``{.*}`` regex joins multiple objects and
    causes avoidable ``JSONDecodeError`` failures, so parse the first complete
    object with ``raw_decode`` instead.  Non-object JSON remains an explicit
    protocol error and is never silently treated as a vote.
    """
    text = str(reply or "").strip()
    if not text:
        raise ValueError(f"{actor} returned no JSON")

    decoder = json.JSONDecoder()
    candidates = [
        m.group(1).strip()
        for m in re.finditer(r"```(?:json)?\s*(.*?)```", text,
                             flags=re.IGNORECASE | re.DOTALL)
        if m.group(1).strip()
    ]
    candidates.append(text)
    last_decode_error: json.JSONDecodeError | None = None

    for candidate in candidates:
        try:
            value = json.loads(candidate)
        except json.JSONDecodeError as exc:
            last_decode_error = exc
        else:
            if not isinstance(value, dict):
                raise ValueError(
                    f"{actor} returned {type(value).__name__}, expected JSON object")
            return value

        # Do not accept an object nested inside an array as the payload.
        if candidate.lstrip().startswith("["):
            try:
                value, _ = decoder.raw_decode(candidate.lstrip())
            except json.JSONDecodeError:
                pass
            else:
                if isinstance(value, list):
                    raise ValueError(
                        f"{actor} returned list, expected JSON object")

        # Leading/trailing prose is tolerated as a transport quirk.  Return the
        # first complete object and ignore text outside it.
        start = candidate.find("{")
        while start >= 0:
            try:
                value, _ = decoder.raw_decode(candidate, start)
            except json.JSONDecodeError as exc:
                last_decode_error = exc
            else:
                if isinstance(value, dict):
                    return value
            start = candidate.find("{", start + 1)

    if last_decode_error is not None:
        excerpt = _redact_judge_error(text[:200], limit=200)
        raise ValueError(
            f"{actor} returned invalid JSON object: {last_decode_error.msg}; "
            f"response_prefix={excerpt!r}")
    excerpt = _redact_judge_error(text[:200], limit=200)
    raise ValueError(f"{actor} returned no JSON object; response_prefix={excerpt!r}")


def _complete_judge_json(client, system: str, prompt: str,
                        actor: str) -> tuple[dict, int]:
    """Call a judge and retry once when its response violates JSON protocol."""
    attempts = 0
    try:
        attempts = 1
        reply = client.complete(
            system=system,
            messages=[{"role": "user", "content": prompt}],
            max_tokens=_judge_max_tokens(),
            temperature=0.0,
        )
        try:
            return _parse_judge_json(reply, actor), attempts
        except ValueError as first_error:
            # Do not echo the potentially huge/secret-bearing failed response.
            attempts = 2
            retry_prompt = (
                "Your previous response violated the JSON protocol "
                f"({type(first_error).__name__}). Return exactly one JSON object "
                "matching the required schema. Do not include analysis, Markdown, "
                "or any text before or after it.\n\n" + prompt
            )
            reply = client.complete(
                system=system,
                messages=[{"role": "user", "content": retry_prompt}],
                max_tokens=_judge_max_tokens(),
                temperature=0.0,
            )
            return _parse_judge_json(reply, actor), attempts
    except Exception as exc:
        try:
            setattr(exc, "judge_attempts", attempts or 1)
        except Exception:  # pragma: no cover - defensive for unusual exceptions
            pass
        raise


def _judge_call(client, kid: str, desc: str, questions: list[str],
                triggers: list[str] | None = None) -> dict:
    """单次判定。temperature=0 仍不确定，故由上层多次投票收敛。"""
    numbered = "\n".join(f"{i}. {q}" for i, q in enumerate(questions, 1))
    prompt = _PROMPT.format(kid=kid, desc=desc, questions=numbered,
                            triggers=_triggers_block(triggers or []))
    payload, _ = _complete_judge_json(client, _SYSTEM, prompt, "judge")
    return payload


def _arbitration_call(client, kid: str, desc: str, questions: list[str],
                      ballots: list[dict],
                      triggers: list[str] | None = None) -> dict:
    """Ask the explicitly configured arbiter to resolve non-unanimous votes."""
    numbered = "\n".join(f"{i}. {q}" for i, q in enumerate(questions, 1))
    ballot_text = json.dumps(ballots, ensure_ascii=False, indent=2)
    prompt = _ARBITRATION_PROMPT.format(
        kid=kid, desc=desc, questions=numbered,
        triggers=_triggers_block(triggers or []), ballots=ballot_text)
    payload, _ = _complete_judge_json(
        client, _ARBITRATION_SYSTEM, prompt, "arbitrator")
    return payload


def _judge_once(client, kid: str, desc: str, questions: list[str],
                triggers: list[str] | None = None,
                votes: int = JUDGE_VOTES, clients: list | None = None,
                arbitrator=None, arbitration_model: str | None = None) -> dict:
    """多模型各判一次并取多数。

    ``clients`` 传入生产环境的五个不同模型；未传时保留旧 API 语义，用同一个
    client 重复 ``votes`` 次，方便离线测试和旧调用方兼容。返回值附 `_agreement`
    记录多数派占比，供报告 judge 一致性；question_indices 取多数派各次的并集。
    """
    ballots, errors = [], 0
    active_clients = list(clients) if clients is not None else [client] * votes
    labels = []
    for i, active_client in enumerate(active_clients):
        label = getattr(active_client, "model", None)
        if not label and clients is not None and i < len(JUDGE_MODELS):
            label = JUDGE_MODELS[i][1]
        labels.append(str(label or f"judge_{i + 1}"))

    def run_one(active_client, label):
        try:
            return _judge_call(active_client, kid, desc, questions, triggers)
        except Exception as exc:
            return {
                "_error": {
                    "model": label,
                    "error_type": type(exc).__name__,
                    "error_message": _redact_judge_error(str(exc)),
                    "attempts": int(getattr(exc, "judge_attempts", 1) or 1),
                }
            }

    # 生产模式传入五个独立 client，可并行请求；注入单 client 的旧兼容模式保持
    # 串行，避免同一 SDK client 被多个线程同时初始化/复用连接。
    parallel = clients is not None and len(active_clients) > 1
    if parallel:
        from concurrent.futures import ThreadPoolExecutor
        with ThreadPoolExecutor(max_workers=min(5, len(active_clients)),
                                thread_name_prefix="clarify-judge") as pool:
            outcomes = list(pool.map(run_one, active_clients, labels))
    else:
        outcomes = [run_one(active_client, label)
                    for active_client, label in zip(active_clients, labels)]
    records = []
    for label, ballot in zip(labels, outcomes):
        if isinstance(ballot, dict) and ballot.get("_error"):
            errors += 1
            detail = dict(ballot["_error"])
            detail["model"] = label
            records.append({"model": label, "error": True, **detail})
        else:
            ballots.append(ballot)
            records.append({
                "model": label,
                "error": False,
                "attempted": bool(ballot.get("attempted")),
                "covered": bool(ballot.get("covered")),
            })
    if not ballots:
        exc = ValueError("judge 全部失败")
        try:
            setattr(exc, "judge_error_details", list(records))
        except Exception:  # pragma: no cover - defensive
            pass
        raise exc

    def majority(field: str) -> bool:
        yes = sum(1 for b in ballots if b.get(field))
        return yes * 2 > len(ballots)

    attempted = majority("attempted")
    covered = majority("covered")
    idx: set[int] = set()
    for b in ballots:
        if bool(b.get("covered")) == covered:
            idx |= {int(i) for i in b.get("question_indices", []) if str(i).isdigit()}
    agree = sum(1 for b in ballots if bool(b.get("covered")) == covered) / len(ballots)
    for record in records:
        if not record["error"]:
            record["covered_disagree"] = record["covered"] != covered
            record["attempted_disagree"] = record["attempted"] != attempted
    arbitration = {
        "enabled": arbitrator is not None,
        "model": arbitration_model,
        "triggered": False,
        "used": False,
        "error": None,
    }
    # A unanimous five-way result needs no extra model.  A partial failure is
    # also a trigger: the arbiter can resolve it while seeing the failed vote.
    disagreement = errors > 0 or len({bool(b.get("covered")) for b in ballots}) > 1
    if arbitrator is not None and disagreement:
        arbitration["triggered"] = True
        try:
            arb = _arbitration_call(
                arbitrator, kid, desc, questions, records, triggers)
            arbitration.update({
                "used": True,
                "attempted": bool(arb.get("attempted")),
                "covered": bool(arb.get("covered")),
                "question_indices": sorted({
                    int(i) for i in arb.get("question_indices", [])
                    if str(i).isdigit()}),
            })
            attempted = arbitration["attempted"]
            covered = arbitration["covered"]
            idx = set(arbitration["question_indices"])
        except Exception as exc:
            arbitration["error"] = {
                "error_type": type(exc).__name__,
                "error_message": _redact_judge_error(str(exc)),
                "attempts": int(getattr(exc, "judge_attempts", 1) or 1),
            }

    return {
        "attempted": attempted,
        "covered": covered,
        "question_indices": sorted(idx),
        "_agreement": round(agree, 3),
        "_votes": len(ballots),
        "_vote_errors": errors,
        "_ballots": records,
        "_arbitration": arbitration,
    }


def _task_advanced(run_dir: Path) -> bool | None:
    """CE-B 的推进判据：最终是否产出有效解（validity>0）。"""
    res = run_dir / "result.json"
    if not res.is_file():
        return None
    try:
        data = json.loads(res.read_text(encoding="utf-8"))
    except json.JSONDecodeError:
        return None
    v = data.get("validity_score", data.get("validity"))
    return bool(v and v > 0) if v is not None else None


def score_clarification(case_dir: Path, run_dir: Path, client=None,
                        model: str | None = None,
                        arbitration_model: str | None = None) -> dict:
    """评估一次 run 的澄清质量，结果写入 run_dir/clarify_score.json。

    ``arbitration_model`` is opt-in.  When set, that model is called only for a
    blocker whose five-way vote is non-unanimous or contains a failed ballot;
    the normal five-model vote remains the fallback if arbitration fails.
    """
    blockers = load_blockers(case_dir)
    triggers = load_triggers(case_dir)
    qs = load_questions(run_dir)
    score = ClarifyScore(n_blockers=len(blockers), n_questions=len(qs))
    score.tags = load_tags(case_dir)
    score.task_advanced = _task_advanced(run_dir)

    if not blockers:
        score.judge_errors = -1          # 该 case 未标注考点，无法评估
    elif qs:
        # judge 端点和 token 读自 config.toml 的 [judge] 段；模型组与抽取器
        # 冻结一致，每个模型对每个考点判定一次。
        judge_clients = None
        arbitrator = None
        if client is None:
            from dataclasses import replace
            from ..config import load_role, make_client
            role = load_role("judge")
            judge_clients = [make_client(replace(role, model=model_id))
                             for _, model_id in JUDGE_MODELS]
            score.judge_models = [model_id for _, model_id in JUDGE_MODELS]
            score.judge_model = ",".join(score.judge_models)
            if arbitration_model:
                arbitrator = make_client(
                    replace(role, model=arbitration_model))
        else:
            # 注入 client 是离线测试/兼容旧调用方的单模型模式。
            score.judge_models = [model or getattr(client, "model", "unknown")]
            score.judge_model = score.judge_models[0]
            if arbitration_model:
                # Test/compatibility injection: use the injected client rather
                # than silently constructing a second client without credentials.
                arbitrator = client
        score.arbitration_model = arbitration_model
        judge_labels = list(score.judge_models)
        texts = [q for _, q in qs]
        score.questions_judged = list(texts)
        first_turn = {i for i, (t, _) in enumerate(qs, 1) if t == 1}
        # 1-based 提问序号 → 它属于第几轮，用于 ATC
        turn_of = {i: t for i, (t, _) in enumerate(qs, 1)}

        for kid, desc in blockers.items():
            try:
                verdict = _judge_once(client, kid, desc, texts,
                                      triggers.get(kid), clients=judge_clients,
                                      arbitrator=arbitrator,
                                      arbitration_model=arbitration_model)
            except Exception as exc:
                score.judge_errors += 1
                # When every vote for this blocker fails, _judge_once attaches
                # the per-model diagnostics to the exception.  Keep them even
                # though no majority verdict exists.
                failed_details = getattr(exc, "judge_error_details", None)
                if isinstance(failed_details, list):
                    for detail in failed_details:
                        if not isinstance(detail, dict):
                            continue
                        score.judge_error_details.append({
                            "blocker": kid,
                            "model": str(detail.get("model") or "unknown"),
                            "error_type": str(detail.get("error_type") or "Error"),
                            "error_message": _redact_judge_error(
                                detail.get("error_message", "")),
                            "attempts": int(detail.get("attempts", 1) or 1),
                        })
                # 全部模型失败时没有 verdict 可记录，仍把这次失败计入每个模型，
                # 使跨 run 的 judge failure rate 不会漏掉全军覆没的考点。
                for label in judge_labels:
                    d = score.judge_vote_stats.setdefault(label, {
                        "votes": 0, "errors": 0, "covered_yes": 0,
                        "covered_disagree": 0, "attempted_yes": 0,
                        "attempted_disagree": 0,
                    })
                    d["votes"] += 1
                    d["errors"] += 1
                continue
            if "_agreement" in verdict:
                score.agreements.append(verdict["_agreement"])
            for ballot in verdict.get("_ballots", []):
                label = str(ballot.get("model") or "unknown")
                d = score.judge_vote_stats.setdefault(label, {
                    "votes": 0, "errors": 0, "covered_yes": 0,
                    "covered_disagree": 0, "attempted_yes": 0,
                    "attempted_disagree": 0,
                })
                d["votes"] += 1
                if ballot.get("error"):
                    d["errors"] += 1
                    score.judge_error_details.append({
                        "blocker": kid,
                        "model": label,
                        "error_type": str(ballot.get("error_type") or "Error"),
                        "error_message": _redact_judge_error(
                            ballot.get("error_message", "")),
                        "attempts": int(ballot.get("attempts", 1) or 1),
                    })
                else:
                    d["covered_yes"] += bool(ballot.get("covered"))
                    d["covered_disagree"] += bool(ballot.get("covered_disagree"))
                    d["attempted_yes"] += bool(ballot.get("attempted"))
                    d["attempted_disagree"] += bool(ballot.get("attempted_disagree"))
            score.audit.append({
                "blocker": kid,
                "attempted": bool(verdict.get("attempted")),
                "covered": bool(verdict.get("covered")),
                "matched_questions": [
                    texts[i - 1] for i in sorted(
                        {int(x) for x in verdict.get("question_indices", [])
                         if str(x).isdigit()})
                    if 1 <= i <= len(texts)
                ],
                "agreement": verdict.get("_agreement"),
                "vote_errors": verdict.get("_vote_errors", 0),
                # 保留逐模型投票（包括失败原因），使低一致度考点可定位到
                # 具体 Judge，而不改变旧字段或指标计算口径。
                "judge_ballots": verdict.get("_ballots", []),
                "arbitration": verdict.get("_arbitration", {
                    "enabled": False, "triggered": False, "used": False,
                    "error": None,
                }),
            })
            arb_info = verdict.get("_arbitration") or {}
            if arb_info.get("triggered"):
                score.arbitration_calls += 1
            if arb_info.get("used"):
                score.arbitration_used += 1
            if arb_info.get("error"):
                score.arbitration_errors += 1
                detail = dict(arb_info["error"])
                detail["blocker"] = kid
                detail["model"] = arbitration_model or "unknown"
                score.arbitration_error_details.append(detail)
            if verdict.get("attempted") or verdict.get("covered"):
                score.asked_blockers.append(kid)   # CE-A/CE-B 的分母
            if not verdict.get("covered"):
                continue
            score.covered.append(kid)
            idx = {int(i) for i in verdict.get("question_indices", []) if str(i).isdigit()}
            score.relevant_qs |= idx
            if idx & first_turn:
                score.first_turn_covered.append(kid)
            # ATC：取命中该考点的最早那一轮。judge 已返回 question_indices，
            # 所以这是零额外调用的——不必按轮次重复判定。
            turns = [turn_of[i] for i in idx if i in turn_of]
            if turns:
                score.covered_at_turn[kid] = min(turns)

    result = score.to_dict()
    (run_dir / "clarify_score.json").write_text(
        json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    return result
