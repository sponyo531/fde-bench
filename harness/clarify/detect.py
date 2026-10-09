"""判定「agent 这一轮是在提问，还是在交付」，并抽出原子问句。

为什么需要它
────────────
澄清的通用原生通道是**多轮对话本身**，不是某个 tool。实测（2026-09-02 复核）：

    Claude Code headless  25 个工具里不注册 AskUserQuestion
    Codex exec            二进制里搜不到 ask 类工具
    Gemini CLI yolo       ask_user 非交互下不暴露给模型
    Kimi -p               auto 模式，系统提示告知 AskUserQuestion 已禁用
    四者都在无任何格式提示下，用回复正文自然提问，并停在无产物状态

opencode serve 的 `question` tool 是**例外**（serve 面向 GUI 客户端才保留），
不是常态。因此框架不能只认结构化 tool 调用，否则除 opencode 外全都跑不了 C/CF。

反过来也说明：`<clarify>` 标记的机制（文本提问 + 回灌下一轮）本来是对的，
错在**强制格式**——要求模型学一个它平时不用的标记，测的就变成"会不会用这个
格式"。改成语义判定后，agent 怎么问都行，这才是脚手架的原生形态。

三级判定，先便宜的
────────────────
  0. 软标记：Interact* prompt **建议**把问题列在 `## Questions for the client`
     标题下。有就直接正则取——零 LLM 调用、边界由 agent 自己定。没有不惩罚，
     落到下面两级。这是它和已废弃的 <clarify> 强制标记的区别：那个是格式门槛，
     这个只是快捷通道。
  1. 无问号 → 直接判"不是提问"，不花 LLM 调用。
     （中英文问号都算；实测两家提问必带问号）
  2. 有问号 → LLM 抽原子问句。这一步不能省：报告正文里的设问
     （"那么如何平衡时长与均衡？我们采用…"）带问号但不是提问，
     正则区分不了，而误判会让框架把答案回灌给一个没在问的 agent。

有问号时使用与最终考点评分相同的五模型 judge 组并行判定，只有严格多数认为
是在向客户索取信息才会进入澄清回路；无问号仍然零调用短路。投票只决定
"是否在提问"；原子问句列表只取其中一份（数量居中者），不取并集——并集会把
同一问题的不同措辞全算成新问题（实测 12 问膨胀成 46 条）。

produced_delta 是强信号但不作硬判据：agent 可以边问边写脚本（此时有新文件
却仍在等答案），也可以问完就停（无新文件）。它只作为 LLM 判定的辅助上下文。
"""

from __future__ import annotations

import re
import sys

_QMARK = re.compile(r"[?？]")

_SYSTEM = """You judge whether a coding agent's reply is ASKING the client for \
information, or REPORTING/DELIVERING work.

Return JSON only:
{"asking": true|false, "questions": ["...", "..."]}

Rules:
- "asking" is true only if the agent is waiting for the client to supply
  information it cannot determine itself.
- Rhetorical questions inside an explanation or report are NOT asking.
  ("How do we balance time against fairness? We weight them 1:90." -> false)
- Offers to continue are NOT asking. ("Shall I also plot the routes?" -> false)
- Split compound questions into atomic ones, one fact per entry.
  ("What speed, and is loading time included?" -> two entries)
- Keep each question in its original language, close to the agent's wording.
- If "asking" is false, "questions" must be []."""


def _looks_like_question(text: str) -> bool:
    return bool(_QMARK.search(text))


# ── 软标记：`## Questions for the client` 区块 ────────────────────────────────
#
# Interact* 的 prompt **建议**（不强制）agent 把问题集中列在这个标题下。有就
# 直接正则取，零 LLM 调用、边界由 agent 自己定；没有就落回语义判定。不惩罚
# 不遵守的 agent——这是它和已废弃的 <clarify> 强制标记的本质区别。
#
# 标题故意写成 "Questions for the client" 而非泛泛的 "Questions"：交付报告里
# 常见 "Open questions" / "Questions for future work" 一类小节，泛匹配会把
# 一份正在交付的报告误判成提问、把答案回灌给没在问的 agent。
_MARK_HEADING = re.compile(
    r"^[ \t]*#{1,6}[ \t]*questions?\s+for\s+the\s+client[ \t:：]*$",
    re.IGNORECASE | re.MULTILINE,
)
_NEXT_HEADING = re.compile(r"^[ \t]*#{1,6}[ \t]+\S", re.MULTILINE)
_ITEM_PREFIX = re.compile(r"^\s*(?:[-*+•]|\d+[.)、]|[（(]\d+[)）]|[a-zA-Z][.)])\s*")


def extract_marked(text: str) -> list[str]:
    """从软标记区块里取问句；没有区块或区块为空返回 []。

    取标题之后、下一个 markdown 标题（或文末）之前的内容，按行切；去掉列表前缀
    （- / * / 1. / (1) / a.），丢掉空行。多个区块时全部合并（agent 偶尔分组）。
    不要求每行带问号——agent 用了标记就是在明确表示"这些是要问的"。
    """
    if not text:
        return []
    out: list[str] = []
    seen: set[str] = set()
    for m in _MARK_HEADING.finditer(text):
        start = m.end()
        nxt = _NEXT_HEADING.search(text, start)
        block = text[start: nxt.start() if nxt else len(text)]
        for line in block.splitlines():
            item = _ITEM_PREFIX.sub("", line, count=1).strip()
            # 去掉 markdown 强调符和句尾多余空白，保持与 agent 原话一致
            item = item.strip("*_ ").strip()
            if item and item not in seen:
                seen.add(item)
                out.append(item)
    return out


_ROLE = "judge"      # 复用 judge 的网关与 token，不新增配置段

# 与 clarify/score.py 的考点评分共用同一组冻结 judge 模型，避免文本澄清通路
# 用单一模型、最终评分却用五模型投票而产生协议漂移。
# make_client uses the full route to select the endpoint, then sends the bare model.
from ..scoring.triple import _MODELS as JUDGE_MODELS
from .score import _parse_judge_json


def _llm():
    """惰性建客户端：无问号的轮次根本不该触发配置检查。"""
    from ..config import load_role
    from ..llm_client import ChatLLM
    role = load_role(_ROLE)
    return ChatLLM(protocol=role.protocol, base_url=role.base_url,
                   auth_token=role.token, model=role.model), role.model


def _llms():
    """创建与考点评分相同的五个 judge client。

    返回 (opencode 形式的模型 ID, client)。ID 保留 provider 前缀作为标签
    （与 score.py 的 judge_models 记录一致），只有发给网关的那一份被剥掉。
    """
    from dataclasses import replace
    from ..config import load_role, make_client
    role = load_role(_ROLE)
    return [
        (model_id, make_client(replace(role, model=model_id)))
        for _, model_id in JUDGE_MODELS
    ]


def _clip(text: str, head: int = 2000, tail: int = 8000) -> str:
    """给 judge 看的正文截断。

    问号门禁看的是**全文**，judge 看的却曾是 text[:6000]——agent 的典型节奏是
    先长篇分析再列问题，分析一超过 6000 字，门禁放行、judge 却一个问题也
    看不到，五票齐投"没在提问"，问题在花完 5 次调用之后被丢。问题几乎总在
    末尾，所以保尾不保头：超长时留开头一段做语境、中间省略、末尾整段保留。
    """
    if len(text) <= head + tail:
        return text
    return text[:head] + "\n\n[... 中间省略 ...]\n\n" + text[-tail:]


def _detect_once(llm, model: str, text: str, hint: str) -> list[str]:
    """单模型提问判定；异常向上抛，由多数投票层统计为失败。"""
    # glm 系默认开 thinking，会把 max_tokens 全部耗在 reasoning 上导致 content=''
    extra = {"thinking": {"type": "disabled"}} if "glm" in model.lower() else None
    prompt = f"Agent reply:\n\n{_clip(text)}{hint}"
    attempts = 0
    try:
        attempts = 1
        raw = llm.complete(_SYSTEM, [{"role": "user", "content": prompt}],
                           max_tokens=2000, extra_body=extra)
        try:
            data = _parse_judge_json(raw, "clarify detector")
        except ValueError as first_error:
            # A malformed/truncated response is a protocol failure, not evidence
            # that the agent was not asking.  Retry once with a concise hard
            # instruction, without echoing the failed response into the prompt.
            retry = (
                "Your previous response violated the JSON protocol "
                f"({type(first_error).__name__}). Return exactly one JSON object "
                "with keys asking and questions, and no other text.\n\n" + prompt
            )
            attempts = 2
            raw = llm.complete(_SYSTEM, [{"role": "user", "content": retry}],
                               max_tokens=2000, extra_body=extra)
            data = _parse_judge_json(raw, "clarify detector")
        if not data.get("asking"):
            return []
        qs = data.get("questions") or []
        return [str(q).strip() for q in qs if str(q).strip()]
    except Exception as exc:
        # Preserve the retry count for detect's per-model failure audit.
        try:
            setattr(exc, "judge_attempts", attempts or 1)
        except Exception:  # pragma: no cover - defensive
            pass
        raise


def detect(text: str, *, produced_delta: list[str] | None = None) -> list[str]:
    """返回本轮的原子问句列表；空列表 = agent 不在提问。

    三级判定，先便宜的：
      1. 软标记区块 `## Questions for the client` → 直接取，零 LLM 调用
      2. 全文无 ?/？ → []，零 LLM 调用
      3. 语义判定：judge 多数投票（兜底）

    produced_delta 仅作为判定上下文传给 LLM（本轮新增的交付物文件名）。
    """
    if not text or not text.strip():
        return []
    marked = extract_marked(text)
    if marked:
        return marked
    if not _looks_like_question(text):
        return []

    hint = ""
    if produced_delta is not None:
        hint = (f"\n\nFiles the agent created this turn: "
                f"{produced_delta or '(none)'}")

    # 五模型并行判定。只有严格多数认为在提问时才回灌问题；少数模型误报不会
    # 污染文本澄清上下文。调用失败的模型从投票分母中剔除，但会在最终 judge
    # 一致性统计中由考点评分链路单独记录。
    clients = _llms()
    from concurrent.futures import ThreadPoolExecutor

    def run(item):
        model, llm = item
        try:
            return model, _detect_once(llm, model, text, hint), None
        except Exception as exc:                          # noqa: BLE001
            return model, None, f"{type(exc).__name__}: {str(exc)[:120]}"

    with ThreadPoolExecutor(max_workers=min(5, len(clients)),
                            thread_name_prefix="clarify-detect") as pool:
        outcomes = list(pool.map(run, clients))
    valid = [(model, qs) for model, qs, err in outcomes if err is None]
    if not valid:
        # 全军覆没不能静默：它就是"没在提问"的模样，而实际是基础设施故障。
        # 2026-09-02 之前 detect 因模型 ID 带前缀恒定五票 503，整个实验周期无人察觉。
        errs = "; ".join(f"{m}: {e}" for m, _, e in outcomes)
        print(f"  [clarify.detect] ⚠ 全部 {len(outcomes)} 个 judge 调用失败，"
              f"本轮按「没在提问」处理 —— {errs}", file=sys.stderr, flush=True)
        return []
    asking = [(model, qs) for model, qs in valid if qs]
    if len(asking) * 2 <= len(valid):
        return []
    return _pick_questions(asking)


def _pick_questions(asking: list[tuple[str, list[str]]]) -> list[str]:
    """从投「在提问」的 judge 里只取**一份**问句列表。

    不取并集：五个模型对同一段文字各自切分，措辞总有差异（"系数取多少？"
    vs "绕路系数取多少？"），按字符串去重一条也去不掉。实测一段含 12 个问题的
    Claude Code 回复，四票并集得 46 条——这 46 条会原样进 clarify.json，再进
    Ask-Precision 的分母，精度被凭空压掉近四倍。

    投票只回答"是否在提问"这一个布尔量；问句内容只需要一个可靠版本。取问句
    数量居中的那一份（偶数取偏少的一侧），既避开切得最碎的、也避开合并最狠的；
    数量相同则按 JUDGE_MODELS 的固定顺序取先者，保证同输入同输出。
    """
    ranked = sorted(asking, key=lambda mq: len(mq[1]))
    _, chosen = ranked[(len(ranked) - 1) // 2]
    out: list[str] = []
    seen: set[str] = set()
    for q in chosen:                 # 单模型内部仍可能重复，去一次
        if q not in seen:
            seen.add(q)
            out.append(q)
    return out


def _parse(raw: str) -> list[str]:
    """容错解析 judge 输出。坏 JSON 时保守判"没在提问"。

    保守方向的选择：误判"在提问"会让框架把答案回灌给一个正在交付的 agent，
    污染它的上下文并多烧一轮；误判"没在提问"只是少一轮澄清，且 CF 条件下
    agent 通常会再问一次。前者的破坏更大。
    """
    try:
        data = _parse_judge_json(raw, "clarify detector")
    except ValueError:
        return []
    if not data.get("asking"):
        return []
    qs = data.get("questions") or []
    return [str(q).strip() for q in qs if str(q).strip()]
