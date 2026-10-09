"""信息条件 prompt 生成。

Hidden / Interact / Full 及其扩展条件共用同一份主模板
(``agents/prompt.md``)，只替换其中的 ``{{condition_block}}``。这样各条件
除了「给多少信息」和是否应答以外逐字相同，条件间的分数落差才能归因到
信息量本身，而不是措辞差异。

生成结果会随实验一起落盘（prompt.txt），供论文公开与复现核对。
"""

from __future__ import annotations

import json
import re
from dataclasses import dataclass
from pathlib import Path

# 主表四条件 + 消融条件。Full 系列按 information.md 的小节切分，用于把
# `Full − Hidden` 这个混合量拆开——它同时含业务背景/产物形式/数据说明/评估概要四类信息，
# 其中"产物形式"尤其要控制：Hidden 条件下 agent 并不知道该交什么格式。
# 条件名沿用 Ambig-SWE 的谱系（Hidden / Interaction / Full），让读者一眼
# 知道我们在哪个脉络里；强制档与信息切片是我们的扩展，加后缀区分。
CONDITIONS = (
    "Hidden",          # 只有客户原话，提问无人应答            （Ambig-SWE 同名）
    "Interact",        # 可提问，是否问由 agent 自定           （Ambig-SWE: Interaction）
    "Interact-Req",    # 强制问到够为止——不设此档 Interact 会退化成 Hidden
    "Full",            # information.md 全给，无人应答          （Ambig-SWE 同名）
    "Interact-Conf",   # E2：Interact-Req + 要求确认"自以为懂的"
    "Full-Base",       # E3：业务背景 + 产物形式
    "Full-Data",       # E3：Full-Base + 数据说明
    "Full-Rule",       # E3：Full-Base + 评估概要
)

# 旧内部名 → 新名。历史结果目录名里带旧条件名（results/**/…__CF__run1），
# 不做映射的话那些数据会读不出来、白跑。
_LEGACY = {
    "R": "Hidden", "C": "Interact", "CF": "Interact-Req", "F": "Full",
    "CF-confirm": "Interact-Conf",
    "F_base": "Full-Base", "F_data": "Full-Data", "F_rule": "Full-Rule",
}


# 可提问的三档——判断散落多处，抽成常量避免漏改（重命名时就漏过一次）
_INTERACT = ("Interact", "Interact-Req", "Interact-Conf")


def canonical(condition: str) -> str:
    """把旧名归一到新名；新名与 oracle 原样返回。"""
    return _LEGACY.get(condition, condition)

# E5（Oracle 部分澄清曲线）：oracle_k0 … oracle_kN，按 gt.json 的考点顺序注入
# 前 k 条答案，不给澄清渠道。k=0 的措辞不同于 Hidden，不复用 Hidden 分数。
# 用途见 docs/FRAMEWORK.md §12：画 考点覆盖率 → 交付质量 曲线，回答
# 「边际收益递减吗 / 哪个考点最值钱 / 部分澄清值不值」。
#
# 为什么不预枚举成固定档：各 case 的考点数不同（3–13 个），写死会漏。
_ORACLE_RE = re.compile(r"^oracle_k(\d+)$")


def is_oracle(condition: str) -> int | None:
    """oracle_k<N> → N；其余 → None。"""
    m = _ORACLE_RE.match(condition or "")
    return int(m.group(1)) if m else None


def valid_condition(condition: str) -> bool:
    from .oracle_coverage import parse_condition
    condition = canonical(condition)
    return (condition in CONDITIONS or is_oracle(condition) is not None
            or parse_condition(condition) is not None)


def oracle_conditions(case: Path) -> list[str]:
    """该 case 可用的全部 oracle 档（k=0…考点数）。"""
    n = len(_load_blockers(case))
    return [f"oracle_k{k}" for k in range(n + 1)]


def _load_blockers(case: Path) -> list[tuple[str, str]]:
    """从 gt.json 取考点，保持文件里的书写顺序。

    顺序即注入顺序——它决定曲线的形状，所以必须稳定、可复现，不能用 set。

    值有两种形态：现行是嵌套 dict（`澄清问题` 为答案正文，`标签` 只做论文统计、
    不注入），旧式是裸字符串。这里必须取 `澄清问题` 而不是 str(dict) ——
    否则注入给 agent 的会是 "{'澄清问题': ..., '标签': ...}" 这种字面量。
    """
    gt = case / "gt.json"
    if not gt.is_file():
        return []
    data = json.loads(gt.read_text(encoding="utf-8"))
    out: list[tuple[str, str]] = []
    for k, v in data.items():
        if not k.startswith(("澄清项", "考点")):
            continue
        if isinstance(v, dict):
            out.append((k, str(v.get("澄清问题") or v.get("answer") or "")))
        else:
            out.append((k, str(v)))
    return out

# information.md 的小节 → 条件可见的部分。实测 56 个 case 结构规整：
#   业务背景 56/56、产物形式 56/56、评估概要 56/56、数据说明 52/56、注意事项 49/56
_F_SECTIONS: dict[str, tuple[str, ...]] = {
    "Full-Base": ("业务背景", "产物形式"),
    "Full-Data": ("业务背景", "产物形式", "数据说明"),
    "Full-Rule": ("业务背景", "产物形式", "评估概要"),
}


def _format_oracle(items: list[tuple[str, str]]) -> str:
    """把考点渲染成「客户已经交代过的口径」。

    刻意不带考点名（`澄清项·时长速度分段` 这种内部标签），只留内容——
    agent 不该知道这些正是评分要考的点，否则等于泄漏评分口径。
    """
    if not items:
        return "(none)"
    return "\n".join(f"- {desc.strip()}" for _, desc in items)


def _slice_information(text: str, keep: tuple[str, ...]) -> str:
    """按 ## 小节切 information.md，只保留 keep 里的小节。

    缺失的小节静默跳过（如 4 个 case 无「数据说明」），由 build_prompt 记录。
    """
    parts = re.split(r"^##\s+", text, flags=re.M)
    out = []
    for blk in parts[1:]:
        head = blk.split("\n", 1)[0].strip()
        if head in keep:
            out.append("## " + blk.rstrip())
    return "\n\n".join(out)

_ROOT = Path(__file__).resolve().parent.parent
_AGENTS = _ROOT / "agents"


@dataclass(frozen=True)
class ClarifyLimits:
    """Interact 条件的提问预算。

    两者都只作兜底，不作约束：prompt 里不告知任何上限，暗中截断会让 recall
    被系统性压低而 agent 毫不知情（实测一轮问 10 个被丢 6 个）。过度提问由
    Ask-F1 的 precision 项惩罚。
    """
    max_rounds: int = 30
    max_questions: int = 0       # 0 = 不限


def _render(template: str, values: dict[str, str]) -> str:
    out = template
    for key, val in values.items():
        out = out.replace("{{" + key + "}}", val)
    return out


def _data_files(case: Path) -> str:
    data_dir = case / "data"
    if not data_dir.is_dir():
        return "(none)"
    names = sorted(p.name for p in data_dir.iterdir() if p.is_file())
    return "\n".join(f"- data/{n}" for n in names) if names else "(none)"


def build_prompt(
    case: Path,
    condition: str,
    workspace: Path,
    limits: ClarifyLimits | None = None,
    *, run_index: int = 1,
) -> str:
    """(case, 条件) → 完整 prompt。

    条件块的来源：
      Hidden — agents/conditions/Hidden.md，不提任何提问渠道
      Interact — agents/conditions/Interact.md，是否提问由 agent 自定
      Interact-Req — agents/conditions/Interact-Req.md，强制"问到够为止"。
           自主档实测 0 轮提问（与 Ambig-SWE "Without compulsory interaction,
           the model defaults to non-interactive behavior" 一致），
           不设强制档则 Interact 会退化成 Hidden
      Full — agents/conditions/Full.md，内联 case 的 information.md 全文
    """
    condition = canonical(condition)          # 兼容旧名（R/C/CF/F…）
    from .oracle_coverage import load_payload, parse_condition
    coverage = parse_condition(condition)
    k = is_oracle(condition)
    if not valid_condition(condition):
        raise ValueError(
            f"unknown condition {condition!r}; expected one of {CONDITIONS} "
            "or oracle_k<N> / oracle_cov_v1_p<P>_s<S>"
        )
    limits = limits or ClarifyLimits()
    info_path = case / "information.md"
    info_text = info_path.read_text(encoding="utf-8") if info_path.is_file() else ""
    if condition in _F_SECTIONS:
        info_text = _slice_information(info_text, _F_SECTIONS[condition])

    if coverage is not None:
        payload = load_payload(case, condition, run_index)
        block = _render(
            (_AGENTS / "conditions" / "oracle.md").read_text(encoding="utf-8").strip(),
            {"answers": _format_oracle([("", answer) for answer in payload["answers"]])},
        )
    elif k is not None:
        blockers = _load_blockers(case)
        if not blockers:
            raise FileNotFoundError(
                f"condition {condition} needs 考点 entries in {case}/gt.json"
            )
        if k > len(blockers):
            raise ValueError(
                f"{condition}: 该 case 只有 {len(blockers)} 个考点，k={k} 越界"
            )
        block = _render(
            (_AGENTS / "conditions" / "oracle.md").read_text(encoding="utf-8").strip(),
            {"answers": _format_oracle(blockers[:k])},
        )
    else:
        block = (_AGENTS / "conditions" / f"{condition}.md").read_text(encoding="utf-8").strip()

    if k is not None or coverage is not None:
        pass                                  # oracle 的 block 已渲染完
    elif condition in _INTERACT:
        block = _render(block, {
            "max_rounds": str(limits.max_rounds),
            "max_questions": str(limits.max_questions),
        })
    elif condition == "Full" or condition in _F_SECTIONS:
        if not info_text.strip():
            raise FileNotFoundError(
                f"condition {condition} needs information.md in {case}; "
                "without it this condition is identical to Hidden"
            )
        block = _render(block, {"information": info_text.strip()})

    # instruction.md 对所有条件无条件注入：information.md 是 instruction 的**补集**
    # （Full.md 原话 "details that the request itself leaves implicit"），不含客户诉求本身。
    request_block = (
        "## The client's request\n\n"
        + (case / "instruction.md").read_text(encoding="utf-8").strip()
    )

    template = (_AGENTS / "prompt.md").read_text(encoding="utf-8")
    return _render(template, {
        "workspace": str(workspace),
        "data_files": _data_files(case),
        "request_block": request_block,
        "condition_block": block,
    }).strip() + "\n"
