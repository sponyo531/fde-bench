"""考点泄漏检查：gt.json 里说要考的东西，是不是已经在 instruction.md 里白送了。

为什么需要这个脚本
────────────────
考点值一旦出现在 instruction.md 里，R 条件就白拿了它，C/CF 也没什么可问的，
`CF − R` 的落差被人为压平——而那个落差正是本 benchmark 要测的东西。

实测 03_city_delivery_route_planning 就有这个问题：gt.json 写着「数据里没给
这套速度参数，agent 需澄清」，而 instruction.md 直接写了 70/35/120。
57 个 case 靠人工比对不现实，故固化成脚本。

泄漏的实际后果（2026-08-14 实测）：Kimi 在 CF 条件下**一个问题都没问**，
却拿到 quality=0.9265（当时最高分）。查它的 NOTES.md，三个考点全部绕过了澄清——
速度分段直接引用客户原话「出仓/回仓的干线段 70 km/h」，载重引用「7125 公斤」，
仓库识别自己从数据推出「唯一没有重量的行」。也就是说这个 case 目前**测不出
澄清的价值**：该问的都不用问。CF 相对 C 的提升（0.3853 vs 0.2527）来自
answerer 补充的口径细节（均衡指标、地理集中度严格程度），而非 gt 认定的考点。

借鉴 SWE-RPG 的 GT 验证流程——它的 validation agent 明确检查
"absence of implementation detail leakage"（见 docs/paper/NOTE_SWE-RPG.md）。

只查考点，不查硬约束：硬约束（载重上限、车辆数）本就允许写进业务需求，
客户交代活儿时说清楚是自然的；考点才是"客户认为不言自明、因而没说"的东西。

    python3 -m harness.tools.leak_check                     # 查全部 case
    python3 -m harness.tools.leak_check --case-root ../x    # 指定根目录
"""

from __future__ import annotations

import argparse
import json
import re
from pathlib import Path

# 本文件在 harness/tools/ 下，仓库根要上跳三层
_ROOT = Path(__file__).resolve().parents[2]

_NUM = re.compile(r"\d+(?:\.\d+)?")

# 考点里数字后面跟着这些单位，它就是个卡点值——客户不说、agent 得问
_UNIT_HINT = ("km/h", "kg", "秒", "分钟", "米", "吨", "台", "辆", "%")

# 日期 / 时间跨度：客户交代活儿时本来就会说（"排接下来 72 小时，从 2026-06-15 起"），
# 出现在 instruction 里是任务设定而非泄漏。实测 49_port 因此整片误报。
_DATE_CTX = re.compile(r"\d{4}[-/年]|[-/]\d{1,2}[-/日]|\d+\s*(?:小时|天|周|月)内?")


def _date_spans(text: str) -> list[tuple[int, int]]:
    return [m.span() for m in _DATE_CTX.finditer(text)]


def significant_numbers(text: str) -> set[str]:
    """从考点描述里挑出「值得当成卡点」的数字。

    两道过滤，都是被误报逼出来的：
      1. 纯序号 0/1/2 —— "第 1 步"、"恰好 1 次" 满屏都是
      2. 日期与时间跨度 —— 属于任务设定，客户本来就会说
    留下的是带单位的量、或三位以上的具体数值。
    """
    spans = _date_spans(text)
    out: set[str] = set()
    for m in _NUM.finditer(text):
        val, start = m.group(), m.start()
        if val in ("0", "1", "2"):
            continue
        if any(s <= start < e for s, e in spans):
            continue
        tail = text[m.end():m.end() + 6]
        has_unit = any(u in tail for u in _UNIT_HINT)
        # 三位以上的具体数值（7125、120）即便没单位也算；两位数必须带单位
        if has_unit or len(val.replace(".", "")) >= 3:
            out.add(val)
    return out


def check_case(case: Path) -> list[dict]:
    """返回该 case 的泄漏项。空列表 = 干净。"""
    gt_path, inst_path = case / "gt.json", case / "instruction.md"
    if not gt_path.is_file() or not inst_path.is_file():
        return []
    try:
        gt = json.loads(gt_path.read_text(encoding="utf-8"))
    except json.JSONDecodeError:
        return [{"blocker": "(gt.json 解析失败)", "leaked": []}]
    inst = inst_path.read_text(encoding="utf-8")

    findings = []
    for key, desc in gt.items():
        if not key.startswith(("澄清项", "考点")):
            continue
        leaked = sorted(n for n in significant_numbers(str(desc)) if _appears(n, inst))
        if leaked:
            findings.append({"blocker": key, "leaked": leaked})
    return findings


def _appears(num: str, text: str) -> bool:
    """num 是否作为一个**完整的数**出现在 text 里。

    不能用 `num in text`：那是子串匹配，"3" 会命中 "43%蛋白"、"5" 会命中 "45"，
    于是产品牌号、百分比全成了泄漏。实测这一条就贡献了大半误报
    （44_soybean 的 43/45 是蛋白含量牌号，本就该写在需求里）。
    """
    return re.search(rf"(?<![\d.]){re.escape(num)}(?![\d.])", text) is not None


def main() -> None:
    p = argparse.ArgumentParser(description="考点泄漏检查")
    p.add_argument("--case-root", default=None,
                   help="case 根目录，默认读 experiments.toml 的 [defaults].case_root")
    args = p.parse_args()

    if args.case_root:
        root = Path(args.case_root).resolve()
    else:
        with open(_ROOT / "experiments.toml", "rb") as fh:
            cfg = __import__("tomllib").load(fh)
        root = (_ROOT / cfg["defaults"]["case_root"]).resolve()
    if not root.is_dir():
        raise SystemExit(f"case 根目录不存在: {root}")

    cases = sorted(d for d in root.iterdir() if d.is_dir())
    dirty = 0
    for case in cases:
        findings = check_case(case)
        if not findings:
            continue
        dirty += 1
        print(f"\n{case.name}")
        for f in findings:
            print(f"  {f['blocker']}  →  instruction.md 已含: {', '.join(f['leaked'])}")

    print(f"\n{'-' * 62}")
    print(f"扫描 {len(cases)} 个 case，{dirty} 个需人工复核")
    if dirty:
        print("这是**待复核清单**，不是判决。纯数字比对无法区分两种情况：")
        print("  真泄漏 —— 考点值被白送（03 的 7125 kg / 120 秒 / 12 辆）")
        print("  假阳性 —— 同一个数在需求里另有身份（44 的 43%/45% 是产品牌号，"
              "客户交代活儿时本就会说）")
        print("逐条看考点原文再定。真泄漏的修法：instruction.md 只留维度名词"
              "（「载重有上限」），数值移进 information.md。")


if __name__ == "__main__":
    main()
