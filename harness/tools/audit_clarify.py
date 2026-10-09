"""澄清评分的体检：把静默偏差变成可见的报警。

澄清指标全靠 LLM judge，错了不会崩、只会让分数悄悄偏移。已经踩过两次：

  1. 按问号切分时，问号**之后**的陈述也被切成"原子问句"，进了 precision
     的分母却永远命中不了考点。实测历史数据 78 条片段里 6 条如此（7.7%），
     三个 judge 的 Ask-F1 分歧（0.50~0.64）主要就来自这里。
  2. judge 调用全部失败时 recall 算成 0.0，与"一个都没问到"无法区分——
     基础设施故障被伪装成 agent 表现差。

这两类都不会报错。故提供一个主动扫描：跑完实验先过一遍，再看指标。

    python3 -m harness.tools.audit_clarify results/E1
"""

from __future__ import annotations

import argparse
import json
import re
from pathlib import Path

_QMARK = re.compile(r"[?？]")
_SPLIT = re.compile(r"(?<=[?？])\s*")


def _load(p: Path) -> dict:
    try:
        return json.loads(p.read_text(encoding="utf-8"))
    except Exception:
        return {}


def audit(root: Path) -> dict:
    stats = {
        "runs": 0, "with_clarify": 0, "scored": 0,
        "judge_failed": [], "no_audit": [],
        "non_question_fragments": 0, "total_fragments": 0,
        "zero_recall_but_asked": [], "samples": [],
    }
    for run in sorted(root.rglob("clarify.json")):
        d = run.parent
        stats["runs"] += 1
        cj = _load(run)
        if not cj.get("rounds"):
            continue
        stats["with_clarify"] += 1

        # ① 切分质量：陈述句混进提问。**必须与评分同一把刀**（score.split_atomic 会
        # 丢掉无问号的陈述片段）；之前这里自己按问号硬切，把评分早已丢弃的
        # 括注/解释句也数进来，报出 8% 的假警。
        from ..clarify.score import split_atomic
        for r in cj["rounds"]:
            for q in r.get("questions", []):
                txt = (q.get("question") or q.get("label") or "").strip()
                for frag in split_atomic(txt):
                    stats["total_fragments"] += 1
                    if not _QMARK.search(frag):
                        stats["non_question_fragments"] += 1
                        if len(stats["samples"]) < 8:
                            stats["samples"].append(frag[:70])

        sj = _load(d / "clarify_score.json")
        if not sj:
            continue
        stats["scored"] += 1

        # ② judge 全败
        if sj.get("judge_failed"):
            stats["judge_failed"].append(d.name)
        # ③ 判定留痕缺失（旧版本产的结果，无法审计）
        if "audit" not in sj:
            stats["no_audit"].append(d.name)
        # ④ 问了却 recall=0：可能真没问到，也可能 judge 判偏了，值得人工看
        if (sj.get("recall") == 0.0 and (sj.get("n_questions") or 0) > 0
                and not sj.get("judge_failed")):
            stats["zero_recall_but_asked"].append(d.name)
    return stats


def main() -> None:
    p = argparse.ArgumentParser(description="澄清评分体检")
    p.add_argument("results", type=Path, nargs="+")
    args = p.parse_args()

    total = {"runs": 0, "with_clarify": 0, "scored": 0,
             "non_question_fragments": 0, "total_fragments": 0}
    flags = {"judge_failed": [], "no_audit": [], "zero_recall_but_asked": [],
             "samples": []}
    for root in args.results:
        s = audit(root)
        for k in total:
            total[k] += s[k]
        for k in flags:
            flags[k] += s[k]

    print(f"扫描 {total['runs']} 个 run，{total['with_clarify']} 个有澄清记录，"
          f"{total['scored']} 个已评分\n")

    frag, nonq = total["total_fragments"], total["non_question_fragments"]
    pct = 100 * nonq / max(frag, 1)
    mark = "⚠️" if pct > 1 else "✅"
    print(f"{mark} 切分质量：{frag} 个片段中 {nonq} 个无问号（{pct:.1f}%）")
    if nonq:
        print("   这些会进 precision 的分母却命中不了考点，压低分数：")
        for s in flags["samples"][:5]:
            print(f"     · {s}")

    for key, desc, why in [
        ("judge_failed", "judge 全败", "指标为 None，须重跑而非当成低分"),
        ("no_audit", "缺判定留痕", "旧版本产出，无法核对分数来源，建议重评"),
        ("zero_recall_but_asked", "问了但 recall=0", "值得人工抽检判定是否合理"),
    ]:
        n = len(flags[key])
        print(f"{'⚠️' if n else '✅'} {desc}：{n} 个 run" + (f"（{why}）" if n else ""))
        for name in flags[key][:3]:
            print(f"     · {name}")


if __name__ == "__main__":
    main()
