"""自动失败归因与人工抽样校准。

归因分两层：``infer_failure_label`` 只使用 run 已落盘的状态、结果和
错误信息做保守的规则预标注；``make_review_sample`` 生成待人工复核的
分层样本。人工复核写入 ``reviewed_*`` 字段后，``calibration_report``
自动计算自动标签与人工标签的一致率。

自动标签永远写到 ``failure_stage.auto.json``，不覆盖人工维护的
``failure_stage.json``。这使得同一批结果可以先全量统计，再逐步校准。
"""

from __future__ import annotations

import argparse
import hashlib
import json
import random
from pathlib import Path
from typing import Iterable

from .metrics import _number


def _read(path: Path) -> dict:
    try:
        return json.loads(path.read_text(encoding="utf-8")) if path.is_file() else {}
    except (OSError, ValueError, json.JSONDecodeError):
        return {}


def infer_failure_label(run_dir: Path) -> dict | None:
    """从已有账本保守推断一次 run 的失败标签。

    这是可审计的预标注而非声称理解了 Agent 的全部内部意图：证据不足时
    使用 ``unknown``，并把 ``needs_review`` 置为 true。有效 run 不生成标签。
    """
    use = _read(run_dir / "usage.json")
    res = _read(run_dir / "result.json")
    status = str(use.get("status") or "").lower()
    validity = _number(res.get("validity_score", res.get("validity")))
    score = _number(res.get("combined_score", res.get("overall_score")))
    produced = use.get("produced") or []
    killed = str(use.get("killed_reason") or "")
    err = res.get("error_info") or {}
    text = json.dumps(err, ensure_ascii=False).lower()

    # 无评分的运行不属于 FFR 候选，但仍可自动标为运行故障供 operational
    # attribution 使用；这里返回的标签不会被 failure_attribution 纳入。
    if status in {"error", "degraded"}:
        subtype = "provider_or_runtime_error" if status == "error" else "no_output_degraded"
        recorded_reason = str(use.get("failure_reason") or "")
        if killed or recorded_reason:
            subtype = killed or recorded_reason
        return {
            "failure_scope": "infrastructure",
            "failure_stage": "execution",
            "failure_subtype": subtype,
            "failure_reason": recorded_reason or f"自动预标注：status={status}",
            "evidence": {"status": status, "killed_reason": killed,
                         "failure_reason": recorded_reason, "produced": produced},
            "label_source": "auto_rule_v1",
            "confidence": 0.98 if status == "degraded" else 0.85,
            "needs_review": False if status == "degraded" else True,
        }

    if status == "timeout" and validity is None:
        return {
            "failure_scope": "infrastructure",
            "failure_stage": "execution",
            "failure_subtype": killed or "timeout_unscored",
            "failure_reason": "自动预标注：超时且没有 evaluator 分数",
            "evidence": {"status": status, "killed_reason": killed,
                         "produced": produced},
            "label_source": "auto_rule_v1", "confidence": 0.9,
            "needs_review": True,
        }

    if validity is None or validity > 0:
        return None

    # 有 evaluator 结果的 0 分才是 FFR 候选。缺文件/格式错误优先归 artifact；
    # 约束违规只凭现有证据归 evaluator，避免把结果层证据冒充成建模阶段事实。
    missing = any(k in text for k in (
        "缺 solution", "missing", "no solution", "unknown artifact",
        "schema", "json 解析", "json parse", "required file"))
    if missing:
        stage, subtype, confidence = "artifact", "missing_or_invalid_artifact", 0.9
    elif err or score is not None:
        stage, subtype, confidence = "evaluator", "evaluator_hard_violation", 0.65
    else:
        stage, subtype, confidence = "unknown", "unclassified_zero_validity", 0.2
    return {
        "failure_scope": "agent",
        "failure_stage": stage,
        "failure_subtype": subtype,
        "failure_reason": "自动预标注：evaluator validity<=0",
        "evidence": {"status": status, "validity": validity,
                     "error_info": err, "produced": produced},
        "label_source": "auto_rule_v1",
        "confidence": confidence,
        "needs_review": confidence < 0.9,
    }


def write_auto_label(run_dir: Path, *, overwrite: bool = False) -> Path | None:
    label = infer_failure_label(run_dir)
    if label is None:
        return None
    out = run_dir / "failure_stage.auto.json"
    if out.exists() and not overwrite:
        return out
    out.write_text(json.dumps(label, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    return out


def _iter_runs(roots: Iterable[str | Path]):
    from ..results import discover_run_dirs
    return discover_run_dirs(roots)


def make_review_sample(roots: Iterable[str | Path], *, fraction: float = 0.1,
                       seed: int = 0, min_per_stratum: int = 1) -> list[dict]:
    """生成按 ``stage × scope`` 分层的确定性人工复核样本。"""
    if not 0 <= fraction <= 1:
        raise ValueError("fraction 必须在 [0,1] 内")
    rows = []
    for run in _iter_runs(roots):
        label = infer_failure_label(run)
        if label is None:
            continue
        key = (label["failure_scope"], label["failure_stage"])
        rows.append({"run_dir": str(run), "run_id": _read(run / "manifest.json").get("run_id", run.name),
                     "stratum": {"scope": key[0], "stage": key[1]},
                     "auto": label,
                     "reviewed_failure_scope": None,
                     "reviewed_failure_stage": None,
                     "reviewed_failure_subtype": None,
                     "reviewer": None,
                     "review_notes": None})
    groups = {}
    for row in rows:
        groups.setdefault((row["stratum"]["scope"], row["stratum"]["stage"]), []).append(row)
    rng = random.Random(seed)
    sample = []
    for key, group in sorted(groups.items()):
        group = sorted(group, key=lambda x: hashlib.sha256(x["run_id"].encode()).hexdigest())
        n = min(len(group), max(min_per_stratum, round(len(group) * fraction)))
        # seed affects the tie-breaking order without making the sample depend on
        # filesystem traversal order.
        chosen = group[:n]
        rng.shuffle(chosen)
        sample.extend(chosen)
    return sorted(sample, key=lambda x: x["run_id"])


def calibration_report(sample: Iterable[dict]) -> dict:
    """比较已填写人工字段与自动标签；未填写的样本不计入分母。"""
    sample = list(sample)
    total = complete = 0
    fields = ("failure_scope", "failure_stage", "failure_subtype")
    matches = {f: 0 for f in fields}
    for row in sample:
        auto = row.get("auto") or {}
        if not all(row.get("reviewed_" + f) not in (None, "") for f in fields):
            continue
        complete += 1
        for f in fields:
            total += 1
            if row.get("reviewed_" + f) == auto.get(f):
                matches[f] += 1
    return {
        "n_sampled": len(sample),
        "n_reviewed": complete,
        "agreement": {f: (matches[f] / complete if complete else None) for f in fields},
        "n_comparisons": total,
        "label_source": "auto_rule_v1_vs_manual_review",
    }


def apply_reviewed_labels(sample: Iterable[dict], *, overwrite: bool = False) -> int:
    """将已完整填写的抽样复核结果提升为人工 sidecar。

    默认不覆盖已有 ``failure_stage.json``；需要修订时显式传 ``overwrite``。
    """
    n = 0
    for row in sample:
        fields = ("failure_scope", "failure_stage", "failure_subtype")
        if not all(row.get("reviewed_" + f) not in (None, "") for f in fields):
            continue
        run = Path(row["run_dir"])
        target = run / "failure_stage.json"
        if target.exists() and not overwrite:
            continue
        payload = {"failure_scope": row["reviewed_failure_scope"],
                   "failure_stage": row["reviewed_failure_stage"],
                   "failure_subtype": row["reviewed_failure_subtype"],
                   "failure_reason": row.get("review_notes") or "人工抽样复核",
                   "evidence": (row.get("auto") or {}).get("evidence"),
                   "label_source": "manual_review",
                   "reviewer": row.get("reviewer"),
                   "auto_label": row.get("auto")}
        target.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
        n += 1
    return n


def main() -> None:
    p = argparse.ArgumentParser(description="全量自动失败预标注 + 人工抽样校准")
    p.add_argument("results", nargs="+", type=Path)
    p.add_argument("--write", action="store_true", help="在每个失败 run 写 failure_stage.auto.json")
    p.add_argument("--sample-out", type=Path, help="写人工复核 JSON 文件")
    p.add_argument("--fraction", type=float, default=0.1)
    p.add_argument("--min-per-stratum", type=int, default=1)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--calibration", type=Path, help="读取复核 JSON 并输出一致率")
    p.add_argument("--apply-reviewed", type=Path,
                   help="将已完整填写的复核标签写入 failure_stage.json")
    p.add_argument("--overwrite", action="store_true", help="允许覆盖已有人工 sidecar")
    a = p.parse_args()
    if a.write:
        n = sum(write_auto_label(r, overwrite=True) is not None for r in _iter_runs(a.results))
        print(f"自动预标注 {n} 个 run")
    if a.sample_out:
        sample = make_review_sample(a.results, fraction=a.fraction,
                                    seed=a.seed, min_per_stratum=a.min_per_stratum)
        a.sample_out.parent.mkdir(parents=True, exist_ok=True)
        a.sample_out.write_text(json.dumps(sample, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
        print(f"写出人工复核样本 {len(sample)} 条：{a.sample_out}")
    if a.calibration:
        sample = json.loads(a.calibration.read_text(encoding="utf-8"))
        print(json.dumps(calibration_report(sample), ensure_ascii=False, indent=2))
    if a.apply_reviewed:
        sample = json.loads(a.apply_reviewed.read_text(encoding="utf-8"))
        print(f"写入人工 sidecar {apply_reviewed_labels(sample, overwrite=a.overwrite)} 个 run")


if __name__ == "__main__":
    main()
