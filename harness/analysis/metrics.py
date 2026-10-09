"""基于已有 run 产物的 case-level 统计指标。

该模块只读取 ``manifest.json``、``usage.json`` 和 ``result.json``，不重新调用
Agent 或评分模型。重复运行先在 ``case × scaffold × model × condition`` 内求
均值，再进行 case-level 配对比较，避免把同一个 case 的重复运行当成独立任务。

示例::

    python -m harness.analysis.metrics results/E1 \\
      --thresholds docs/thresholds.json \\
      --pair Full Interact --pair Interact Hidden \\
      --json results/E1/metrics.json

``--thresholds`` 是可选的；没有它时 SuccessRate/pass@k/pass^k 直接使用当前
evaluator 的 0/1 可行性（``validity_score > 0``）。传入它才会额外叠加质量门槛。
"""

from __future__ import annotations

import argparse
import json
import math
import random
import statistics
from collections import defaultdict
from pathlib import Path
from typing import Iterable, Sequence

EXCLUDED_STATUS = {"error", "degraded"}

# 失败阶段是一个可扩展但必须冻结的枚举。新标签可以增加，但已有标签不能
# 改名，否则不同批次的 FFR/SAL/LossShare 将无法合并。
FAILURE_STAGES = (
    "clarification",  # 需求理解 / 澄清
    "data",           # 数据读取、字段或单位检查
    "modeling",       # 业务规则、目标或数学建模
    "execution",      # 代码执行、求解器运行、超时
    "artifact",       # 产物格式、缺文件或交付不完整
    "evaluator",      # evaluator 硬约束或质量校验
)

# ``failure_stage`` 表示交付链路中最早可定位的环节；``failure_scope`` 则表明
# 根因应归属谁。两者不可混为一谈：例如 provider 无响应不是 agent 的
# ``execution`` 失败，case 自身不可行也不应计为 agent 的 ``modeling`` 失败。
# 该集合可以扩展；已有名称不得改名，以保证跨批次统计可合并。
FAILURE_SCOPES = (
    "agent",           # Agent 的需求理解、数据、建模、执行或交付问题
    "interaction",     # user simulator / 澄清协议等交互层故障
    "infrastructure",  # provider、容器、网络、脚手架或评分服务故障
    "benchmark",       # case 规格/数据矛盾，或 evaluator 本身的问题
)

_CONDITION_ALIASES = {
    "R": "Hidden", "C": "Interact", "CF": "Interact-Req", "F": "Full",
    "CF-confirm": "Interact-Conf", "F_base": "Full-Base",
    "F_data": "Full-Data", "F_rule": "Full-Rule",
}


def _read(path: Path) -> dict:
    try:
        return json.loads(path.read_text(encoding="utf-8")) if path.is_file() else {}
    except (OSError, json.JSONDecodeError):
        return {}


def _number(value):
    return value if isinstance(value, (int, float)) and math.isfinite(value) else None


def _case(manifest: dict, run_dir: Path) -> str | None:
    # source_case 兼容未来的新 manifest；当前矩阵运行的 case 已是完整 clean 名。
    return manifest.get("source_case") or manifest.get("case") or run_dir.name.split("__", 1)[0]


def _canonical_condition(condition):
    return _CONDITION_ALIASES.get(condition, condition)


def _quality(result: dict):
    return _number(result.get("quality_score", result.get("quality")))


def load_run(run_dir: Path) -> dict:
    """把一个 run 目录读成分析记录。"""
    man, use, res = (_read(run_dir / n) for n in ("manifest.json", "usage.json", "result.json"))
    from ..results import derive_run_statuses
    derived = derive_run_statuses(run_dir)
    # 人工 sidecar 优先；没有人工复核时自动使用规则预标注。自动标签由
    # ``harness.analysis.attribution`` 从 status/result/usage 证据推断，不会
    # 覆盖原始账本，且带 label_source/confidence 供抽样校准。
    failure = _read(run_dir / "failure_stage.json")
    if not failure:
        failure = _read(run_dir / "failure_stage.auto.json")
    if not failure:
        from .attribution import infer_failure_label
        failure = infer_failure_label(run_dir) or {}
    score = _number(res.get("combined_score", res.get("overall_score")))
    validity = _number(res.get("validity_score", res.get("validity")))
    return {
        "run_id": man.get("run_id", run_dir.name),
        "case": _case(man, run_dir),
        "source_case": man.get("source_case"),
        "condition": _canonical_condition(man.get("condition")),
        "scaffold": man.get("scaffold") or "unknown",
        "model": (man.get("agent_env") or {}).get("model") or man.get("model") or "unknown",
        "run_index": man.get("run_index"),
        "status": use.get("status"),
        "run_status": use.get("run_status") or derived["run_status"],
        "artifact_status": use.get("artifact_status") or derived["artifact_status"],
        "scoring_status": use.get("scoring_status") or derived["scoring_status"],
        "validity_status": use.get("validity_status") or derived["validity_status"],
        "score": score,
        "validity": validity,
        "quality": _quality(res),
        # 阶段标注可放在 sidecar（推荐）或 usage/manifest 中，兼容手工标注和
        # 运行时自动分类。sidecar 优先，避免修改原始评分账本。
        "failure_stage": (failure.get("failure_stage")
                           or use.get("failure_stage")
                           or man.get("failure_stage")),
        "failure_scope": (failure.get("failure_scope")
                          or use.get("failure_scope")
                          or man.get("failure_scope")),
        "failure_subtype": (failure.get("failure_subtype")
                             or use.get("failure_subtype")
                             or man.get("failure_subtype")),
        "failure_reason": (failure.get("failure_reason")
                            or use.get("failure_reason")
                            or man.get("failure_reason")),
        "failure_evidence": (failure.get("evidence")
                              or use.get("failure_evidence")
                              or man.get("failure_evidence")),
        "failure_label_source": (failure.get("label_source")
                                  or use.get("failure_label_source")
                                  or man.get("failure_label_source")),
        "failure_label_confidence": failure.get("confidence"),
        "failure_needs_review": failure.get("needs_review"),
        "killed_reason": use.get("killed_reason"),
    }


def load_runs(results_roots: Iterable[str | Path]) -> list[dict]:
    """扫描直接或 batch 布局，并按 run_id 保留最新一次重试。"""
    from ..results import discover_run_dirs
    return [load_run(d) for d in discover_run_dirs(results_roots)]


def _usable(rows: Iterable[dict]) -> list[dict]:
    return [r for r in rows if r.get("status") not in EXCLUDED_STATUS
            and _number(r.get("score")) is not None]


def _system_key(row: dict) -> tuple[str, str]:
    return str(row.get("scaffold") or "unknown"), str(row.get("model") or "unknown")


def _mean(values: Sequence[float]) -> float | None:
    return statistics.fmean(values) if values else None


def _median(values: Sequence[float]) -> float | None:
    return statistics.median(values) if values else None


def _percentile(values: Sequence[float], q: float) -> float:
    ordered = sorted(values)
    if len(ordered) == 1:
        return ordered[0]
    pos = (len(ordered) - 1) * q
    lo, hi = math.floor(pos), math.ceil(pos)
    if lo == hi:
        return ordered[lo]
    return ordered[lo] + (ordered[hi] - ordered[lo]) * (pos - lo)


def bootstrap_mean(values: Sequence[float], *, n_resamples: int = 10_000,
                   seed: int = 0, confidence: float = 0.95) -> dict | None:
    """对 case-level 数值做 percentile bootstrap CI。"""
    vals = list(values)
    if not vals:
        return None
    if n_resamples <= 0:
        raise ValueError("n_resamples 必须为正数")
    rng = random.Random(seed)
    n = len(vals)
    samples = [sum(vals[rng.randrange(n)] for _ in range(n)) / n
               for _ in range(n_resamples)]
    alpha = (1.0 - confidence) / 2.0
    return {"estimate": _mean(vals), "lower": _percentile(samples, alpha),
            "upper": _percentile(samples, 1.0 - alpha), "n_cases": n,
            "n_resamples": n_resamples, "seed": seed}


def _wilcoxon(values: Sequence[float]) -> float | None:
    """返回双侧 paired Wilcoxon p 值；没有 scipy 时明确返回 None。"""
    vals = [v for v in values if v != 0]
    if not vals:
        return 1.0 if values else None
    try:
        from scipy.stats import wilcoxon
        return float(wilcoxon(vals, alternative="two-sided", method="auto").pvalue)
    except (ImportError, ValueError):
        return None


def holm_adjust(pvalues: Sequence[float | None]) -> list[float | None]:
    """Holm-Bonferroni 校正，保留输入顺序和缺失值。"""
    valid = sorted(((float(p), i) for i, p in enumerate(pvalues) if p is not None),
                   key=lambda x: x[0])
    out: list[float | None] = [None] * len(pvalues)
    running = 0.0
    m = len(valid)
    for rank, (p, original) in enumerate(valid):
        adjusted = min(1.0, (m - rank) * p)
        running = max(running, adjusted)
        out[original] = running
    return out


def _replicate_means(rows: Iterable[dict]) -> dict[tuple, float]:
    groups: dict[tuple, list[float]] = defaultdict(list)
    for row in _usable(rows):
        groups[(_case_key(row), row.get("condition"))].append(float(row["score"]))
    return {key: statistics.fmean(vals) for key, vals in groups.items()}


def _case_key(row: dict) -> tuple:
    return (*_system_key(row), row.get("case"))


def _pair_key(row: dict) -> tuple | None:
    """唯一的 Full 参考配对键。

    ``run_index`` 是重复实验的配对标识。缺失时宁可返回 ``None`` 让 SAL /
    LossShare 记为未配对，也不把不同重复运行的分数错误混合。
    """
    run_index = row.get("run_index")
    if run_index is None:
        return None
    try:
        run_index = int(run_index)
    except (TypeError, ValueError):
        return None
    return (*_case_key(row), run_index)


def _effect(rows: Iterable[dict], condition_a: str, condition_b: str,
            *, seed: int, n_resamples: int) -> list[dict]:
    means = _replicate_means(rows)
    systems = sorted({_system_key(r) for r in rows})
    effects: list[dict] = []
    for scaffold, model in systems:
        cases = sorted({key[0][2] for key in means if key[0][:2] == (scaffold, model)})
        pairs = []
        for case in cases:
            a = means.get(((scaffold, model, case), condition_a))
            b = means.get(((scaffold, model, case), condition_b))
            if a is not None and b is not None:
                pairs.append({"case": case, "a": a, "b": b, "difference": a - b})
        diffs = [p["difference"] for p in pairs]
        ci = bootstrap_mean(diffs, seed=seed, n_resamples=n_resamples)
        wins = sum(d > 0 for d in diffs)
        losses = sum(d < 0 for d in diffs)
        ties = len(diffs) - wins - losses
        effects.append({
            "scaffold": scaffold, "model": model,
            "condition_a": condition_a, "condition_b": condition_b,
            "n_pairs": len(pairs), "mean_difference": _mean(diffs),
            "median_difference": _median(diffs), "bootstrap_ci": ci,
            "wilcoxon_p": _wilcoxon(diffs),
            "win_rate": ((wins + 0.5 * ties) / len(diffs)) if diffs else None,
            "wins": wins, "losses": losses, "ties": ties, "pairs": pairs,
        })
    return effects


def _three_condition_effect(rows: Iterable[dict], *, seed: int,
                            n_resamples: int) -> list[dict]:
    """计算 Full−Interact、Interact−Hidden 和逐 case rho_info。"""
    means = _replicate_means(rows)
    systems = sorted({_system_key(r) for r in rows})
    out = []
    for scaffold, model in systems:
        cases = sorted({key[0][2] for key in means if key[0][:2] == (scaffold, model)})
        records = []
        for case in cases:
            h = means.get(((scaffold, model, case), "Hidden"))
            i = means.get(((scaffold, model, case), "Interact"))
            f = means.get(((scaffold, model, case), "Full"))
            if h is None or i is None or f is None:
                continue
            denom = f - h
            records.append({"case": case, "input_gap": f - i,
                            "g_info": i - h,
                            "rho_info": (i - h) / denom if denom != 0 else None})
        def stat(field: str):
            vals = [r[field] for r in records if r[field] is not None]
            return {"mean": _mean(vals), "median": _median(vals),
                    "bootstrap_ci": bootstrap_mean(vals, seed=seed,
                                                    n_resamples=n_resamples),
                    "n_cases": len(vals)}
        out.append({"scaffold": scaffold, "model": model,
                    "input_gap": stat("input_gap"),
                    "g_info": stat("g_info"), "rho_info": stat("rho_info"),
                    "n_complete_cases": len(records), "cases": records})
    return out


def _load_thresholds(path: str | Path | None) -> dict[str, float]:
    if path is None:
        return {}
    data = _read(Path(path))
    # 允许直接 {case: tau}，也允许 {"thresholds": {case: tau}}。
    data = data.get("thresholds", data)
    return {str(k): float(v) for k, v in data.items()
            if isinstance(v, (int, float)) and math.isfinite(v)}


def _pass(row: dict, thresholds: dict[str, float] | None = None) -> bool | None:
    if row.get("validity") is None:
        return None
    feasible = float(row["validity"]) > 0
    if not thresholds:
        return feasible
    tau = thresholds.get(str(row.get("case")))
    if tau is None or row.get("quality") is None:
        return None
    return feasible and float(row["quality"]) >= tau


def _comb(n: int, k: int) -> int:
    return math.comb(n, k) if 0 <= k <= n else 0


def pass_metrics(rows: Iterable[dict], thresholds: dict[str, float] | None, k: int) -> list[dict]:
    """按 system/condition 宏平均可行率、pass@k 与 pass^k。

    不传 thresholds 时，Pass 就是当前 evaluator 的 0/1 可行性
    ``validity_score > 0``；传入 thresholds 才启用额外的质量门槛。
    """
    if k < 1:
        raise ValueError("k 必须为正数")
    groups: dict[tuple, list[bool]] = defaultdict(list)
    for row in rows:
        if row.get("status") in EXCLUDED_STATUS:
            continue
        passed = _pass(row, thresholds)
        if passed is not None:
            groups[(*_system_key(row), row.get("condition"), row.get("case"))].append(passed)
    by_group: dict[tuple, list[dict]] = defaultdict(list)
    for (scaffold, model, condition, case), vals in groups.items():
        n, c = len(vals), sum(vals)
        if n < k:
            at_k = None
            all_k = None
        else:
            at_k = 1.0 - (_comb(n - c, k) / _comb(n, k))
            all_k = _comb(c, k) / _comb(n, k)
        by_group[(scaffold, model, condition)].append(
            {"case": case, "n": n, "c": c, "pass_at_k": at_k, "pass_power_k": all_k})
    out = []
    for (scaffold, model, condition), cases in sorted(by_group.items()):
        at = [x["pass_at_k"] for x in cases if x["pass_at_k"] is not None]
        pw = [x["pass_power_k"] for x in cases if x["pass_power_k"] is not None]
        success = [x["c"] / x["n"] for x in cases if x["n"]]
        out.append({"scaffold": scaffold, "model": model, "condition": condition,
                    "k": k, "n_cases": len(cases), "n_cases_with_k": len(at),
                    "success_rate": _mean(success),
                    "pass_at_k": _mean(at), "pass_power_k": _mean(pw), "cases": cases})
    return out


def _failed_row(row: dict) -> bool:
    """可归因的失败：有 evaluator 结果且 validity 明确为 0。"""
    return (row.get("status") not in EXCLUDED_STATUS
            and _number(row.get("validity")) is not None
            and float(row["validity"]) <= 0)


def _valid_stage(stage) -> str:
    stage = str(stage or "").strip()
    return stage if stage in FAILURE_STAGES else "unknown"


def _valid_scope(scope, *, default: str = "agent") -> str:
    """规范化归因范围。

    历史 ``failure_stage.json`` 没有 ``failure_scope``。为保持既有 FFR/SAL
    结果可复现，这类已评分的可行性失败暂按 ``agent`` 处理；新实验应显式写入
    scope，尤其是判定为 infrastructure / interaction / benchmark 的样本。
    """
    scope = str(scope or "").strip()
    if not scope:
        return default
    return scope if scope in FAILURE_SCOPES else "unknown"


def _reference_means(rows: Iterable[dict], condition: str = "Full") -> dict[tuple, float]:
    """同一 case/system/run_index 的参考条件分数。

    Full 与失败运行必须按 ``run_index`` 一一配对。相同键出现重试副本时才
    在该键内取均值；不会把 run1/run2 的 Full 分数混成一个参考值。
    """
    groups: dict[tuple, list[float]] = defaultdict(list)
    for row in _usable(rows):
        key = _pair_key(row)
        if row.get("condition") == condition and key is not None:
            groups[key].append(float(row["score"]))
    return {key: statistics.fmean(values) for key, values in groups.items()}


def failure_attribution(rows: Iterable[dict], *, reference_condition: str = "Full",
                        target_condition: str | None = None) -> dict:
    """计算 FFR、SAL、LossShare。

    候选失败为非 ``error/degraded`` 且 evaluator 明确给出 ``validity <= 0``
    的 run。正式的 Agent FFR/SAL/LossShare 仅以 ``failure_scope=agent`` 的
    候选失败为分母；``interaction``、``infrastructure``、``benchmark`` 会
    单列报告，绝不混入 Agent 能力归因。历史上没有 scope 的结果兼容为 agent。

    没有 ``failure_stage`` 的 Agent 失败进入 ``unknown``，仍计入 FFR 分母；
    SAL/LossShare 只有存在参考条件配对分数的样本才参与，并报告未配对数量。

    ``loss`` 使用有符号的 ``S_reference - S_failed``，不偷偷裁剪负值。
    """
    rows = list(rows)
    candidates = [r for r in rows if _failed_row(r)
                  and (target_condition is None
                       or r.get("condition") == target_condition)]
    failed = [r for r in candidates
              if _valid_scope(r.get("failure_scope")) == "agent"]
    ref = _reference_means(rows, reference_condition)
    ffr_counts = {stage: 0 for stage in (*FAILURE_STAGES, "unknown")}
    stage_rows: dict[str, list[dict]] = defaultdict(list)
    unpaired = 0
    for row in failed:
        stage = _valid_stage(row.get("failure_stage"))
        ffr_counts[stage] += 1
        reference = ref.get(_pair_key(row))
        failed_score = _number(row.get("score"))
        if reference is None or failed_score is None:
            unpaired += 1
            continue
        stage_rows[stage].append({
            "run_id": row.get("run_id"), "case": row.get("case"),
            "run_index": row.get("run_index"),
            "loss": reference - float(failed_score),
            "reference_score": reference, "failed_score": float(failed_score),
        })

    total_failed = len(failed)
    total_paired = sum(len(v) for v in stage_rows.values())
    total_loss = sum(x["loss"] for values in stage_rows.values() for x in values)
    stages = (*FAILURE_STAGES, "unknown")
    ffr = [{"stage": stage, "n_failures": ffr_counts[stage],
            "rate": (ffr_counts[stage] / total_failed if total_failed else None)}
           for stage in stages]
    sal, loss_share = [], []
    for stage in stages:
        values = stage_rows.get(stage, [])
        losses = [x["loss"] for x in values]
        sal.append({"stage": stage, "n_failures": len(values),
                    "mean_loss": _mean(losses), "median_loss": _median(losses),
                    "total_loss": sum(losses) if losses else None,
                    "n_paired": len(values)})
        loss_share.append({"stage": stage, "n_paired": len(values),
                           "loss_share": (sum(losses) / total_loss
                                          if values and total_loss != 0 else None)})
    scoped = {scope: 0 for scope in (*FAILURE_SCOPES, "unknown")}
    for row in candidates:
        scoped[_valid_scope(row.get("failure_scope"))] += 1

    return {
        "candidate_failure_definition": "status not in {error,degraded} and validity <= 0",
        "failure_definition": (
            "candidate failure with failure_scope=agent "
            "(missing scope defaults to agent for backward compatibility)"
        ),
        "reference_condition": reference_condition,
        "target_condition": target_condition,
        "n_failed": total_failed,
        "n_candidate_failures": len(candidates),
        "n_non_agent_failures": len(candidates) - total_failed,
        "candidate_failures_by_scope": scoped,
        "n_unlabelled": ffr_counts["unknown"],
        "n_paired_for_loss": total_paired,
        "n_unpaired_for_loss": unpaired,
        "total_paired_loss": total_loss if total_paired else None,
        "ffr": ffr, "sal": sal, "loss_share": loss_share,
        "paired_rows": [x for values in stage_rows.values() for x in values],
    }


def grouped_failure_attribution(rows: Iterable[dict], *,
                                reference_condition: str = "Full") -> list[dict]:
    """按 ``Agent system × condition`` 输出 FFR/SAL/LossShare。

    目标条件决定哪些失败进入分子；Full 参考分仍从同一 system 的所有行中获取，
    避免先筛选 Interact 后把配对参考一起筛掉。
    """
    all_rows = list(rows)
    groups = sorted({(*_system_key(r), r.get("condition")) for r in all_rows},
                    key=lambda key: tuple(str(x) for x in key))
    out = []
    for scaffold, model, condition in groups:
        system_rows = [r for r in all_rows if _system_key(r) == (scaffold, model)]
        target_rows = [r for r in system_rows if r.get("condition") == condition]
        analysis_rows = target_rows + [
            r for r in system_rows
            if r.get("condition") == reference_condition and r not in target_rows
        ]
        result = failure_attribution(analysis_rows,
                                     reference_condition=reference_condition,
                                     target_condition=condition)
        out.append({"scaffold": scaffold, "model": model,
                    "condition": condition, **result})
    return out


def operational_attribution(rows: Iterable[dict]) -> dict:
    """单列不能作为 Agent 可行性 0 分归因的运行故障。

    ``error`` / ``degraded``、无评分 timeout、未完成及其他无评分运行没有可靠的
    evaluator 可行性结果，因此不进入 FFR。这里按状态和已标注的 failure_scope
    计数；没有 sidecar 的运行保留为 ``unknown``，不擅自判为 Agent 失败。
    """
    rows = list(rows)

    def category(row: dict) -> str | None:
        status = row.get("status")
        if status in EXCLUDED_STATUS:
            return str(status)
        scored = (_number(row.get("score")) is not None
                  and _number(row.get("validity")) is not None)
        if scored:
            return None
        if status == "timeout":
            return "timeout_unscored"
        if status is None:
            return "incomplete"
        return "unscored"

    failed = [(row, category(row)) for row in rows if category(row) is not None]
    statuses = {status: 0 for status in
                ("degraded", "error", "timeout_unscored", "incomplete", "unscored")}
    scopes = {scope: 0 for scope in (*FAILURE_SCOPES, "unknown")}
    reasons: dict[str, int] = defaultdict(int)
    for row, status in failed:
        statuses[status] += 1
        scopes[_valid_scope(row.get("failure_scope"), default="unknown")] += 1
        if row.get("killed_reason"):
            reasons[str(row["killed_reason"])] += 1
    return {
        "n_operational_failures": len(failed),
        "operational_failure_rate": len(failed) / len(rows) if rows else None,
        "by_status": statuses,
        "by_scope": scopes,
        "by_killed_reason": dict(sorted(reasons.items())),
    }


def analyze(rows: Iterable[dict], *, pairs: Sequence[tuple[str, str]] = (),
            thresholds: dict[str, float] | None = None, ks: Sequence[int] = (1, 3),
            seed: int = 0, n_resamples: int = 10_000) -> dict:
    """计算所有可由现有 run 结果得到的离线指标。"""
    rows = list(rows)
    effects = []
    for a, b in pairs:
        effects.extend(_effect(rows, a, b, seed=seed, n_resamples=n_resamples))
    pvals = holm_adjust([e["wilcoxon_p"] for e in effects])
    for effect, adjusted in zip(effects, pvals):
        effect["wilcoxon_p_holm"] = adjusted
    operational = operational_attribution(rows)
    result = {
        "n_runs": len(rows),
        "n_usable_score_runs": len(_usable(rows)),
        "n_excluded": operational["n_operational_failures"],
        "effects": effects,
        "three_condition_effects": _three_condition_effect(rows, seed=seed,
                                                              n_resamples=n_resamples),
        "failure_attribution": failure_attribution(rows),
        "failure_attribution_by_group": grouped_failure_attribution(rows),
        "operational_attribution": operational,
    }
    thresholds = thresholds or {}
    if thresholds:
        result["thresholds_n"] = len(thresholds)
        result["pass_predicate"] = "validity_score > 0 and quality_score >= case threshold"
    else:
        result["pass_predicate"] = "validity_score > 0"
    result["pass_metrics"] = [pass_metrics(rows, thresholds, k) for k in ks]
    # SuccessRate 不依赖 k，单独提供去重后的易读视图；每个 k 的明细中也
    # 保留该字段，便于只消费某一组 pass 指标的调用方使用。
    first = result["pass_metrics"][0] if result["pass_metrics"] else []
    result["success_rate"] = [
        {key: row[key] for key in ("scaffold", "model", "condition", "n_cases", "success_rate")}
        for row in first
    ]
    return result


def _json_default(obj):
    if isinstance(obj, Path):
        return str(obj)
    raise TypeError(type(obj).__name__)


def main() -> None:
    parser = argparse.ArgumentParser(description="计算 FDE-Bench case-level 离线指标")
    parser.add_argument("results", nargs="+", type=Path)
    parser.add_argument("--pair", nargs=2, action="append", metavar=("A", "B"),
                        help="计算 A−B 的配对效应，可重复传入")
    parser.add_argument("--thresholds", type=Path,
                        help="JSON: {case: tau}，用于 SuccessRate/pass@k")
    parser.add_argument("--k", type=int, action="append", dest="ks",
                        help="pass@k/pass^k 的 k，可重复传入（默认 1、3）")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--bootstrap", type=int, default=10_000,
                        help="bootstrap 重采样次数（默认 10000）")
    parser.add_argument("--json", type=Path, help="输出 JSON 文件")
    args = parser.parse_args()
    rows = load_runs(args.results)
    if not rows:
        raise SystemExit("未找到包含 manifest.json 的 run")
    result = analyze(rows, pairs=args.pair or (),
                     thresholds=_load_thresholds(args.thresholds),
                     ks=args.ks or (1, 3), seed=args.seed,
                     n_resamples=args.bootstrap)
    payload = json.dumps(result, ensure_ascii=False, indent=2, default=_json_default)
    if args.json:
        args.json.parent.mkdir(parents=True, exist_ok=True)
        args.json.write_text(payload + "\n", encoding="utf-8")
        print(f"已写出 {args.json}")
    else:
        print(payload)


if __name__ == "__main__":
    main()
