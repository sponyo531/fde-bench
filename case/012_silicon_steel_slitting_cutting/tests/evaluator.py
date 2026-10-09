"""
EXTRACTOR_SPEC:
  plan_file: solution.csv
  required_columns: [mother_id, mother_width, mother_cut_weight, sub1_width, sub1_long]
  optional_columns: [sub2_width, sub3_width, sub4_width, sub5_width,
                     sub2_long, sub3_long, sub4_long, sub5_long]
  # 评估器只消费上面这些列；agent 顺手写的其它中间量(各类 weight/thin/long 汇总列)一律忽略,
  # 不必补齐、也不影响判分。
  column_formats:
    mother_id: string, mother coil identifier. Must reference a coil in data/mother_coil.csv
               (the evaluator assigns IDs M_0..M_11 by row order to mother_coil.csv when
               the raw file has no ID column). Same mother reused across multiple cuts must
               use the same mother_id.
    mother_width: number (mm), width of the mother coil used in this cut. Must equal the
                  width of the referenced mother_id in mother_coil.csv.
    mother_cut_weight: number (kg), the section weight cut off from the mother coil in
                       THIS cut (the "length-wise slab" weight). Must be >= 100 kg.
                       Cumulative sum across cuts of the same mother_id must not exceed
                       that mother's total 重量 in mother_coil.csv.
    sub1_width ... sub5_width: number (mm), width of each sub-strip produced in this cut.
                       Between 1 and 5 slots must be filled. Each width MUST appear in
                       requirements.csv's 宽度 column (exact match).
    sub1_long ... sub5_long: number (mm), length of each sub-strip produced in this cut.
                       Informational only — **not scored**. By definition it equals the
                       cut's length (= mother_cut_weight / (mother_width × 7.65) × 10000),
                       so it carries no information the evaluator does not already have;
                       output weight is derived from mother_cut_weight and the widths, not
                       from this field. Report what your plan actually computed; a mismatch
                       is surfaced as a warning and costs nothing. Unused slots leave both
                       sub_i_width and sub_i_long blank (that pairing is how the evaluator
                       detects which slots are filled).
  notes: >
    Each row = one 断刀 (cutting operation). Total number of rows = cut_times.
    A cut position produces exactly one sub-strip of one width; if the same width appears
    twice in one cut, use TWO different sub_i_width slots (e.g. sub1_width=555 and
    sub2_width=555). Do NOT use a `count` field. The evaluator counts distinct filled
    slots per row (max 5).

    Column name fallbacks the extractor should map to canonical English names:
      mother_id           → 包号, coil_id, mother_coil_id, mid
      mother_width        → 母板宽度, 母卷宽度, coil_width
      mother_cut_weight   → 断刀重量, 切下重量, cut_weight, section_weight
      sub{i}_width        → 子板{i}宽度, sub_width_{i}, strip_{i}_width, w{i}
      sub{i}_long         → 子板{i}长度, sub_len_{i}, strip_{i}_length, l{i}

    Constraints the evaluator will enforce (do not need to be reported in the CSV; just
    make sure each row obeys them):
      - Σ sub_i_width ≤ mother_width (no edge-waste threshold, natural cut width sum)
      - sub_i_width ∈ requirements.csv widths (exact match)
      - cumulative mother_cut_weight per mother_id ≤ mother.重量 (10 kg tolerance)
      - per width, Σ produced sub weight ≈ demand (±10 kg), where each sub-strip's weight
        is prorated from the cut: cut_w × sub_i_width / mother_width. Density and
        sub_i_long do not enter this calculation.
"""

from __future__ import annotations

import argparse
import json
from collections import defaultdict
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import pandas as pd

# ── 常量 ──────────────────────────────────────────────
EVAL_MIN_CUT_WEIGHT = 100.0      # kg - 最小断刀重量
DENSITY = 7.65                   # g/cm^3 - 硅钢密度
EVAL_LENGTH_TOL = 10.0           # mm - 子板长度与母卷本刀长度一致性容差
EVAL_WIDTH_TOL = 10.0            # mm - 母卷宽度一致性容差
EVAL_WEIGHT_TOL = 10.0           # kg - 需求重量/库存重量的判断容差
EVAL_CUT_WEIGHT_TOL = 10.0       # kg - 断刀最小重量的浮点毛刺容差

# 容差比较的浮点余量。容差是**闭区间**：剩余恰好 10.00kg 必须算达标。
# 不加这一项时，「恰好卡在边界」的解会因累加噪声随机翻车——co5/try_3 排产时
# 按容差精确留量，35 个宽度里 31 个剩余是 ±10.00000000000000（通过），
# 另外 4 个是 10.00000000000091 / 10.00000000000182（判违规）。同一份计划里
# 哪几个中招完全取决于浮点累加顺序，与方案质量无关。
TOL_EPS = 1e-6

TARGET_CUT_TIMES = 46            # 参考解的断刀次数（目标断刀）
CUT_TIMES_PENALTY_RATE = 0.20    # 每超出目标 1 次，cut_score 损失 20%

PLAN_FILE = "solution.csv"


# ── 工具 ──────────────────────────────────────────────

def _safe_float(val) -> Optional[float]:
    try:
        if val is None or pd.isna(val):
            return None
        return float(val)
    except (TypeError, ValueError):
        return None


# ── 数据加载 ──────────────────────────────────────────

def load_data(data_dir: Path) -> Tuple[pd.DataFrame, pd.DataFrame]:
    mother_df = pd.read_csv(data_dir / "mother_coil.csv")
    req_df = pd.read_csv(data_dir / "requirements.csv")

    mother_df.columns = [c.strip() for c in mother_df.columns]
    req_df.columns = [c.strip() for c in req_df.columns]

    mother_df = mother_df.rename(columns={
        "材质": "mother_mat",
        "宽度": "mother_width",
        "重量": "mother_weight",
        "包号": "mother_id",
        "厚度": "mother_thin",
        "厚度mm": "mother_thin",
    })
    req_df = req_df.rename(columns={
        "宽度": "sub_width",
        "重量": "sub_weight",
        "厚度": "sub_thin",
    })

    if "mother_id" not in mother_df.columns:
        mother_df["mother_id"] = ["M_" + str(i) for i in range(len(mother_df))]
    mother_df["mother_id"] = mother_df["mother_id"].astype(str).str.strip()

    mother_df["mother_width"] = pd.to_numeric(mother_df["mother_width"], errors="coerce")
    mother_df["mother_weight"] = pd.to_numeric(mother_df["mother_weight"], errors="coerce")
    if "mother_thin" not in mother_df.columns:
        mother_df["mother_thin"] = pd.NA
    mother_df["mother_thin"] = pd.to_numeric(mother_df["mother_thin"], errors="coerce").fillna(1.0)

    req_df["sub_width"] = pd.to_numeric(req_df["sub_width"], errors="coerce")
    req_df["sub_weight"] = pd.to_numeric(req_df["sub_weight"], errors="coerce")
    if "sub_thin" not in req_df.columns:
        req_df["sub_thin"] = pd.NA
    req_df["sub_thin"] = pd.to_numeric(req_df["sub_thin"], errors="coerce").fillna(1.0)

    return mother_df, req_df


# ── 硬约束 & 目标独立重算 ───────────────────────────

def check_and_score(
    solution: pd.DataFrame,
    mother_df: pd.DataFrame,
    req_df: pd.DataFrame,
) -> Dict[str, Any]:
    """逐行仿真：硬约束校验 + 按宽度占比分摊断刀重量得出产出。

    返回一个 dict：
      is_feasible: bool
      violations: List[str]
      warnings: List[str]      不计违规的提示（子板长度自洽性）
      cut_times: int
      requirement_ratio: float   (余料率 = (Σcut_w - Σsub_w_recalc) / Σcut_w)
      spec_stats: {satisfied_count, unsatisfied_count, total_specs, total_shortage_kg}
    """
    violations: List[str] = []
    # 不计违规的提示（当前只有子板长度自洽性——它不参与判分，见下方说明）
    warnings: List[str] = []

    # 母卷剩余重量、母卷宽度映射（按 mother_id）
    mother_remaining: Dict[str, float] = {}
    mother_width_by_id: Dict[str, float] = {}
    for _, row in mother_df.iterrows():
        mid = str(row["mother_id"])
        mw = _safe_float(row["mother_weight"])
        mwidth = _safe_float(row["mother_width"])
        if mid and mw is not None and mw > 0:
            mother_remaining[mid] = mother_remaining.get(mid, 0.0) + mw
            mother_width_by_id.setdefault(mid, mwidth)

    # 需求：按宽度汇总
    required_by_width: Dict[float, float] = defaultdict(float)
    for _, row in req_df.iterrows():
        w = _safe_float(row.get("sub_width"))
        wt = _safe_float(row.get("sub_weight"))
        if w is not None and wt is not None and w > 0:
            required_by_width[w] += wt
    demand_remaining: Dict[float, float] = dict(required_by_width)
    valid_sub_widths = set(required_by_width.keys())

    produced_by_width: Dict[float, float] = defaultdict(float)
    total_cut_weight = 0.0
    total_sub_weight = 0.0

    # 逐行仿真
    for row_idx, row in solution.iterrows():
        mid_raw = row.get("mother_id")
        mid = str(mid_raw).strip() if pd.notna(mid_raw) else ""
        if not mid:
            violations.append(f"第{row_idx}行: mother_id 缺失")
            continue
        if mid not in mother_remaining:
            violations.append(
                f"第{row_idx}行: mother_id={mid} 不在库存中"
            )
            continue

        actual_width = _safe_float(row.get("mother_width"))
        expected_width = mother_width_by_id.get(mid)
        if actual_width is not None and expected_width is not None:
            if abs(actual_width - expected_width) > EVAL_WIDTH_TOL + TOL_EPS:
                violations.append(
                    f"第{row_idx}行: mother_width={actual_width} 与原始 {expected_width} 不一致"
                )
                actual_width = expected_width  # 用原始宽度继续算
        # 修复：申报宽度只用于一致性校验，后续 Σsub_width 上限、长度换算、
        # 子板分摊分母一律用库存真实宽度。否则容差内的谎报（+10mm 多塞宽度、
        # −10mm 凭空放大产出）会直接改分。
        if expected_width is not None:
            actual_width = expected_width
        if actual_width is None:
            actual_width = expected_width

        cut_w = _safe_float(row.get("mother_cut_weight"))
        if cut_w is None or cut_w <= 0:
            violations.append(f"第{row_idx}行: mother_cut_weight 缺失或非正 ({cut_w})")
            continue
        if cut_w + EVAL_CUT_WEIGHT_TOL + TOL_EPS < EVAL_MIN_CUT_WEIGHT:
            violations.append(
                f"第{row_idx}行: mother_cut_weight={cut_w:.2f} < {EVAL_MIN_CUT_WEIGHT}"
            )

        # 提取本刀有效子板
        valid_subs: List[Dict[str, float]] = []
        for i in range(1, 6):
            sw = _safe_float(row.get(f"sub{i}_width"))
            sl = _safe_float(row.get(f"sub{i}_long"))
            if sw is not None and sw > 0 and sl is not None and sl > 0:
                valid_subs.append({"width": sw, "long": sl, "idx": i})

        if not (1 <= len(valid_subs) <= 5):
            violations.append(
                f"第{row_idx}行: 本刀有效子板数={len(valid_subs)}，应在 1~5"
            )
            continue

        # 每个子板宽度必须在需求规格中
        for sub in valid_subs:
            if sub["width"] not in valid_sub_widths:
                violations.append(
                    f"第{row_idx}行: sub{sub['idx']}_width={sub['width']} 不在需求宽度集中"
                )

        # Σ 子板宽度 ≤ 母卷宽度
        sum_widths = sum(s["width"] for s in valid_subs)
        if actual_width is not None and sum_widths > actual_width + 1e-6:
            violations.append(
                f"第{row_idx}行: Σsub_width={sum_widths} > mother_width={actual_width}"
            )

        # 重算本刀母卷长度（按厚度 1 折算，本数据集所有厚度都为 1）
        if actual_width is not None and actual_width > 0:
            mother_cut_long = cut_w / (actual_width * DENSITY) * 10000.0
        else:
            mother_cut_long = None

        # 子板产出重量：按宽度占比分摊断刀重量，**不用 agent 自报的长度**。
        #
        #     weight_calc = mother_cut_long × sub_width × ρ / 10000
        #     mother_cut_long = cut_w / (mother_width × ρ) × 10000
        #     ── 代入后 ρ 与长度双双约掉 ──────────────────────
        #     weight_calc = cut_w × sub_width / mother_width
        #
        # 为什么改：sub_i_long 的信息量是零——它按定义就该等于 mother_cut_long，
        # 而后者完全由 cut_w 和母卷宽度决定。真正的决策只有三件：用哪个母卷、
        # 这一刀切多少重量、怎么分宽度。拿一个可反算的字段去参与判分有两个后果：
        #   ① 长度报大 → 产出跟着变大。原先只靠 10mm 容差拦着，容差就成了搜索空间。
        #   ② 抽取器可以用判分公式本身反算长度，让校验按构造必然通过（自证）。
        #      dsv4f/try_1 即此例：agent 用了 7.85 密度，两个抽取器按题面 7.65
        #      把长度整列重算，一致性校验全过，0 分变 0.871——它们的辩解
        #      「长度是派生量，我没改任何决策」在事实层面是对的，错的是评估器
        #      把派生量放进了判分路径。
        # 改成按宽度占比分摊后，Σ产出 = cut_w × Σsub_width/mother_width ≤ cut_w
        # （上面已校验 Σsub_width ≤ mother_width），质量守恒是结构性的，
        # 且密度取值完全不进判分——7.65/7.85 之争在这里没有着力点。
        for sub in valid_subs:
            sub["weight_calc"] = cut_w * sub["width"] / actual_width

        # 长度只做自洽性提示，不计违规：它不携带独立信息，判它等于判 agent 的
        # 算术，而这条恰恰是可以被反算绕过的。真错了（比如密度用成碳钢的 7.85）
        # 仍然留痕，供人工排查。
        if mother_cut_long is not None:
            for sub in valid_subs:
                if abs(sub["long"] - mother_cut_long) > EVAL_LENGTH_TOL:
                    warnings.append(
                        f"第{row_idx}行: sub{sub['idx']}_long={sub['long']} != "
                        f"mother_cut_long={mother_cut_long:.2f} 超出 {EVAL_LENGTH_TOL}mm"
                        f"（仅提示，不计违规；长度不参与判分）"
                    )

        # 状态扣减
        mother_remaining[mid] = mother_remaining.get(mid, 0.0) - cut_w
        total_cut_weight += cut_w
        for sub in valid_subs:
            sw = sub["width"]
            swc = sub["weight_calc"]
            demand_remaining[sw] = demand_remaining.get(sw, 0.0) - swc
            produced_by_width[sw] += swc
            total_sub_weight += swc

        # 库存超支
        if mother_remaining[mid] < -(EVAL_WEIGHT_TOL + TOL_EPS):
            violations.append(
                f"第{row_idx}行: 母板{mid} 累计消耗超出库存 (剩余 {mother_remaining[mid]:.2f}kg < -{EVAL_WEIGHT_TOL})"
            )

    # 收尾：各规格产出 vs 需求
    satisfied = 0
    unsatisfied = 0
    total_shortage = 0.0
    for width_key in sorted(valid_sub_widths):
        remaining = demand_remaining.get(width_key, 0.0)
        if abs(remaining) > EVAL_WEIGHT_TOL + TOL_EPS:
            if remaining > EVAL_WEIGHT_TOL + TOL_EPS:
                violations.append(
                    f"宽度{width_key}mm 欠产: 剩余 {remaining:.2f}kg > {EVAL_WEIGHT_TOL}"
                )
                unsatisfied += 1
                total_shortage += remaining
            else:
                violations.append(
                    f"宽度{width_key}mm 超产: 剩余 {remaining:.2f}kg < -{EVAL_WEIGHT_TOL}"
                )
                unsatisfied += 1
        else:
            satisfied += 1

    is_feasible = len(violations) == 0

    if total_cut_weight > 0:
        requirement_ratio = float(max((total_cut_weight - total_sub_weight) / total_cut_weight, 0.0))
    else:
        requirement_ratio = 0.0

    return {
        "is_feasible": is_feasible,
        "violations": violations,
        "warnings": warnings,
        "cut_times": int(len(solution)),
        "requirement_ratio": requirement_ratio,
        "spec_stats": {
            "satisfied_count": satisfied,
            "unsatisfied_count": unsatisfied,
            "total_specs": len(required_by_width),
            "total_shortage_kg": round(total_shortage, 2),
        },
        "total_cut_weight": total_cut_weight,
        "total_sub_weight": total_sub_weight,
    }


# ── 主评估入口 ────────────────────────────────────────

def resolve_plan_path(submission_dir: Path) -> Tuple[Optional[Path], List[str]]:
    preferred = submission_dir / PLAN_FILE
    if preferred.is_file():
        return preferred, []
    matches = [p for p in submission_dir.glob(f"**/{PLAN_FILE}") if p.is_file()]
    if not matches:
        return None, [f"{PLAN_FILE} not found under {submission_dir}"]
    return matches[0], []


def evaluate(data_dir: Path, baseline_dir: Path, submission_dir: Path) -> Dict[str, Any]:
    # 防御: 若调用方不认三角参、按 (submission_dir, data_dir) 传, 探测并修正顺序。
    if not (Path(submission_dir) / PLAN_FILE).is_file() and (Path(data_dir) / PLAN_FILE).is_file():
        data_dir, submission_dir = submission_dir, data_dir
    baseline = json.loads((baseline_dir / "reference_metrics.json").read_text(encoding="utf-8"))
    ref_value = float(baseline["reference_value"])

    mother_df, req_df = load_data(data_dir)

    plan_path, discovery_errors = resolve_plan_path(submission_dir)
    if discovery_errors:
        return {
            "validity_score": 0.0,
            "quality_score": 0.0,
            "overall_score": 0.0,
            "errors": discovery_errors,
            "reference_value": ref_value,
        }

    try:
        solution = pd.read_csv(plan_path)
        solution.columns = [c.strip() for c in solution.columns]
    except Exception as exc:
        return {
            "validity_score": 0.0,
            "quality_score": 0.0,
            "overall_score": 0.0,
            "errors": [f"Failed to read {PLAN_FILE}: {exc}"],
            "resolved_artifact": str(plan_path),
            "reference_value": ref_value,
        }

    # 清理空行/无 mother_cut_weight 的说明行
    solution = solution.dropna(how="all")
    if "mother_cut_weight" in solution.columns:
        solution = solution[
            solution["mother_cut_weight"].notna()
            & (pd.to_numeric(solution["mother_cut_weight"], errors="coerce") > 0)
        ].reset_index(drop=True)

    if solution.empty:
        return {
            "validity_score": 0.0,
            "quality_score": 0.0,
            "overall_score": 0.0,
            "errors": [f"{PLAN_FILE} 无有效数据行"],
            "resolved_artifact": str(plan_path),
            "reference_value": ref_value,
        }

    result = check_and_score(solution, mother_df, req_df)
    cut_times = result["cut_times"]

    if not result["is_feasible"]:
        return {
            "validity_score": 0.0,
            "quality_score": 0.0,
            "overall_score": 0.0,
            "errors": result["violations"][:8],
            "warnings": result["warnings"][:8],
            "resolved_artifact": str(plan_path),
            "submission_metrics": {
                "cut_times": cut_times,
                "spec_satisfied": f"{result['spec_stats']['satisfied_count']}/{result['spec_stats']['total_specs']}",
                "spec_total_shortage_kg": result["spec_stats"]["total_shortage_kg"],
            },
            "reference_value": ref_value,
        }

    # 合法方案：计算得分
    requirement_ratio = result["requirement_ratio"]
    utilization_score = max(0.0, 1.0 - requirement_ratio)
    excess_cuts = max(0, cut_times - TARGET_CUT_TIMES)
    if cut_times <= TARGET_CUT_TIMES:
        cut_score = 1.0 + 0.05 * (TARGET_CUT_TIMES - cut_times)
    else:
        cut_score = max(0.0, 1.0 - CUT_TIMES_PENALTY_RATE * excess_cuts)

    combined_score = 0.05 * utilization_score + 0.95 * cut_score
    quality_score = combined_score / (ref_value + 1e-6)

    return {
        "validity_score": 1.0,
        "quality_score": round(quality_score, 6),
        "overall_score": round(quality_score, 6),
        "errors": [],
        "warnings": result["warnings"][:8],
        "resolved_artifact": str(plan_path),
        "submission_metrics": {
            "cut_times": cut_times,
            "target_cut_times": TARGET_CUT_TIMES,
            "excess_cuts": excess_cuts,
            "requirement_ratio": round(requirement_ratio, 4),
            "utilization_score": round(utilization_score, 4),
            "cut_score": round(cut_score, 4),
            "combined_score": round(combined_score, 4),
            "spec_satisfied": f"{result['spec_stats']['satisfied_count']}/{result['spec_stats']['total_specs']}",
            "spec_total_shortage_kg": result["spec_stats"]["total_shortage_kg"],
        },
        "reference_value": ref_value,
    }


def main():
    parser = argparse.ArgumentParser(description="硅钢母卷断刀方案评估器 (tbdg)")
    parser.add_argument("--data-dir", type=Path, required=True)
    parser.add_argument(
        "--baseline-dir",
        default=Path(__file__).resolve().parent / "baseline",
        type=Path,
    )
    parser.add_argument("--submission-dir", type=Path, required=True)
    args = parser.parse_args()

    out = evaluate(
        args.data_dir.resolve(),
        args.baseline_dir.resolve(),
        args.submission_dir.resolve(),
    )
    print(json.dumps(out, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
