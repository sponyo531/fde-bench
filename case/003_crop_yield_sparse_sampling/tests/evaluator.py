"""
EXTRACTOR_SPEC:
  plan_file: submission.csv
  required_columns: [year, state_id, county_id, predicted_yield]
  notes: >
    The agent trains ONE model on data/train.npz (has labels, years < 2020),
    predicts a county-level soybean yield for every test row in data/test.npz
    (no labels; years 2020..2024), and writes a submission.csv with exactly four
    columns: `year`, `state_id`, `county_id` (the test row's identity) and
    `predicted_yield` (finite float, bushels/acre, roughly 7-80). Exactly one row
    per test (year, state_id, county_id).

    Locate the agent's prediction CSV in the workspace (prefer a file literally
    named submission.csv; otherwise the most recent .csv holding one prediction
    per test (year, state_id, county_id)). Normalize its column names to exactly
    `year`, `state_id`, `county_id`, `predicted_yield` (common aliases: state ->
    state_id; county -> county_id; pred/yield/prediction/predicted -> predicted_
    yield). Copy the result VERBATIM as submission.csv. Do NOT retrain,
    re-predict, re-order, run any subprocess, or fabricate rows — the evaluator
    reads the CSV and scores per-year RMSE against private truth on its own.

PSA3 Soybean Yield Prediction — multi-year forward RMSE CSV scorer.
Interface (skill contract):
    evaluate(submission_dir, data_dir) -> {validity_score, quality_score,
                                            overall_score, error_info, ...}
"""

import argparse
import csv
import json
import math
from pathlib import Path

import numpy as np

TOL = 1e-9
PLAN_FILE = "submission.csv"
REQUIRED_COLUMNS = ["year", "state_id", "county_id", "predicted_yield"]

# ── Business constants (internal; not disclosed to the agent) ───────────────
TEST_YEARS = [2020, 2021, 2022, 2023, 2024]
STATES_11 = frozenset({5, 17, 18, 19, 20, 27, 29, 31, 38, 39, 46})

_HERE = Path(__file__).resolve().parent
PRIVATE_DIR = _HERE / "_private"
TRUTH_NPZ = PRIVATE_DIR / "test_truth.npz"
BASELINE_JSON = _HERE / "baseline" / "reference_metrics.json"


# 2026-08-25 修:去掉 quality 的 5.0 封顶 —— 好解一旦超过基准 5 倍就被压平,
# 跟刷分的混在一起分不出高低。分母为零(完美解)时给一个有限哨兵值,避免 inf 破坏 JSON。
PERFECT_Q = 1e6

def resolve_plan_path(submission_dir: Path):
    preferred = submission_dir / PLAN_FILE
    if preferred.is_file():
        return preferred, []
    matches = [p for p in submission_dir.glob(f"**/{PLAN_FILE}") if p.is_file()]
    if not matches:
        return None, [f"{PLAN_FILE} not found under {submission_dir}."]
    return matches[0], []


def load_truth():
    """{(year, state_id, county_id): true_yield} from the private truth npz."""
    d = np.load(TRUTH_NPZ)
    year = d["year"].astype(np.int64)
    state_id = d["state_id"].astype(np.int64)
    county_id = d["county_id"].astype(np.int64)
    ys = d["yield_true"].astype(np.float64)
    return {
        (int(y), int(s), int(c)): float(v)
        for y, s, c, v in zip(year, state_id, county_id, ys)
    }


def load_reference():
    with BASELINE_JSON.open("r", encoding="utf-8") as f:
        d = json.load(f)
    return float(d["reference_value"]), d.get("direction", "higher_is_better")


def load_submission(plan_path: Path):
    with plan_path.open("r", encoding="utf-8-sig") as f:
        return list(csv.DictReader(f))


def evaluate(submission_dir, data_dir) -> dict:
    submission_dir = Path(submission_dir)

    reference_value, direction = load_reference()
    truth = load_truth()  # {(year, state, county): yield}

    result = {
        "validity_score": 0.0,
        "quality_score": 0.0,
        "overall_score": 0.0,
        "error_info": {},
        "reference_value": reference_value,
    }

    # ── locate submission.csv ──────────────────────────────────────────────
    plan_path, errs = resolve_plan_path(submission_dir)
    if errs:
        result["error_info"] = {"fatal": errs}
        return result
    result["resolved_artifact"] = str(plan_path)

    try:
        rows = load_submission(plan_path)
    except Exception as e:
        result["error_info"] = {"fatal": [f"Failed to read {plan_path.name}: {e}"]}
        return result

    if not rows:
        result["error_info"] = {"fatal": ["Empty submission file."]}
        return result

    # ── HC1: required columns present ───────────────────────────────────────
    missing_cols = [c for c in REQUIRED_COLUMNS if c not in rows[0].keys()]
    if missing_cols:
        result["error_info"] = {"constraint": [f"HC1: missing required columns {missing_cols}"]}
        return result

    # ── HC2: parse rows; keys integer, values finite numeric; HC3 no dups ───
    pred_map = {}
    errors = []
    for i, row in enumerate(rows, start=1):
        try:
            yr = int(float(row["year"]))
            st = int(float(row["state_id"]))
            co = int(float(row["county_id"]))
            val = float(row["predicted_yield"])
        except (KeyError, ValueError, TypeError) as exc:
            errors.append(f"HC2: unparseable row #{i}: {exc}")
            continue
        if math.isnan(val) or math.isinf(val):
            errors.append(f"HC2: non-finite predicted_yield at row #{i} (year={yr}, state={st}, county={co}).")
            continue
        key = (yr, st, co)
        if key in pred_map:
            errors.append(f"HC3: duplicate (year,state_id,county_id)={key} at row #{i}.")
            continue
        pred_map[key] = val
    if errors:
        result["error_info"] = {"constraint": errors[:12]}
        return result

    # ── HC4: every test key present exactly once, no extras ─────────────────
    missing_keys = [k for k in truth if k not in pred_map]
    extra_keys = [k for k in pred_map if k not in truth]
    if missing_keys:
        errors.append(f"HC4: missing {len(missing_keys)} test (year,state,county) row(s) (first 5: {sorted(missing_keys)[:5]}).")
    if extra_keys:
        errors.append(f"HC4: {len(extra_keys)} extra (year,state,county) row(s) not in test set (first 5: {sorted(extra_keys)[:5]}).")
    if errors:
        result["error_info"] = {"constraint": errors[:12]}
        return result

    # ── Per-year RMSE, then mean of the 5 per-year RMSEs (NOT pooled) ───────
    per_year_rmse = {}
    rmses = []
    for yr in TEST_YEARS:
        keys = [k for k in truth if k[0] == yr]
        if not keys:
            result["error_info"] = {"constraint": [f"HC4: no truth rows for year {yr}"]}
            return result
        y_true = np.array([truth[k] for k in keys], dtype=np.float64)
        y_pred = np.array([pred_map[k] for k in keys], dtype=np.float64)
        rmse = float(np.sqrt(np.mean((y_pred - y_true) ** 2)))
        if not math.isfinite(rmse) or rmse < 0:
            result["error_info"] = {"constraint": [f"HC2: invalid RMSE {rmse} for year {yr}"]}
            return result
        per_year_rmse[yr] = round(rmse, 6)
        rmses.append(rmse)

    mean_rmse = float(np.mean(rmses))
    if not math.isfinite(mean_rmse):
        result["error_info"] = {"constraint": ["HC2: non-finite mean_rmse"]}
        return result

    # mean_rmse 越小越好, 故 quality = reference_value / player_mean_rmse
    if direction == "higher_is_better":
        quality = mean_rmse / reference_value if reference_value > TOL else 0.0
    elif mean_rmse > TOL:
        quality = reference_value / mean_rmse
    else:
        # 完美预测(mean_rmse≈0)时不再产生 inf: 满分按基准惯例封顶 5.0。
        # 旧 float("inf") 是泄题刷分入口(真值一旦泄露就能拿无限分)。
        quality = PERFECT_Q if reference_value > TOL else 1.0

    result["validity_score"] = 1.0
    result["quality_score"] = round(quality, 6)
    result["overall_score"] = result["quality_score"]
    result["mean_rmse"] = round(mean_rmse, 6)
    result["per_year_rmse"] = per_year_rmse
    result["n_test"] = len(truth)
    return result


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--submission-dir", required=True)
    p.add_argument("--data-dir", default=str(_HERE.parent / "data"))
    a = p.parse_args()
    out = evaluate(a.submission_dir, a.data_dir)
    print(json.dumps(out, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
