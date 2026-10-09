"""
EXTRACTOR_SPEC:
  plan_file: submission.csv
  required_columns: [sample_id, horizon, bkd_pred]
  notes: >
    MLE-bench style flight-booking demand forecasting with a fixed time-based
    train/test split. The agent submits PREDICTIONS (a csv), not code.

    - data/train.csv       : full daily booking-curve rows for every departure whose
      FLIGHT_DATE is before the cutoff (all columns, all observation days). Use for training.
    - data/test_history.csv: for each held-out (late-departure) test flight, the EARLY part
      of its booking curve only — the final 7 observation days (the target window near
      departure) are removed. Use as the model input for the test samples.
    - data/test_samples.csv: the list of test samples to predict. One row per sample:
      sample_id, FLIGHT_NO, FLIGHT_DATE. For each sample the agent must predict BKD for the
      7 held-out observation days (the last week of the booking curve, ordered by
      OBSERVATION_DATE), i.e. horizon = 1..7.

    Deliverable to produce: submission.csv with columns EXACTLY
        sample_id, horizon, bkd_pred
    one row per (sample_id in test_samples.csv, horizon in 1..7) — 7 rows per sample,
    covering EVERY test sample. `bkd_pred` is the predicted total booked count (BKD),
    a non-negative number. The evaluator reads these column names verbatim (no fuzzy
    matching) and joins to the private truth by (sample_id, horizon), so exact names +
    full coverage are mandatory and are YOUR (the extractor's) responsibility.

    Your job: find the agent's prediction output (any name / format) and reshape it into
    submission.csv keyed by (sample_id, horizon):
    - a wide table with columns bkd_1..bkd_7 (or day1..day7) per sample -> melt to long
      (sample_id, horizon, bkd_pred).
    - a numpy [N,7] array or per-sample list -> attach sample_id from test_samples.csv row
      order ONLY if the agent explicitly kept that order; prefer an explicit key column.
    - a column named pred / y_pred / BKD -> rename to bkd_pred; map the sample/id and the
      horizon/day columns to sample_id / horizon.
    - horizon stored 0..6 -> shift to 1..7; sample_id / horizon stored as float -> cast int.
    Align strictly by (sample_id, horizon); do NOT attach ids by row position unless the
    agent's output is unambiguously in test_samples.csv order. If the agent output lacks a
    usable (sample_id, horizon) key, set status="partial" and explain.
    Do NOT alter predicted values (no rescaling / clipping); a wrong-but-well-formed
    prediction is a legitimate low score, not something to fix.
"""
import argparse
import json
import os
import traceback
from pathlib import Path

import numpy as np
import pandas as pd

PLAN_FILE = "submission.csv"
KEY = ["sample_id", "horizon"]

_HERE = Path(__file__).resolve().parent
_TRUTH = _HERE / "_private" / "test_truth.csv"   # 评估器私有, 绝不在 data/


def load_baseline():
    with open(_HERE / "baseline" / "reference_metrics.json", encoding="utf-8") as f:
        d = json.load(f)
    return float(d["reference_value"]), d.get("direction", "higher_is_better")


def _safe(arr):
    arr = np.asarray(arr, dtype=np.float64)
    return np.nan_to_num(arr, nan=0.0, posinf=0.0, neginf=0.0)


def _metrics(y_true, y_pred):
    yt, yp = _safe(y_true), _safe(y_pred)
    rmse = float(np.sqrt(np.mean((yt - yp) ** 2)))
    mae = float(np.mean(np.abs(yt - yp)))
    wape = float(np.sum(np.abs(yt - yp)) / max(np.sum(np.abs(yt)), 1e-6))
    denom = np.maximum(np.abs(yt) + np.abs(yp), 1e-6)
    smape = float(np.mean(2.0 * np.abs(yt - yp) / denom))
    combined = 1.0 / (1.0 + mae)          # 与源评估器一致: combined = MAE 的倒数变换
    return combined, {"RMSE": round(rmse, 6), "MAE": round(mae, 6),
                      "WAPE": round(wape, 6), "SMAPE": round(smape, 6),
                      "combined_score": round(combined, 6)}


def evaluate(data_dir, submission_dir):
    m = {"validity_score": 0.0, "quality_score": 0.0, "overall_score": 0.0, "error_info": {}}
    try:
        plan = Path(submission_dir) / PLAN_FILE
        if not plan.is_file():
            m["error_info"] = {"fatal": [f"未找到 {PLAN_FILE}"]}; return m

        pred = pd.read_csv(plan)
        miss = [c for c in ("sample_id", "horizon", "bkd_pred") if c not in pred.columns]
        if miss:
            m["error_info"] = {"fatal": [f"{PLAN_FILE} 缺列: {miss}"]}; return m
        for c in ("sample_id", "horizon", "bkd_pred"):
            pred[c] = pd.to_numeric(pred[c], errors="coerce")
        if pred[["sample_id", "horizon", "bkd_pred"]].isnull().any().any():
            m["error_info"] = {"fatal": ["含非数值/缺失(sample_id/horizon/bkd_pred)"]}; return m
        if not np.isfinite(pred["bkd_pred"]).all():
            m["error_info"] = {"fatal": ["bkd_pred 含 inf/nan"]}; return m
        if (pred["bkd_pred"] < 0).any():
            m["error_info"] = {"fatal": ["bkd_pred 含负值"]}; return m
        pred = pred.astype({"sample_id": "int64", "horizon": "int64"})
        if pred.duplicated(KEY).any():
            n = int(pred.duplicated(KEY).sum())
            m["error_info"] = {"fatal": [f"有 {n} 条重复 (sample_id,horizon)"]}; return m

        truth = pd.read_csv(_TRUTH)
        truth = truth.astype({"sample_id": "int64", "horizon": "int64"})

        # 覆盖性: 每个测试样本的 7 个 horizon 都要有预测
        need = truth[KEY].drop_duplicates()
        chk = need.merge(pred[KEY].drop_duplicates(), on=KEY, how="left", indicator=True)
        missing = int((chk["_merge"] == "left_only").sum())
        if missing:
            miss_sid = sorted(chk.loc[chk["_merge"] == "left_only", "sample_id"].unique())[:5]
            m["error_info"] = {"fatal": [f"预测未覆盖 {missing} 个 (sample_id,horizon)(如样本 {miss_sid})"]}; return m

        m["validity_score"] = 1.0

        df = truth.merge(pred[KEY + ["bkd_pred"]], on=KEY, how="left").sort_values(KEY)
        combined, metrics = _metrics(df["bkd_true"].values, df["bkd_pred"].values)

        baseline, direction = load_baseline()
        if direction == "higher_is_better":
            quality = combined / baseline if baseline > 0 else 0.0
        else:
            quality = baseline / combined if combined > 0 else 0.0
        m["quality_score"] = round(quality, 6)          # 不加 min(,1) 截断
        m["overall_score"] = round(quality, 6)
        m["reference_value"] = baseline
        m["n_test_samples"] = int(need["sample_id"].nunique())
        m.update(metrics)
        return m
    except Exception as e:
        m["validity_score"] = 0.0
        m["error_info"] = {"exception": str(e), "traceback": traceback.format_exc()[-800:]}
        return m


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data-dir", type=Path, default=_HERE.parent / "data")
    ap.add_argument("--submission-dir", type=Path, required=True)
    a = ap.parse_args()
    print(json.dumps(evaluate(str(a.data_dir), str(a.submission_dir)), indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
