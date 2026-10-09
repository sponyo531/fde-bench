"""
EXTRACTOR_SPEC:
  plan_file: submission.csv
  required_columns: [row_id, conductivity, activation, density]
  notes: >
    MLE-bench style forward-prediction task. The agent submits PREDICTIONS (a csv),
    not code. train + test are both given to the agent under data/ (test has features
    only, no targets); the private held-out targets live in tests/_private/test_truth.csv
    and are NEVER visible to the agent.

    - data/train.xlsx : labeled rows (raw, dirty) — composition + process features +
      the 3 raw target columns + row_id.
    - data/test.xlsx  : the rows to predict — same raw/dirty features + row_id, NO
      target columns (and none of the leakage columns 是否单相 / 主相类型).

    Deliverable to produce: submission.csv with columns EXACTLY
        [row_id, conductivity, activation, density]
    one row per data/test.xlsx data row (98 rows; row 0 of test.xlsx is a legend line
    with a null row_id — not a prediction row). The evaluator reads these column names
    verbatim (no fuzzy matching) and joins to the private truth by row_id, so exact
    names + correct row_id are mandatory and are YOUR (the extractor's) responsibility.

    Your job: find the agent's prediction output (any name / format — csv, xlsx, a table
    printed in a report, etc.), reshape it into submission.csv. Align predictions to test
    rows by this priority:
      1. If the agent output has a row_id / id column → join on it (best).
      2. Else if the agent output has exactly 98 prediction rows → attach data/test.xlsx's
         row_id BY POSITION/ORDER (read data/test.xlsx, drop its null-row_id legend row,
         take the remaining row_id list in order).
      3. Else (no row_id and row count != 98) → cannot align safely; set status="partial"
         and explain. Do NOT fabricate or pad predictions.
    Rename the agent's target columns to conductivity / activation / density (it may use
    any spelling: 总离子电导率/σ/sigma; 活化能/Ea; 相对密度/ρ; …). Do NOT alter the
    predicted VALUES (no rescaling / unit conversion) — units are the agent's responsibility
    (conductivity in 1e-4 S/cm, density in %). A wrong-but-well-formed prediction is a
    legitimate low score, not something to fix.
"""

import argparse
import json
import traceback
from pathlib import Path

import numpy as np
import pandas as pd

TARGETS = ["conductivity", "activation", "density"]
WEIGHTS = {"conductivity": 0.5, "activation": 0.3, "density": 0.2}
LOG1P_TARGETS = {"conductivity"}
KEY = "row_id"
PLAN_FILE = "submission.csv"

_HERE = Path(__file__).resolve().parent
_PRIV = _HERE / "_private"
_TRUTH = _PRIV / "test_truth.csv"


def load_baseline():
    with open(_HERE / "baseline" / "reference_metrics.json", encoding="utf-8") as f:
        return float(json.load(f)["reference_value"])


def _one_minus_wape(y_true, y_pred):
    y_true = np.asarray(y_true, float); y_pred = np.asarray(y_pred, float)
    s = np.sum(np.abs(y_true))
    return 0.0 if s == 0 else 1.0 - np.sum(np.abs(y_true - y_pred)) / s


def evaluate(data_dir, submission_dir):
    # 防御: 调用方若按 (submission_dir, data_dir) 传反(两参同为路径, 传反不报错只退化成
    # validity=0), 这里探测并换回来, 保证两种调用顺序都能正确读到 submission.csv。
    if not (Path(submission_dir) / PLAN_FILE).is_file() and (Path(data_dir) / PLAN_FILE).is_file():
        data_dir, submission_dir = submission_dir, data_dir
    m = {"validity_score": 0.0, "quality_score": 0.0, "overall_score": 0.0, "error_info": {}}
    try:
        sub_path = Path(submission_dir) / PLAN_FILE
        if not sub_path.is_file():
            m["error_info"] = {"fatal": [f"未找到 {PLAN_FILE}"]}; return m

        pred = pd.read_csv(sub_path)

        need = [KEY] + TARGETS
        miss = [c for c in need if c not in pred.columns]
        if miss:
            m["error_info"] = {"fatal": [f"预测缺列: {miss}"]}; return m
        pred[KEY] = pd.to_numeric(pred[KEY], errors="coerce")
        for t in TARGETS:
            pred[t] = pd.to_numeric(pred[t], errors="coerce")
        if pred[need].isnull().any().any() or not np.isfinite(pred[TARGETS].values).all():
            m["error_info"] = {"fatal": ["预测含缺失/非有限值"]}; return m
        pred[KEY] = pred[KEY].astype(int)
        if pred[KEY].duplicated().any():
            m["error_info"] = {"fatal": ["预测有重复 row_id"]}; return m

        truth = pd.read_csv(_TRUTH); truth[KEY] = truth[KEY].astype(int)
        missing_ids = set(truth[KEY]) - set(pred[KEY])
        if missing_ids:
            m["error_info"] = {"fatal": [f"预测未覆盖 {len(missing_ids)} 个 test row_id"]}; return m
        m["validity_score"] = 1.0

        merged = truth.merge(pred[need], on=KEY, how="left", suffixes=("_true", "_pred"))
        per = {}
        for t in TARGETS:
            yt = merged[f"{t}_true"].values; yp = merged[f"{t}_pred"].values
            mask = ~np.isnan(yt)
            if mask.sum() == 0:
                per[t] = 0.0; continue
            a, b = yt[mask], yp[mask]
            if t in LOG1P_TARGETS:
                a = np.log1p(np.clip(a, 0, None)); b = np.log1p(np.clip(b, 0, None))
            per[t] = float(_one_minus_wape(a, b))

        combined = sum(WEIGHTS[t] * per[t] for t in TARGETS)
        baseline = load_baseline()
        # quality floor at 0: 1-WAPE can be negative when the prediction is worse than
        # a constant guess, but the quality score must never dip below zero.
        quality = max(0.0, combined) / baseline if baseline > 0 else 0.0
        m["quality_score"] = round(quality, 6)
        m["overall_score"] = round(quality, 6)
        m["combined_score"] = round(combined, 6)
        m["per_target_1_wape"] = {k: round(v, 4) for k, v in per.items()}
        m["reference_value"] = baseline
        return m
    except Exception as e:
        m["error_info"] = {"exception": str(e), "traceback": traceback.format_exc()[-1200:]}
        return m


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--submission-dir", required=True)
    ap.add_argument("--data-dir", default=str(_HERE.parent / "data"))
    a = ap.parse_args()
    print(json.dumps(evaluate(a.data_dir, a.submission_dir), ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
