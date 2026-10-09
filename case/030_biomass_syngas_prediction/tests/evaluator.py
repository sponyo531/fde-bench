"""
EXTRACTOR_SPEC:
  plan_file: predictions.csv
  required_columns: [sample_id, "H2(vol%)", "CO(vol%)", "CO2(vol%)", "CH4(vol%)"]
  notes: >
    生物质气化合成气组分预测。选手产物是一个 CSV：对 data/test_features.csv 里每个
    sample_id 各一行，给出四种气体的体积分数预测。

    列名必须恰好是：sample_id, H2(vol%), CO(vol%), CO2(vol%), CH4(vol%)
    （带单位后缀的列名与 train.csv 的目标列名一致）。

    若选手输出的列名不同（如 H2 / h2_pred / pred_H2），映射到上面的规范列名；
    若选手把四个组分拆成四个文件，合并成一个 CSV 按 sample_id 对齐。
    不要改动预测数值、不要重新训练模型。
"""
from __future__ import annotations

import argparse
import json
import os
import traceback
from pathlib import Path
from typing import Any, Dict

import numpy as np
import pandas as pd

PLAN_FILE = "predictions.csv"
TARGETS = ["H2(vol%)", "CO(vol%)", "CO2(vol%)", "CH4(vol%)"]
_HERE = Path(__file__).resolve().parent
_TRUTH = _HERE / "_private" / "test_truth.csv"


def load_baseline() -> tuple:
    with open(_HERE / "baseline" / "reference_metrics.json", encoding="utf-8") as f:
        d = json.load(f)
    return float(d["reference_value"]), d.get("direction", "higher_is_better")


def evaluate(submission_dir: str, data_dir: str) -> Dict[str, Any]:
    m: Dict[str, Any] = {
        "validity_score": 0.0, "quality_score": 0.0, "overall_score": 0.0, "error_info": {},
    }
    try:
        plan = os.path.join(submission_dir, PLAN_FILE)
        if not os.path.exists(plan):
            m["error_info"] = {"fatal": [f"缺 {PLAN_FILE}"]}
            return m

        pred = pd.read_csv(plan, encoding="utf-8-sig")
        miss = [c for c in ["sample_id"] + TARGETS if c not in pred.columns]
        if miss:
            m["error_info"] = {"fatal": [f"{PLAN_FILE} 缺列: {miss}"]}
            return m

        truth = pd.read_csv(_TRUTH, encoding="utf-8-sig")
        pred["sample_id"] = pd.to_numeric(pred["sample_id"], errors="coerce")
        for c in TARGETS:
            pred[c] = pd.to_numeric(pred[c], errors="coerce")

        if pred[["sample_id"] + TARGETS].isnull().any().any():
            m["error_info"] = {"fatal": ["含非数值/缺失"]}
            return m
        if not np.isfinite(pred[TARGETS].values).all():
            m["error_info"] = {"fatal": ["预测值含 inf/nan"]}
            return m
        if (pred[TARGETS].values < 0).any():
            m["error_info"] = {"fatal": ["预测值含负数"]}
            return m
        if (pred[TARGETS].values > 1).any():
            m["error_info"] = {"fatal": ["体积分数必须落在 [0,1] 内"]}
            return m
        if pred["sample_id"].duplicated().any():
            m["error_info"] = {"fatal": ["sample_id 重复"]}
            return m

        need = set(truth["sample_id"].astype(int))
        have = set(pred["sample_id"].astype(int))
        if need - have:
            missing = sorted(need - have)[:5]
            m["error_info"] = {"fatal": [f"缺 {len(need-have)} 个样本, 例: {missing}"]}
            return m

        m["validity_score"] = 1.0

        df = truth.merge(pred[["sample_id"] + TARGETS], on="sample_id",
                         how="left", suffixes=("_true", "_pred"))
        r2s = {}
        for c in TARGETS:
            yt = df[f"{c}_true"].values.astype(float)
            yp = df[f"{c}_pred"].values.astype(float)
            ss_res = float(np.sum((yt - yp) ** 2))
            ss_tot = float(np.sum((yt - yt.mean()) ** 2))
            r2s[c] = 1.0 - ss_res / ss_tot if ss_tot > 0 else 0.0
        mean_r2 = float(np.mean(list(r2s.values())))

        baseline, direction = load_baseline()
        if direction == "higher_is_better":
            # R2 can be arbitrarily negative for a terrible prediction; a negative
            # mean_r2 must map to quality 0, never to a negative score.
            quality = max(0.0, mean_r2) / baseline if baseline > 0 else 0.0
        else:
            quality = baseline / mean_r2 if mean_r2 > 0 else 0.0

        m["quality_score"] = round(quality, 6)
        m["overall_score"] = m["quality_score"]
        m["player_objective"] = round(mean_r2, 6)
        m["reference_value"] = baseline
        m["r2_by_component"] = {k: round(v, 6) for k, v in r2s.items()}
        return m

    except Exception as e:
        m["validity_score"] = 0.0
        m["error_info"] = {"exception": str(e), "traceback": traceback.format_exc()[-800:]}
        return m


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--submission-dir", required=True)
    ap.add_argument("--data-dir", default=str(_HERE.parent / "data"))
    a = ap.parse_args()
    print(json.dumps(evaluate(a.submission_dir, a.data_dir), ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
