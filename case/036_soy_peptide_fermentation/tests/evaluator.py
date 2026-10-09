"""
EXTRACTOR_SPEC:
  plan_file: predictions.csv
  required_columns: [batch_id, "成品小肽"]
  notes: >
    大豆发酵成品小肽预测。选手产物是一个 CSV：对 data/test_batches.csv 里每个
    batch_id 各一行，给出成品小肽预测值。

    列名必须恰好是：batch_id, 成品小肽

    若选手用了别的预测列名（如 pred / predicted / y_pred / peptide / 预测值），
    映射到 成品小肽。不要改动任何预测数值、不要重新训练模型。
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
ID_COL = "batch_id"
VAL_COL = "成品小肽"
_HERE = Path(__file__).resolve().parent
_TRUTH = _HERE / "_private" / "test_truth.csv"


# 2026-08-25 修:去掉 quality 的 5.0 封顶 —— 好解一旦超过基准 5 倍就被压平,
# 跟刷分的混在一起分不出高低。分母为零(完美解)时给一个有限哨兵值,避免 inf 破坏 JSON。
PERFECT_Q = 1e6

def load_baseline() -> float:
    with open(_HERE / "baseline" / "reference_metrics.json", encoding="utf-8") as f:
        return float(json.load(f)["reference_value"])


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
        for c in [ID_COL, VAL_COL]:
            if c not in pred.columns:
                m["error_info"] = {"fatal": [f"{PLAN_FILE} 缺列 {c}, 实际: {list(pred.columns)}"]}
                return m

        pred[VAL_COL] = pd.to_numeric(pred[VAL_COL], errors="coerce")
        if pred[VAL_COL].isnull().any():
            m["error_info"] = {"fatal": [f"{VAL_COL} 含非数值/缺失"]}
            return m
        v = pred[VAL_COL].values.astype(float)
        if not np.isfinite(v).all():
            m["error_info"] = {"fatal": ["预测值含 inf/nan"]}
            return m
        if (v <= 0).any():
            m["error_info"] = {"fatal": ["预测值须为正数"]}
            return m
        if pred[ID_COL].duplicated().any():
            m["error_info"] = {"fatal": [f"{ID_COL} 重复"]}
            return m

        truth = pd.read_csv(_TRUTH, encoding="utf-8-sig")
        truth[ID_COL] = truth[ID_COL].astype(str)
        pred[ID_COL] = pred[ID_COL].astype(str)
        need = set(truth[ID_COL])
        if need - set(pred[ID_COL]):
            m["error_info"] = {"fatal": [f"缺 {len(need - set(pred[ID_COL]))} 个批次"]}
            return m

        m["validity_score"] = 1.0

        df = truth.merge(pred[[ID_COL, VAL_COL]], on=ID_COL, how="left",
                         suffixes=("_true", "_pred"))
        yt = df[f"{VAL_COL}_true"].values.astype(float)
        yp = df[f"{VAL_COL}_pred"].values.astype(float)
        mae = float(np.mean(np.abs(yt - yp)))
        rmse = float(np.sqrt(np.mean((yt - yp) ** 2)))

        base = load_baseline()
        m["quality_score"] = round((base / mae) if mae > 0 else PERFECT_Q, 6)
        m["overall_score"] = m["quality_score"]
        m["player_objective"] = round(mae, 6)
        m["reference_value"] = base
        m["rmse"] = round(rmse, 6)
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
