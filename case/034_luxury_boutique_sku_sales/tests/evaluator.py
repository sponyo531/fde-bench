"""
EXTRACTOR_SPEC:
  plan_file: predictions.csv
  required_columns: [date, boutique_id, product_id, predicted_qty]
  notes: >
    高端珠宝门店 门店×SKU×日 销量预测。选手产物是一个 CSV：对
    data/test_features.csv 里每一行给出一条预测。

    列名必须恰好是：date, boutique_id, product_id, predicted_qty

    若选手用了别的预测列名（如 sales_qty / pred / yhat / forecast / y_pred），
    映射到 predicted_qty。键列（date/boutique_id/product_id）保持原值，
    不要重新格式化日期。若选手输出了额外列，丢弃即可。
    不要改动任何预测数值、不要重新训练模型、不要对预测取整。
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
KEYS = ["date", "boutique_id", "product_id"]
VAL = "predicted_qty"
_HERE = Path(__file__).resolve().parent
_TRUTH = _HERE / "_private" / "test_truth.csv"


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
        miss = [c for c in KEYS + [VAL] if c not in pred.columns]
        if miss:
            m["error_info"] = {"fatal": [f"{PLAN_FILE} 缺列: {miss}"]}
            return m

        pred[VAL] = pd.to_numeric(pred[VAL], errors="coerce")
        if pred[VAL].isnull().any():
            m["error_info"] = {"fatal": [f"{VAL} 含非数值/缺失"]}
            return m
        v = pred[VAL].values.astype(float)
        if not np.isfinite(v).all():
            m["error_info"] = {"fatal": ["预测值含 inf/nan"]}
            return m
        if (v < 0).any():
            m["error_info"] = {"fatal": ["预测值含负数"]}
            return m

        truth = pd.read_csv(_TRUTH, encoding="utf-8-sig")
        for df in (truth, pred):
            df["date"] = pd.to_datetime(df["date"], errors="coerce").dt.strftime("%Y-%m-%d")
            df["boutique_id"] = df["boutique_id"].astype(str)
            df["product_id"] = df["product_id"].astype(str)

        if pred.duplicated(subset=KEYS).any():
            m["error_info"] = {"fatal": ["存在重复的 (date, boutique_id, product_id)"]}
            return m

        merged = truth.merge(pred[KEYS + [VAL]], on=KEYS, how="left")
        if merged[VAL].isnull().any():
            n = int(merged[VAL].isnull().sum())
            m["error_info"] = {"fatal": [f"缺 {n} 条测试样本的预测"]}
            return m

        m["validity_score"] = 1.0

        yt = merged["sales_qty"].values.astype(float)
        yp = merged[VAL].values.astype(float)

        # 数量项：1 - 归一化 MAE（相对于全零预测的 MAE）
        mae = float(np.mean(np.abs(yt - yp)))
        mae_zero = float(np.mean(np.abs(yt)))
        qty_score = max(0.0, 1.0 - mae / mae_zero) if mae_zero > 0 else 0.0

        # 识别项：把预测 >= 0.5 视作"判定会出货"，与真实非零比 F1
        pos_t = yt > 0
        pos_p = yp >= 0.5
        tp = float(np.sum(pos_t & pos_p))
        fp = float(np.sum(~pos_t & pos_p))
        fn = float(np.sum(pos_t & ~pos_p))
        prec = tp / (tp + fp) if (tp + fp) > 0 else 0.0
        rec = tp / (tp + fn) if (tp + fn) > 0 else 0.0
        f1 = 2 * prec * rec / (prec + rec) if (prec + rec) > 0 else 0.0

        combined = 0.5 * qty_score + 0.5 * f1

        base = load_baseline()
        m["quality_score"] = round(combined / base if base > 0 else 0.0, 6)
        m["overall_score"] = m["quality_score"]
        m["player_objective"] = round(combined, 6)
        m["reference_value"] = base
        m["qty_score"] = round(qty_score, 6)
        m["hit_f1"] = round(f1, 6)
        m["mae"] = round(mae, 6)
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
