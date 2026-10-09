"""
EXTRACTOR_SPEC:
  plan_file: predictions.csv
  required_columns: [Timestamp, predicted_traffic]
  notes: >
    大模型推理服务小时级流量预测。选手产物是一个 CSV：对 data/test_timestamps.csv
    里每个小时各一行，给出预测调用量。

    列名必须恰好是：Timestamp, predicted_traffic

    若选手用了别的列名（如 timestamp / ds / time、Traffic / yhat / pred / forecast），
    映射到规范列名。若选手额外输出了置信区间列，丢弃即可。
    时间列保持选手输出的原样，不要重新格式化。
    不要改动任何预测数值、不要重新建模。
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
TIME_COL = "Timestamp"
VAL_COL = "predicted_traffic"
_HERE = Path(__file__).resolve().parent
_TRUTH = _HERE / "_private" / "test_truth.csv"


# 2026-08-25 修:去掉 quality 的 5.0 封顶 —— 好解一旦超过基准 5 倍就被压平,
# 跟刷分的混在一起分不出高低。分母为零(完美解)时给一个有限哨兵值,避免 inf 破坏 JSON。
PERFECT_Q = 1e6

def _norm_time(s: pd.Series) -> pd.Series:
    t = pd.to_datetime(s.astype(str).str.strip(), errors="coerce")
    return t.dt.strftime("%Y-%m-%d %H:%M")


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
        for c in [TIME_COL, VAL_COL]:
            if c not in pred.columns:
                m["error_info"] = {"fatal": [f"{PLAN_FILE} 缺列 {c}, 实际: {list(pred.columns)}"]}
                return m

        pred["_t"] = _norm_time(pred[TIME_COL])
        pred[VAL_COL] = pd.to_numeric(pred[VAL_COL], errors="coerce")
        if pred["_t"].isnull().any():
            m["error_info"] = {"fatal": ["Timestamp 存在无法解析的值"]}
            return m
        if pred[VAL_COL].isnull().any():
            m["error_info"] = {"fatal": ["predicted_traffic 含非数值/缺失"]}
            return m
        v = pred[VAL_COL].values.astype(float)
        if not np.isfinite(v).all():
            m["error_info"] = {"fatal": ["预测值含 inf/nan"]}
            return m
        if (v <= 0).any():
            m["error_info"] = {"fatal": ["预测值须为正数"]}
            return m
        if pred["_t"].duplicated().any():
            m["error_info"] = {"fatal": ["Timestamp 重复"]}
            return m

        truth = pd.read_csv(_TRUTH, encoding="utf-8-sig")
        truth["_t"] = _norm_time(truth[TIME_COL])
        need = set(truth["_t"])
        if need - set(pred["_t"]):
            miss = sorted(need - set(pred["_t"]))[:3]
            m["error_info"] = {"fatal": [f"缺 {len(need - set(pred['_t']))} 个时间点, 例: {miss}"]}
            return m

        m["validity_score"] = 1.0

        df = truth[["_t", "Traffic"]].merge(pred[["_t", VAL_COL]], on="_t", how="left")
        yt = df["Traffic"].values.astype(float)
        yp = df[VAL_COL].values.astype(float)
        # 流量加权 RMSE: 客户要的是"别在大流量的时候翻车", 高峰时段同样大小的
        # 偏差要比低谷更疼。权重取该小时的真实流量, 再归一化回流量量纲。
        w = np.clip(yt, 0.0, None)
        wsum = float(w.sum())
        if wsum > 0:
            rmse = float(np.sqrt(np.sum(w * (yt - yp) ** 2) / wsum))
        else:
            rmse = float(np.sqrt(np.mean((yt - yp) ** 2)))

        base = load_baseline()
        quality = (base / rmse) if rmse > 0 else PERFECT_Q

        m["quality_score"] = round(quality, 6)
        m["overall_score"] = m["quality_score"]
        m["player_objective"] = round(rmse, 3)
        m["reference_value"] = base
        m["mae"] = round(float(np.mean(np.abs(yt - yp))), 3)
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
