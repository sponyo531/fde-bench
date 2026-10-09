"""
EXTRACTOR_SPEC:
  plan_file: prediction.csv
  required_columns: ["数据时间", "总有功功率(kW)"]
  notes: >
    区域电网 15 分钟负荷预测。选手产物是一个 CSV：对 data/test_timestamps.csv 里
    每个时间点各一行，给出预测的总有功功率。

    列名必须恰好是：数据时间, 总有功功率(kW)

    若选手用了别的列名（如 timestamp / datetime / ds、load / power / yhat / prediction），
    映射到上面两个规范列名。时间列的值保持选手输出的原样，不要重新格式化
    （评估器自身会做时间归一化匹配）。不要改动任何预测数值、不要重新建模。
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

PLAN_FILE = "prediction.csv"
TIME_COL = "数据时间"
VAL_COL = "总有功功率(kW)"
_HERE = Path(__file__).resolve().parent
_TRUTH = _HERE / "_private" / "test_truth.csv"
PERFECT_Q = 1e6



def _norm_time(s: pd.Series) -> pd.Series:
    """把各种时间写法归一到统一字符串, 容忍补零/不补零、'-' 与 '/' 混用。"""
    t = pd.to_datetime(s.astype(str).str.strip(), errors="coerce")
    return t.dt.strftime("%Y-%m-%d %H:%M")


def load_baseline() -> tuple:
    with open(_HERE / "baseline" / "reference_metrics.json", encoding="utf-8") as f:
        d = json.load(f)
    return float(d["reference_value"]), d.get("direction", "lower_is_better")


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
                m["error_info"] = {"fatal": [f"{PLAN_FILE} 缺列 {c}, 实际列: {list(pred.columns)}"]}
                return m

        pred["_t"] = _norm_time(pred[TIME_COL])
        pred[VAL_COL] = pd.to_numeric(pred[VAL_COL], errors="coerce")
        if pred["_t"].isnull().any():
            m["error_info"] = {"fatal": ["数据时间 存在无法解析的值"]}
            return m
        if pred[VAL_COL].isnull().any():
            m["error_info"] = {"fatal": ["总有功功率(kW) 含非数值/缺失"]}
            return m
        v = pred[VAL_COL].values.astype(float)
        if not np.isfinite(v).all():
            m["error_info"] = {"fatal": ["预测值含 inf/nan"]}
            return m
        if (v <= 0).any():
            m["error_info"] = {"fatal": ["预测值须为正数"]}
            return m
        if pred["_t"].duplicated().any():
            m["error_info"] = {"fatal": ["数据时间 重复"]}
            return m

        truth = pd.read_csv(_TRUTH, encoding="utf-8-sig")
        truth["_t"] = _norm_time(truth[TIME_COL])
        need = set(truth["_t"])
        have = set(pred["_t"])
        if need - have:
            miss = sorted(need - have)[:3]
            m["error_info"] = {"fatal": [f"缺 {len(need-have)} 个时间点, 例: {miss}"]}
            return m

        m["validity_score"] = 1.0

        df = truth[["_t", VAL_COL]].merge(
            pred[["_t", VAL_COL]], on="_t", how="left", suffixes=("_true", "_pred"))
        yt = df[f"{VAL_COL}_true"].values.astype(float)
        yp = df[f"{VAL_COL}_pred"].values.astype(float)
        mape = float(np.mean(np.abs((yt - yp) / yt)) * 100.0)

        base, direction = load_baseline()
        quality = base / mape if mape > 0 else PERFECT_Q

        m["quality_score"] = round(quality, 6)
        m["overall_score"] = m["quality_score"]
        m["player_objective"] = round(mape, 6)
        m["reference_value"] = base
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
