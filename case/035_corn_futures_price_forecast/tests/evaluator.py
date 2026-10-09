"""
EXTRACTOR_SPEC:
  plan_file: predictions.csv
  required_columns: [date, pred_close]
  notes: >
    玉米期货价格多期预测。历史特征到 2025-01-01 为止（data/train.csv），要求预测
    之后 89 个交易日（2025-01-06 ~ 2025-05-22）每一天的收盘价。选手产物是一个 CSV：
    对 data/test_features.csv 里每个交易日一行，给出该日的收盘价预测。

    列名必须恰好是：date, pred_close

    若选手用了别的列名（如 close / pred / forecast / y_pred / 预测收盘），映射到
    pred_close。若选手输出的是相对起点的涨跌幅而非绝对价格，直接照搬数值
    （不要自行换算）。不要改动任何预测数值、不要重新建模。
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
# 2026-08-23 改题:原形态是「89 行 × 各自预测 t+5 / t+20 的收盘价」,而预测跨度落在
# 测试窗口内部 —— 第 i 行的 t+5 目标就是第 i+5 行,同时 test_features.csv 又给了每一天
# 的同日观测(dce_return / ma5 / macd / OHLC …),于是第 i 行的答案字面上印在第 i+5 行的
# 特征里。实测沿 dce_return 从训练末收盘价连乘可**零误差**复原测试期每日收盘,y5/y20
# 偏差均为 0.0000,照此提交 quality 直接撞 5.0 封顶。删列救不了:剔掉 22 个精确还原列后,
# 用剩余 266 列仍能样本外 nowcast 当日收盘到 0.69%(诚实基线 5.04%),漏洞只是从"精确"
# 降到"很准"。根因是结构性的 —— 连续窗口 + 跨度落窗口内 + 同日特征,三者并存必漏。
# 现改为**单一起点多期预测**:特征只到 2025-01-01(train.csv),要求预测之后 89 个交易日
# 每一天的收盘价。公开的 test_features.csv 只留 date 与 horizon,不含任何同日观测,
# 文件里根本不存在"未来那一天的观测",漏无可漏。
TARGET_PRED, TARGET_TRUE = "pred_close", "close"
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
        cols = ["date", TARGET_PRED]
        miss = [c for c in cols if c not in pred.columns]
        if miss:
            m["error_info"] = {"fatal": [f"{PLAN_FILE} 缺列: {miss}"]}
            return m

        pred[TARGET_PRED] = pd.to_numeric(pred[TARGET_PRED], errors="coerce")
        if pred[[TARGET_PRED]].isnull().any().any():
            m["error_info"] = {"fatal": ["预测值含非数值/缺失"]}
            return m
        v = pred[[TARGET_PRED]].values.astype(float)
        if not np.isfinite(v).all():
            m["error_info"] = {"fatal": ["预测值含 inf/nan"]}
            return m
        if (v <= 0).any():
            m["error_info"] = {"fatal": ["预测价格须为正数"]}
            return m

        truth = pd.read_csv(_TRUTH, encoding="utf-8-sig")
        for df in (truth, pred):
            df["date"] = pd.to_datetime(df["date"], errors="coerce").dt.strftime("%Y-%m-%d")
        if pred["date"].duplicated().any():
            m["error_info"] = {"fatal": ["date 重复"]}
            return m

        merged = truth.merge(pred[cols], on="date", how="left")
        if merged[[TARGET_PRED]].isnull().any().any():
            n = int(merged[TARGET_PRED].isnull().sum())
            m["error_info"] = {"fatal": [f"缺 {n} 个交易日的预测"]}
            return m

        m["validity_score"] = 1.0

        yt = merged[TARGET_TRUE].values.astype(float)
        yp = merged[TARGET_PRED].values.astype(float)
        ape = np.abs((yt - yp) / yt) * 100.0
        avg = float(ape.mean())

        base = load_baseline()
        m["quality_score"] = round((base / avg) if avg > 0 else PERFECT_Q, 6)
        m["overall_score"] = m["quality_score"]
        m["player_objective"] = round(avg, 6)
        m["reference_value"] = base
        # 分段诊断:前 20 / 中 20~60 / 后 60+ 个交易日,看误差随预测跨度怎么涨
        h = merged["horizon"].values.astype(int) if "horizon" in merged else np.arange(1, len(ape) + 1)
        m["mape_by_span"] = {
            "h1_20": round(float(ape[h <= 20].mean()), 4),
            "h21_60": round(float(ape[(h > 20) & (h <= 60)].mean()), 4),
            "h61_": round(float(ape[h > 60].mean()), 4),
        }
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
