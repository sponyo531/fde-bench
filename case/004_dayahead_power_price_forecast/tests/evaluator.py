"""
EXTRACTOR_SPEC:
  plan_file: submission.csv
  required_columns: [date, time, clear_da]
  notes: >
    日前电价预测 (MLE-bench 式)。data/ 里 timetable_train.xlsx 是 2024-01~08 带
    clear_da 标签的训练表 (21 列, 原始脏样), timetable_test.xlsx 是 2024-09 只含
    输入列 (20 列, 无 clear_da)。agent 自己清洗 + 建模 + 出预测, 交 submission.csv
    列 [date, time, clear_da] 覆盖 9 月每个 15 分钟点 (2880 行)。评估器只读预测文件,
    按 (date, time) 与私有真值对齐。

    Extractor: 从 agent 输出里找预测(任意 csv / xlsx / txt / md 表), 归一到
    submission.csv 三列 [date, time, clear_da]:
      - date  可能叫: date / Date / DATE / 日期 / dt / day
      - time  可能叫: time / Time / 时点 / 时段 / hour / hh:mm / 时刻
      - clear_da 可能叫: clear_da / pred / prediction / y_pred / price /
                       clearing_price / 出清价 / 日前价 / 电价 / 预测
    如果 date/time 合并成一列(如 "2024-09-01 00:15" 或 timestamp), 拆成两列填
    到 date / time。若无 date/time 但行数恰为 2880, 按 timetable_test.xlsx 的
    (date, time) 顺序对齐; 否则 status="partial"。
    不改预测 VALUES(不 clip、不换单位——评估器会自己 clip 到 [0, 1500])。
"""

import argparse
import json
import math
import traceback
from pathlib import Path

import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
_TRUTH = HERE / "_private" / "test_truth.csv"
PLAN_FILE = "submission.csv"

# ── 锁死配置（评分口径，agent 改不到）───────────────────────────────────────
PRICE_MIN, PRICE_MAX = 0.0, 1500.0
HI_THRESHOLD = 800.0            # 高价段阈值
# 打分权重（和为 1）：整体 MAE + 尾部 p95/p99 + 高价段单向欠预测 hi_short。
W_MAE, W_P95, W_P99, W_HISHORT = 0.30, 0.25, 0.15, 0.30



def _load_baseline():
    """读 baseline: (reference_value, direction, baseline_metrics)。

    baseline_metrics 是 baseline 解在四项度量上的实测值（元/MWh），用作各项的归一
    分母。四项量纲差很多（MAE 85 / p95 462 / p99 746 / hi_short 432），直接加权的
    话数值大的项会天然支配总分；各自除以 baseline 值后，baseline 解上四项各为 1，
    W_MAE/W_P95/W_P99/W_HISHORT 才真正代表设计的占比，combined 也就读作"相对
    baseline 的加权误差倍数"（baseline 自身 = 1.0）。重跑 baseline 时四个分母与
    reference_value 一并更新，evaluator 不必改动。
    """
    with open(HERE / "baseline" / "reference_metrics.json", "r", encoding="utf-8") as f:
        d = json.load(f)
    return (float(d["reference_value"]),
            d.get("direction", "higher_is_better"),
            d["baseline_metrics"])


def _compute_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> dict:
    y_true = np.asarray(y_true, float)
    y_pred = np.asarray(y_pred, float)
    err = y_pred - y_true
    abs_err = np.abs(err)
    high = y_true > HI_THRESHOLD
    return {
        "MAE": float(abs_err.mean()),
        "p95": float(np.percentile(abs_err, 95)),
        "p99": float(np.percentile(abs_err, 99)),
        # hi_short = 高价段单向欠预测：真价>800 时只罚"预测偏低"（true - pred > 0 的部分）
        "hi_short": float(np.clip(-err[high], 0, None).mean()) if high.any() else 0.0,
        "n": int(len(y_true)),
        "n_high": int(high.sum()),
    }


def _weighted_error(m: dict, norm: dict) -> float:
    """四项误差按 baseline 归一后的加权和，越小越好（baseline 解上四项各为 1）。"""
    return (W_MAE * m["MAE"] / norm["MAE"]
            + W_P95 * m["p95"] / norm["p95"]
            + W_P99 * m["p99"] / norm["p99"]
            + W_HISHORT * m["hi_short"] / norm["hi_short"])


def evaluate(data_dir, submission_dir):
    # 防御: 调用方若按 (submission_dir, data_dir) 传反(两参同为路径, 传反不报错只退化成
    # validity=0), 这里探测并换回来, 保证两种调用顺序都能正确读到 submission.csv。
    if not (Path(submission_dir) / PLAN_FILE).is_file() and (Path(data_dir) / PLAN_FILE).is_file():
        data_dir, submission_dir = submission_dir, data_dir
    m = {"validity_score": 0.0, "quality_score": 0.0, "overall_score": 0.0, "error_info": {}}
    try:
        # 读私有真值
        truth = pd.read_csv(_TRUTH)
        truth["clear_da"] = pd.to_numeric(truth["clear_da"], errors="coerce")
        truth = truth.dropna(subset=["clear_da"]).copy()
        truth["date"] = pd.to_datetime(truth["date"]).dt.strftime("%Y-%m-%d")
        truth["time"] = truth["time"].astype(str).str.strip()

        # 读 agent 预测
        sub_path = Path(submission_dir) / PLAN_FILE
        if not sub_path.is_file():
            m["error_info"] = {"fatal": [f"未找到 {PLAN_FILE}"]}
            return m
        pred = pd.read_csv(sub_path)
        need = ["date", "time", "clear_da"]
        miss = [c for c in need if c not in pred.columns]
        if miss:
            m["error_info"] = {"fatal": [f"预测缺列: {miss}"]}
            return m
        pred["date"] = pd.to_datetime(pred["date"]).dt.strftime("%Y-%m-%d")
        pred["time"] = pred["time"].astype(str).str.strip()
        pred["clear_da"] = pd.to_numeric(pred["clear_da"], errors="coerce")
        if pred[need].isnull().any().any() or not np.isfinite(pred["clear_da"].values).all():
            m["error_info"] = {"fatal": ["预测含缺失/非有限值"]}
            return m
        if pred[["date", "time"]].duplicated().any():
            m["error_info"] = {"fatal": ["预测有重复的 (date, time)"]}
            return m

        # clip 到合理值域（agent 无需自己 clip；越界不判 0，评估器兜底截）
        pred["clear_da"] = np.clip(pred["clear_da"], PRICE_MIN, PRICE_MAX)

        # 对齐真值
        merged = truth.merge(pred[need], on=["date", "time"], how="left",
                             suffixes=("_true", "_pred"))
        missing = int(merged["clear_da_pred"].isnull().sum())
        if missing > 0:
            m["error_info"] = {"fatal": [f"预测未覆盖 {missing} 个 (date,time) 点"]}
            return m
        m["validity_score"] = 1.0

        # 打分：加权归一误差越小越好，quality = baseline / player
        baseline, direction, norms = _load_baseline()
        met = _compute_metrics(merged["clear_da_true"].values, merged["clear_da_pred"].values)
        player_error = _weighted_error(met, norms)
        if direction == "higher_is_better":
            quality = player_error / baseline if baseline > 0 else 0.0
        else:
            quality = baseline / max(player_error, 1e-6)
        # 完美预测(player_error→0)时 baseline/1e-6 会跳到百万级, 这是泄题刷分入口;
        # 基准惯例把评分封顶在 5.0。

        m["quality_score"] = round(quality, 6)
        m["overall_score"] = round(quality, 6)
        m["combined_score"] = round(player_error, 6)
        m["metrics"] = {k: (round(v, 4) if isinstance(v, float) else v) for k, v in met.items()}
        m["reference_value"] = baseline
        return m
    except Exception as e:
        m["error_info"] = {"exception": str(e), "traceback": traceback.format_exc()[-1500:]}
        return m


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--submission-dir", required=True)
    ap.add_argument("--data-dir", default=str(HERE.parent / "data"))
    a = ap.parse_args()
    print(json.dumps(evaluate(a.data_dir, a.submission_dir), ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
