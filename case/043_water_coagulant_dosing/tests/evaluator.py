"""
EXTRACTOR_SPEC:
  plan_file: predictions.csv
  required_columns: [window_id, step, dose_1, dose_2, dose_3, dose_4]
  notes: >
    水厂四条产线投矾量多步预测。选手产物是一个 CSV：对 data/forecast_windows.csv 里
    每个 window_id 的未来 36 个时刻各一行，给出四条产线的投矾量预测值。

    列名必须恰好是：window_id, step, dose_1, dose_2, dose_3, dose_4
    step 取 1~36（1 = 该窗口 hist_step=72 之后的第一个五分钟，36 = 第三小时末）。

    若选手把结果存成宽表（每个 window 一行、36×4 列）或长表（window_id, step,
    line, value），把它整理成上述长表结构：一行一个 (window_id, step)，四条线各一列。
    若用了别的列名（如 horizon / h / t 对应 step，pred_dose_1 / dose1 / 1 号线
    对应 dose_1），按语义映射。若选手另外交了 npy/json 形式的预测矩阵，按
    (window, 36, 4) 展开成上述 CSV。
    不要改动任何预测数值、不要重新训练模型、不要执行选手代码。
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
KEYS = ["window_id", "step"]
DOSE_COLS = ["dose_1", "dose_2", "dose_3", "dose_4"]
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
        "validity_score": 0.0,
        "quality_score": 0.0,
        "overall_score": 0.0,
        "error_info": {},
    }
    try:
        plan = os.path.join(submission_dir, PLAN_FILE)
        if not os.path.exists(plan):
            m["error_info"] = {"fatal": [f"缺 {PLAN_FILE}"]}
            return m

        try:
            pred = pd.read_csv(plan, encoding="utf-8-sig")
        except Exception as e:
            m["error_info"] = {"fatal": [f"{PLAN_FILE} 无法解析: {e}"]}
            return m

        for c in KEYS + DOSE_COLS:
            if c not in pred.columns:
                m["error_info"] = {"fatal": [f"{PLAN_FILE} 缺列 {c}, 实际: {list(pred.columns)}"]}
                return m

        truth = pd.read_csv(_TRUTH, encoding="utf-8-sig")
        pred = pred[KEYS + DOSE_COLS].copy()
        for k in KEYS:
            pred[k] = pd.to_numeric(pred[k], errors="coerce")
        if pred[KEYS].isna().any().any():
            m["error_info"] = {"invalid_submission": ["window_id / step 含非整数值"]}
            return m
        for k in KEYS:
            pred[k] = pred[k].astype(int)
            truth[k] = truth[k].astype(int)

        problems = []

        pk = list(map(tuple, pred[KEYS].values))
        tk = list(map(tuple, truth[KEYS].values))
        dup = pred[pred.duplicated(subset=KEYS, keep=False)][KEYS]
        if len(dup):
            items = [f"window {int(a)} step {int(b)}" for a, b in dup.drop_duplicates().values[:10]]
            problems.append(f"(window_id, step) 重复 {len(dup)} 行: {items}")

        miss = sorted(set(tk) - set(pk))
        if miss:
            problems.append(
                f"缺 {len(miss)} 个 (window_id, step): "
                + str([f"window {a} step {b}" for a, b in miss[:10]])
            )
        extra = sorted(set(pk) - set(tk))
        if extra:
            problems.append(
                f"多出 {len(extra)} 个不在待预测清单里的 (window_id, step): "
                + str([f"window {a} step {b}" for a, b in extra[:10]])
            )

        for c in DOSE_COLS:
            v = pd.to_numeric(pred[c], errors="coerce").replace([np.inf, -np.inf], np.nan)
            bad = pred.loc[v.isna(), KEYS]
            if len(bad):
                items = [f"window {int(a)} step {int(b)}" for a, b in bad.values[:10]]
                problems.append(f"{c} 非数值/缺失/inf {len(bad)} 行: {items}")
            neg = pred.loc[v.notna() & (v < 0), KEYS]
            if len(neg):
                items = [f"window {int(a)} step {int(b)}" for a, b in neg.values[:10]]
                problems.append(f"{c} 出现负投矾量 {len(neg)} 行: {items}")
            pred[c] = v

        if problems:
            m["error_info"] = {"invalid_submission": problems[:8]}
            return m

        m["validity_score"] = 1.0

        df = truth.merge(pred, on=KEYS, how="left", suffixes=("_true", "_pred"))
        per_line = {}
        maes = []
        for c in DOSE_COLS:
            yt = df[f"{c}_true"].values.astype(float)
            yp = df[f"{c}_pred"].values.astype(float)
            mask = yt > 0                      # 停线时段(实际投加量为0)不计考核
            if mask.sum() == 0:
                continue
            mae = float(np.mean(np.abs(yt[mask] - yp[mask])))
            per_line[c] = {"mae": round(mae, 6), "n_scored": int(mask.sum())}
            maes.append(mae)

        combined = float(np.mean(maes))
        base = load_baseline()
        m["quality_score"] = round((base / combined) if combined > 0 else PERFECT_Q, 6)
        m["overall_score"] = m["quality_score"]
        m["player_objective"] = round(combined, 6)
        m["reference_value"] = base
        m["per_line"] = per_line
        m["n_windows"] = int(truth["window_id"].nunique())
        return m

    except Exception as e:
        m["validity_score"] = 0.0
        m["quality_score"] = 0.0
        m["overall_score"] = 0.0
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
