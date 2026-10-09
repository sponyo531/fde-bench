"""
EXTRACTOR_SPEC:
  plan_file: predictions.csv
  required_columns: [customer_id, prob_default]
  notes: >
    个人信贷违约评分卡。选手产物是一个 CSV：对 data/test_features.csv 里每个
    customer_id 各一行，给出违约概率。

    列名必须恰好是：customer_id, prob_default

    若选手输出的概率列名不同（如 pred / score / probability / p1 / default_prob），
    映射到 prob_default。若选手输出的是 0/1 硬分类而非概率，直接照搬该列值
    （不要自行转换成概率）。若输出了两列概率（好/坏各一列），取违约(=1)那一列。
    不要改动任何数值、不要重新训练模型。
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
_HERE = Path(__file__).resolve().parent
_TRUTH = _HERE / "_private" / "test_truth.csv"


def _auc(y: np.ndarray, s: np.ndarray) -> float:
    order = np.argsort(s, kind="mergesort")
    s_sorted = s[order]
    ranks = np.empty(len(s), dtype=float)
    i = 0
    while i < len(s_sorted):
        j = i
        while j + 1 < len(s_sorted) and s_sorted[j + 1] == s_sorted[i]:
            j += 1
        ranks[order[i:j + 1]] = (i + j) / 2.0 + 1.0
        i = j + 1
    n_pos = float((y == 1).sum())
    n_neg = float((y == 0).sum())
    if n_pos == 0 or n_neg == 0:
        return 0.0
    return float((ranks[y == 1].sum() - n_pos * (n_pos + 1) / 2.0) / (n_pos * n_neg))


def _ks(y: np.ndarray, s: np.ndarray) -> float:
    order = np.argsort(-s, kind="mergesort")
    y_s = y[order]
    s_s = s[order]
    n_pos = float((y == 1).sum())
    n_neg = float((y == 0).sum())
    if n_pos == 0 or n_neg == 0:
        return 0.0
    tpr = np.cumsum(y_s == 1) / n_pos
    fpr = np.cumsum(y_s == 0) / n_neg
    # 同分客户不能靠输入行顺序被人为拆成多个阈值；只在每个同分块末尾
    # 观察一次累计差异，将整块视为同一个可执行分割位置。
    block_end = np.r_[s_s[1:] != s_s[:-1], True]
    return float(np.max(np.abs(tpr[block_end] - fpr[block_end])))


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
        for c in ["customer_id", "prob_default"]:
            if c not in pred.columns:
                m["error_info"] = {"fatal": [f"{PLAN_FILE} 缺列 {c}"]}
                return m

        pred["prob_default"] = pd.to_numeric(pred["prob_default"], errors="coerce")
        if pred["prob_default"].isnull().any():
            m["error_info"] = {"fatal": ["prob_default 含非数值/缺失"]}
            return m
        v = pred["prob_default"].values.astype(float)
        if not np.isfinite(v).all():
            m["error_info"] = {"fatal": ["prob_default 含 inf/nan"]}
            return m
        if (v < 0).any() or (v > 1).any():
            m["error_info"] = {"fatal": ["prob_default 越界 [0,1]"]}
            return m
        if pred["customer_id"].duplicated().any():
            m["error_info"] = {"fatal": ["customer_id 重复"]}
            return m

        truth = pd.read_csv(_TRUTH, encoding="utf-8-sig")
        need = set(truth["customer_id"].astype(str))
        have = set(pred["customer_id"].astype(str))
        if need - have:
            m["error_info"] = {"fatal": [f"缺 {len(need-have)} 个客户"]}
            return m

        m["validity_score"] = 1.0

        truth["customer_id"] = truth["customer_id"].astype(str)
        pred["customer_id"] = pred["customer_id"].astype(str)
        df = truth.merge(pred[["customer_id", "prob_default"]], on="customer_id", how="left")
        y = df["default"].values.astype(int)
        s = df["prob_default"].values.astype(float)

        auc = _auc(y, s)
        ks = _ks(y, s)
        combined = 0.5 * auc + 0.5 * ks

        base = load_baseline()
        quality = combined / base if base > 0 else 0.0

        m["quality_score"] = round(quality, 6)
        m["overall_score"] = m["quality_score"]
        m["player_objective"] = round(combined, 6)
        m["reference_value"] = base
        m["auc"] = round(auc, 6)
        m["ks"] = round(ks, 6)
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
