"""
EXTRACTOR_SPEC:
  plan_file: predictions.csv
  required_columns: ["业务/服务名称", "选址评分"]
  notes: >
    银行网点选址评分预测。选手产物是一个 CSV：对 data/test_branches.csv 里每个
    网点各一行，给出 0-100 的选址评分。

    列名必须恰好是：业务/服务名称, 选址评分

    若选手用了别的列名（branch/name/网点名称 代替 业务/服务名称，
    pred/score/predicted/y_pred/预测得分 代替 选址评分），映射到上面两列。
    网点名称保持原样，不要去空格以外的任何改写。
    不要改动任何预测数值、不要做截断或缩放、不要重新训练模型。
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
ID_COL = "业务/服务名称"
VAL_COL = "选址评分"
SCORE_LO, SCORE_HI = 0.0, 100.0
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
            m["error_info"] = {"fatal": [f"{VAL_COL} 含非数值/缺失，共 {int(pred[VAL_COL].isnull().sum())} 行"]}
            return m
        v = pred[VAL_COL].values.astype(float)
        if not np.isfinite(v).all():
            m["error_info"] = {"fatal": ["预测值含 inf/nan"]}
            return m
        out = int(((v < SCORE_LO - 1e-9) | (v > SCORE_HI + 1e-9)).sum())
        if out:
            m["error_info"] = {"fatal": [
                f"有 {out} 个预测值不在 {SCORE_LO:.0f}-{SCORE_HI:.0f} 分区间内"
                f"（最小 {v.min():.4f}，最大 {v.max():.4f}）"
            ]}
            return m

        pred[ID_COL] = pred[ID_COL].astype(str).str.strip()
        if pred[ID_COL].duplicated().any():
            n = int(pred[ID_COL].duplicated().sum())
            m["error_info"] = {"fatal": [f"{ID_COL} 有 {n} 行重复"]}
            return m

        truth = pd.read_csv(_TRUTH, encoding="utf-8-sig")
        truth[ID_COL] = truth[ID_COL].astype(str).str.strip()
        need = set(truth[ID_COL])
        got = set(pred[ID_COL])
        miss = need - got
        if miss:
            m["error_info"] = {"fatal": [
                f"缺 {len(miss)} 个待评网点的预测，例如: {', '.join(sorted(miss)[:3])}"
            ]}
            return m

        m["validity_score"] = 1.0

        df = truth.merge(pred[[ID_COL, VAL_COL]], on=ID_COL, how="left",
                         suffixes=("_true", "_pred"))
        yt = df[f"{VAL_COL}_true"].values.astype(float)
        yp = df[f"{VAL_COL}_pred"].values.astype(float)
        mae = float(np.mean(np.abs(yt - yp)))
        rmse = float(np.sqrt(np.mean((yt - yp) ** 2)))
        ss = float(np.sum((yt - yt.mean()) ** 2))
        r2 = 1.0 - float(np.sum((yt - yp) ** 2)) / ss if ss > 0 else 0.0

        base = load_baseline()
        m["quality_score"] = round((base / mae) if mae > 0 else PERFECT_Q, 6)
        m["overall_score"] = m["quality_score"]
        m["player_objective"] = round(mae, 6)
        m["reference_value"] = base
        m["rmse"] = round(rmse, 6)
        m["r2"] = round(r2, 6)
        m["n_scored"] = int(len(df))
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
