"""
EXTRACTOR_SPEC:
  plan_file: predictions.csv
  required_columns: [timestamp, direction, count]
  notes: >
    电梯外呼流量预测。选手的产物是一份 CSV `predictions.csv`：

        timestamp,direction,count
        2025-06-27 00:00:00,上行,0.4
        2025-06-27 00:00:00,下行,1.2

    - timestamp 为 10 分钟桶起点，direction 取 上行 / 下行
    - count 为非负有限数值（可为小数）
    - 须覆盖 data/forecast_template.csv 的每一行（桶 × 方向），恰好各一行

    Extractor: 从选手工作区找到最终预测结果（可能叫 predictions.csv / submission.csv /
    forecast.csv，或宽表形式"每行一个桶、上行下行各一列"）。若是宽表，融化成上面的长表；
    若 direction 用了 up/down 或 0/1，映射回 上行/下行；timestamp 统一成
    `YYYY-MM-DD HH:MM:SS`。不执行选手代码重新推理，不改动任何预测数值。

    评估器持有私有真值（tests/_private/ground_truth.csv，只含历史上有呼叫的桶），
    对清单中真值缺失的桶按 0 补齐后独立计算整体 MAE 与双向 MAE 偏斜，
    绝不采信选手自报的任何指标。
"""

from __future__ import annotations

import argparse
import json
import math
import os
import traceback

import pandas as pd

_HERE = os.path.dirname(os.path.abspath(__file__))
PLAN_FILE = "predictions.csv"
PRIVATE_TRUTH = os.path.join(_HERE, "_private", "ground_truth.csv")

DIRECTIONS = {"上行", "下行"}
# 综合得分：以整体 MAE 为主，双向 MAE 的偏斜作为惩罚项
SKEW_PENALTY_W = 0.30


def load_baseline():
    with open(os.path.join(_HERE, "baseline", "reference_metrics.json"), encoding="utf-8") as f:
        d = json.load(f)
    return float(d["reference_value"]), d.get("direction", "higher_is_better")


def load_data(data_dir):
    tmpl = pd.read_csv(os.path.join(data_dir, "forecast_template.csv"))
    tmpl["timestamp"] = pd.to_datetime(tmpl["timestamp"], errors="coerce")
    tmpl["direction"] = tmpl["direction"].astype(str).str.strip()
    return tmpl


def evaluate(submission_dir, data_dir):
    m = {"validity_score": 0.0, "quality_score": 0.0, "overall_score": 0.0, "error_info": {}}
    try:
        path = os.path.join(submission_dir, PLAN_FILE)
        if not os.path.exists(path):
            m["error_info"] = {"fatal": [f"缺 {PLAN_FILE}"]}
            return m
        if not os.path.exists(PRIVATE_TRUTH):
            m["error_info"] = {"fatal": ["评估器缺私有真值文件"]}
            return m

        tmpl = load_data(data_dir)
        truth = pd.read_csv(PRIVATE_TRUTH)
        truth["timestamp"] = pd.to_datetime(truth["timestamp"], errors="coerce")
        truth["direction"] = truth["direction"].astype(str).str.strip()
        baseline, direction_flag = load_baseline()

        sub = pd.read_csv(path)
        need = {"timestamp", "direction", "count"}
        if not need.issubset(set(sub.columns)):
            m["error_info"] = {"schema": [f"缺列，需 {sorted(need)}，实得 {list(sub.columns)}"]}
            return m
        sub["timestamp"] = pd.to_datetime(sub["timestamp"], errors="coerce")
        sub["direction"] = sub["direction"].astype(str).str.strip()

        errs = []
        n_bad_ts = int(sub["timestamp"].isna().sum())
        if n_bad_ts:
            errs.append(f"{n_bad_ts} 行 timestamp 无法解析")
        bad_dir = sorted(set(sub["direction"]) - DIRECTIONS)
        if bad_dir:
            errs.append(f"direction 含非法取值: {bad_dir[:4]}")
        vals = pd.to_numeric(sub["count"], errors="coerce")
        if int(vals.isna().sum()):
            errs.append(f"{int(vals.isna().sum())} 行 count 无法解析为数值或为空")
        finite_bad = int((~vals.dropna().apply(math.isfinite)).sum())
        if finite_bad:
            errs.append(f"{finite_bad} 行 count 非有限")
        if int((vals.dropna() < 0).sum()):
            errs.append(f"{int((vals.dropna() < 0).sum())} 行 count 为负")
        if errs:
            m["error_info"] = {"schema": errs[:6]}
            return m

        key = ["timestamp", "direction"]
        dup = int(sub.duplicated(subset=key).sum())
        if dup:
            m["error_info"] = {"schema": [f"(timestamp, direction) 重复 {dup} 行"]}
            return m
        want = set(map(tuple, tmpl[key].values))
        got = set(map(tuple, sub[key].values))
        miss, extra = want - got, got - want
        if miss or extra:
            e = []
            if miss:
                e.append(f"缺少 {len(miss)} 个待预测组合，例如 {sorted(map(str, list(miss)[:3]))}")
            if extra:
                e.append(f"含 {len(extra)} 个清单外组合，例如 {sorted(map(str, list(extra)[:3]))}")
            m["error_info"] = {"schema": e}
            return m

        m["validity_score"] = 1.0

        # ── 真值对齐：清单里真值缺失的桶按 0 补齐 ──
        sub = sub.assign(pred=vals)
        merged = tmpl.merge(truth, on=key, how="left").merge(
            sub[key + ["pred"]], on=key, how="left")
        merged["count"] = merged["count"].fillna(0.0)
        merged["pred"] = merged["pred"].fillna(0.0)

        merged["ae"] = (merged["pred"] - merged["count"]).abs()
        mae = float(merged["ae"].mean())
        per_dir = merged.groupby("direction")["ae"].mean().to_dict()
        up, down = per_dir.get("上行", 0.0), per_dir.get("下行", 0.0)
        skew = abs(up - down)

        # 综合得分：MAE 越小越高，双向偏斜作为惩罚
        score = 100.0 / (1.0 + mae) - SKEW_PENALTY_W * skew
        score = max(0.0, score)

        if direction_flag == "lower_is_better":
            quality = baseline / score if score > 0 else 0.0
        else:
            quality = score / baseline if baseline > 0 else 0.0

        m["quality_score"] = round(quality, 6)
        m["overall_score"] = m["quality_score"]
        m["player_objective"] = round(score, 6)
        m["reference_value"] = round(baseline, 6)
        m["metric"] = {
            "mae_overall": round(mae, 4),
            "mae_up": round(float(up), 4),
            "mae_down": round(float(down), 4),
            "direction_skew": round(float(skew), 4),
            "n_scored": int(len(merged)),
            "n_zero_truth_buckets": int((merged["count"] == 0).sum()),
        }
    except Exception as e:
        m["error_info"] = {"exception": str(e), "traceback": traceback.format_exc()}
    return m


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--submission-dir", required=True)
    ap.add_argument("--data-dir", default=os.path.join(_HERE, "..", "data"))
    a = ap.parse_args()
    print(json.dumps(evaluate(a.submission_dir, a.data_dir), ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
