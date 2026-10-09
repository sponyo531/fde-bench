"""
EXTRACTOR_SPEC:
  plan_file: prediction.csv
  required_columns: [sample_id, uniformity_y]
  notes: >
    Agent output must contain one row per test sample in data/test.csv (22 rows total).
    Each row must include:
      - sample_id: matching the sample_id in data/test.csv (integer string).
      - uniformity_y: predicted UniformityY curve as a JSON array string of floats
        (roughly 186 values covering the same x-range as train.csv's uniformity_y).

    If the agent used a different filename (submission.csv / predictions.csv / …)
    or column names (uniformityY / curve / …), map them to
    prediction.csv with the required columns above. Do NOT rescale or reshape the
    predicted curve — a poorly-shaped curve is a legitimate low score, not something
    to fix.
"""

import argparse
import csv
import json
import traceback
from pathlib import Path

import numpy as np

PLAN_FILE = "prediction.csv"

_HERE = Path(__file__).resolve().parent
_TRUTH = _HERE / "_private" / "test_truth.csv"


def _load_csv(path):
    with path.open("r", encoding="utf-8-sig") as f:
        return list(csv.DictReader(f))


def _parse_curve(s):
    try:
        raw = json.loads(s)
        if not isinstance(raw, list):
            return None
        curve = [float(x) for x in raw]
        if not curve or not np.isfinite(np.asarray(curve, dtype=float)).all():
            return None
        return curve
    except (json.JSONDecodeError, TypeError, ValueError):
        return None


def _align_length(pred, target_len):
    """Resample a predicted curve to match the target ground-truth length."""
    pred = list(pred)
    if len(pred) == target_len:
        return np.asarray(pred, dtype=float)
    x_orig = np.linspace(0.0, 1.0, len(pred))
    x_target = np.linspace(0.0, 1.0, target_len)
    return np.interp(x_target, x_orig, pred)


def _wape_improvement(pred_y, gt_y, first_y):
    """WAPE(pred, gt) 归一化：|pred-gt| / |gt-first|，全曲线评分（含窗口外，
    用于压制 flat-line 提交）。"""
    n = len(gt_y)
    pred = _align_length(pred_y, n)
    gt = np.asarray(gt_y, dtype=float)
    first = _align_length(first_y, n)
    denom = float(np.sum(np.abs(gt - first)))
    if denom < 1e-9:
        return None
    return float(np.sum(np.abs(pred - gt)) / denom)


def _load_baseline():
    with open(_HERE / "baseline" / "reference_metrics.json", encoding="utf-8") as f:
        d = json.load(f)
    return float(d["reference_value"]), d.get("direction", "higher_is_better")


def evaluate(data_dir, submission_dir):
    m = {"validity_score": 0.0, "quality_score": 0.0, "overall_score": 0.0, "error_info": {}}
    try:
        data_dir = Path(data_dir)
        submission_dir = Path(submission_dir)

        plan = submission_dir / PLAN_FILE
        if not plan.is_file():
            m["error_info"] = {"fatal": [f"未找到 {PLAN_FILE}"]}
            return m

        rows = _load_csv(plan)
        if not rows:
            m["error_info"] = {"fatal": [f"{PLAN_FILE} 为空"]}
            return m

        cols = set(rows[0].keys())
        for c in ("sample_id", "uniformity_y"):
            if c not in cols:
                m["error_info"] = {"fatal": [f"{PLAN_FILE} 缺列: {c}"]}
                return m

        # ── Test set (features) and private truth ─────────────────────────
        test_rows = _load_csv(data_dir / "test.csv")
        truth_rows = _load_csv(_TRUTH)

        first_curves = {}
        for r in test_rows:
            cur = _parse_curve(r.get("first_uniformity_y", ""))
            if cur is not None:
                first_curves[r["sample_id"]] = cur

        gt_curves = {}
        for r in truth_rows:
            cur = _parse_curve(r.get("gt_uniformity_y", ""))
            if cur is not None:
                gt_curves[r["sample_id"]] = cur

        # ── Parse submission ─────────────────────────────────────────────
        pred_map = {}
        parse_errs = []
        for row in rows:
            sid = str(row["sample_id"]).strip()
            uy = _parse_curve(row.get("uniformity_y", ""))
            if uy is None:
                parse_errs.append(f"sample_id={sid}: uniformity_y 不是可解析的 JSON 数组")
                continue
            if len(uy) < 10:
                parse_errs.append(f"sample_id={sid}: uniformity_y 长度过短 ({len(uy)} < 10)")
                continue
            if sid in pred_map:
                parse_errs.append(f"sample_id={sid} 重复")
                continue
            pred_map[sid] = uy

        test_ids = {r["sample_id"] for r in test_rows}
        missing = sorted(test_ids - set(pred_map.keys()))
        if missing:
            parse_errs.append(f"缺失 {len(missing)} 个测试样本 (前 5: {missing[:5]})")

        if parse_errs:
            m["error_info"] = {"fatal": parse_errs[:8]}
            return m

        m["validity_score"] = 1.0

        # ── Scoring: WAPE(pred, gt) / |gt - first| per sample, then avg ──
        per_sample_wape = {}
        for sid in sorted(test_ids, key=lambda x: int(x) if x.isdigit() else x):
            if sid not in gt_curves or sid not in first_curves:
                m["error_info"] = {"fatal": [f"sample_id={sid}: 真值/初始曲线缺失"]}
                m["validity_score"] = 0.0
                return m
            w = _wape_improvement(pred_map[sid], gt_curves[sid], first_curves[sid])
            if w is None:
                m["error_info"] = {"fatal": [f"sample_id={sid}: 真值改进量近零, 无法定义 WAPE"]}
                m["validity_score"] = 0.0
                return m
            per_sample_wape[sid] = w

        avg_wape = float(np.mean(list(per_sample_wape.values())))
        absolute_quality = 1.0 / (1.0 + avg_wape)

        baseline, direction = _load_baseline()
        if direction == "higher_is_better":
            quality = absolute_quality / baseline if baseline > 0 else 0.0
        else:
            quality = baseline / absolute_quality if absolute_quality > 0 else 0.0

        m["quality_score"] = round(quality, 6)
        m["overall_score"] = round(quality, 6)
        m["reference_value"] = baseline
        m["absolute_quality"] = round(absolute_quality, 6)
        m["avg_wape"] = round(avg_wape, 6)
        m["player_objective"] = avg_wape
        m["per_sample_wape"] = {sid: round(w, 6) for sid, w in per_sample_wape.items()}
        return m
    except Exception as e:
        m["validity_score"] = 0.0
        m["error_info"] = {"exception": str(e), "traceback": traceback.format_exc()[-800:]}
        return m


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data-dir", type=Path, default=_HERE.parent / "data")
    ap.add_argument("--submission-dir", type=Path, required=True)
    a = ap.parse_args()
    print(json.dumps(evaluate(str(a.data_dir), str(a.submission_dir)), indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
