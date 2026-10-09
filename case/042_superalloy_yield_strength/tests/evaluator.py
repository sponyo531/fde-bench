"""
EXTRACTOR_SPEC:
  plan_file: predictions.csv
  required_columns: [sample_id, predicted_yield_strength]
  notes: >
    agent 的产物是对 data/to_predict.csv 里每个新配方的室温屈服强度预测。把它规范化成
    predictions.csv，两列：
    - sample_id: 与 data/to_predict.csv 的 sample_id 完全一致的编号字符串（如 TEST_0001）。
    - predicted_yield_strength: 该配方的预测强度，单位 MPa，单个数值。
    规范化要求：
    - 若 agent 的预测放在别的文件名（如 result.json / submission.csv / pred.txt）或列名不同
      （如 pred / yield_pred / 预测强度），改名搬过来即可，数值一个都不要改、不要四舍五入、
      不要重排或补齐 agent 没给出的编号。
    - 若 agent 同时给了多组模型的预测，只取它自己标为最终/最优的那一组。
    - 若 agent 给的是区间或多列（如上下界），取其点估计列；确实只有区间时留空该行，
      不要自行取中点。
    - 不要执行 agent 的训练代码，不要自行建模补预测。
"""
import argparse
import csv
import json
import math
import os
import traceback

_HERE = os.path.dirname(os.path.abspath(__file__))
PLAN_FILE = "predictions.csv"
PRIVATE_TRUTH = os.path.join(_HERE, "_private", "test_truth.csv")
LABEL_COL = "Yield sterngth"
ID_COL = "sample_id"
PRED_COL = "predicted_yield_strength"


def load_baseline():
    with open(os.path.join(_HERE, "baseline", "reference_metrics.json"), encoding="utf-8") as f:
        d = json.load(f)
    return float(d["reference_value"]), d.get("direction", "lower_is_better")


def _read_csv(path):
    with open(path, encoding="utf-8-sig", newline="") as f:
        return list(csv.DictReader(f))


def load_data(data_dir):
    """从 data/ 读待预测清单的编号(评估器独立确定应交哪些行), 真值只从 tests/_private/ 读。"""
    ids = [str(r[ID_COL]).strip() for r in _read_csv(os.path.join(data_dir, "to_predict.csv"))]
    truth = {}
    for r in _read_csv(PRIVATE_TRUTH):
        truth[str(r[ID_COL]).strip()] = float(r[LABEL_COL])
    return ids, truth


def evaluate(submission_dir, data_dir):
    m = {"validity_score": 0.0, "quality_score": 0.0, "overall_score": 0.0, "error_info": {}}
    try:
        path = os.path.join(submission_dir, PLAN_FILE)
        if not os.path.exists(path):
            m["error_info"] = {"fatal": ["缺 %s" % PLAN_FILE]}
            return m

        rows = _read_csv(path)
        ids, truth = load_data(data_dir)
        baseline, direction = load_baseline()

        if not rows:
            m["error_info"] = {"fatal": ["predictions.csv 没有数据行"]}
            return m
        header = list(rows[0].keys())
        missing_cols = [c for c in (ID_COL, PRED_COL) if c not in header]
        if missing_cols:
            m["error_info"] = {"fatal": ["predictions.csv 缺列 %s(实际列: %s)" % (missing_cols, header)]}
            return m

        # ── 产物有效性: 编号全覆盖唯一 + 预测值为有限数值 ──
        fatal = []
        pred = {}
        for idx, r in enumerate(rows):
            sid = str(r.get(ID_COL, "")).strip()
            raw = r.get(PRED_COL)
            raw = "" if raw is None else str(raw).strip()
            if sid == "":
                fatal.append("第 %d 行 sample_id 为空" % (idx + 2))
                continue
            if sid in pred:
                fatal.append("编号 %s 重复出现" % sid)
                continue
            if raw == "":
                fatal.append("编号 %s 的预测值为空" % sid)
                continue
            try:
                v = float(raw)
            except ValueError:
                fatal.append("编号 %s 的预测值无法解析为数值: %r" % (sid, raw))
                continue
            if math.isnan(v) or math.isinf(v):
                fatal.append("编号 %s 的预测值不是有限数值: %r" % (sid, raw))
                continue
            pred[sid] = v
        missing = [i for i in ids if i not in pred]
        extra = [i for i in pred if i not in set(ids)]
        if missing:
            fatal.append("有 %d 个待预测编号未给出预测: %s" % (len(missing), missing[:10]))
        if extra:
            fatal.append("有 %d 个编号不在待预测清单中: %s" % (len(extra), sorted(extra)[:10]))
        if fatal:
            m["error_info"] = {"fatal": fatal[:8]}
            return m
        m["validity_score"] = 1.0

        # ── 目标值: 用私有真值独立重算整批预测偏差(MPa) ──
        abs_err = [abs(pred[i] - truth[i]) for i in ids]
        player_obj = sum(abs_err) / len(abs_err)

        eps = 1e-9
        if direction == "lower_is_better":
            quality = baseline / max(player_obj, eps)
        else:
            quality = player_obj / baseline if baseline > 0 else 0.0
        m["quality_score"] = round(quality, 6)
        m["overall_score"] = m["quality_score"]
        m["player_objective"] = round(player_obj, 6)
        m["reference_value"] = baseline
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
