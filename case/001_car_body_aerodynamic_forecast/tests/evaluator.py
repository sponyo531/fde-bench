"""
EXTRACTOR_SPEC:
  plan_file: predictions.csv
  required_columns: [vtu_name, Cd_pred]
  notes: >
    Agent 交一个 CSV `predictions.csv`, 两列:
      vtu_name  : 测试车文件名 (与 data/frontal_areas.json 里"缺 Cd 字段"的键一字不差,
                  含 @05000.vtu 后缀)
      Cd_pred   : 该车预测的风阻系数 Cd (float)

    数据切分隐含在 data/frontal_areas.json: 含 Cd 字段的 150 车为训练集
    (Cd 即真值标签), 缺 Cd 字段的 37 车为测试集 (真值由评估器私有持有)。

    Agent 可用任何方法预测 Cd: 传统流场求解 / GNN 回归 / 数据统计 / 几何特征 等,
    评估器只对比 Cd_pred 与私有真值 Cd_true 的相对误差, 不检查过程。

    Extractor: 从选手工作区找 predictions.csv, 若命名不同(例如 output.csv/sub.csv),
    找列名匹配 [vtu_name, Cd_pred] 的 CSV, 拷贝重命名为 predictions.csv 到输出目录。
    不改数值。
"""
import os, json, argparse, traceback
import numpy as np
import pandas as pd

_HERE = os.path.dirname(os.path.abspath(__file__))
PLAN_FILE = "predictions.csv"
PRIVATE_FRONTAL = os.path.join(_HERE, "_private", "frontal_areas.json")
REQUIRED_COLS = ["vtu_name", "Cd_pred"]
CD_MIN, CD_MAX = 0.05, 1.5  # 物理合理范围: 商用车典型 Cd 0.2~0.5, 极端保守取此范围


def _load_test_truth(data_dir):
    """测试集由公开 frontal_areas.json 里"缺 Cd 字段"的键定义, 真值从私有 frontal_areas.json 取。"""
    pub_path = os.path.join(data_dir, "frontal_areas.json")
    if not os.path.exists(pub_path):
        raise FileNotFoundError(f"缺 {pub_path}")
    with open(pub_path, "r", encoding="utf-8") as f:
        pub = json.load(f)
    test_files = [name for name, meta in pub.items() if "Cd" not in meta]
    with open(PRIVATE_FRONTAL, "r", encoding="utf-8") as f:
        priv = json.load(f)
    truth = {}
    for name in test_files:
        if name not in priv or "Cd" not in priv[name]:
            raise KeyError(f"私有真值缺 test 车: {name}")
        truth[name] = float(priv[name]["Cd"])
    return truth  # {vtu_name: Cd_true}


def _load_baseline():
    p = os.path.join(_HERE, "baseline", "reference_metrics.json")
    with open(p, "r", encoding="utf-8") as f:
        d = json.load(f)
    return float(d["reference_value"]), d.get("direction", "higher_is_better")


def evaluate(submission_dir, data_dir):
    m = {"validity_score": 0.0, "quality_score": 0.0, "overall_score": 0.0, "error_info": {}}
    try:
        # -- 1) 载入 predictions.csv --
        plan = os.path.join(submission_dir, PLAN_FILE)
        if not os.path.exists(plan):
            m["error_info"] = {"fatal": [f"缺 {PLAN_FILE}"]}
            return m
        try:
            df = pd.read_csv(plan)
        except Exception as e:
            m["error_info"] = {"fatal": [f"读取 {PLAN_FILE} 失败: {e}"]}
            return m

        miss = [c for c in REQUIRED_COLS if c not in df.columns]
        if miss:
            m["error_info"] = {"fatal": [f"{PLAN_FILE} 缺列: {miss}"]}
            return m

        # -- 2) 载入真值 --
        truth = _load_test_truth(data_dir)
        n_test = len(truth)

        # -- 3) 硬约束校验 --
        errs = []
        # HC1: 覆盖测试集所有车
        subm_names = set(df["vtu_name"].astype(str))
        miss_cars = [n for n in truth if n not in subm_names]
        if miss_cars:
            errs.append(f"HC-覆盖: 缺 {len(miss_cars)} 辆测试车, 例: {miss_cars[:3]}")
        # HC2: 无重复 vtu_name
        dup = df[df.duplicated("vtu_name", keep=False)]
        if not dup.empty:
            errs.append(f"HC-去重: {len(dup)} 行重复 vtu_name, 例: {dup['vtu_name'].iloc[:3].tolist()}")
        # HC3: Cd_pred 数值合法 (finite + 物理范围)
        cd_col = pd.to_numeric(df["Cd_pred"], errors="coerce")
        n_nan = int(cd_col.isna().sum())
        n_inf = int(np.isinf(cd_col.values).sum())
        if n_nan or n_inf:
            errs.append(f"HC-有限: Cd_pred 含 NaN×{n_nan} / Inf×{n_inf}")
        else:
            oob = df[(cd_col < CD_MIN) | (cd_col > CD_MAX)]
            if not oob.empty:
                errs.append(f"HC-物理范围: {len(oob)} 行 Cd_pred 越界 [{CD_MIN}, {CD_MAX}], 例: {oob['Cd_pred'].iloc[:3].tolist()}")

        if errs:
            m["error_info"] = {"constraint": errs[:8]}
            return m
        m["validity_score"] = 1.0

        # -- 4) 评分: Cd_MRE (mean relative error) --
        df_test = df[df["vtu_name"].isin(truth.keys())].copy()
        df_test["Cd_true"] = df_test["vtu_name"].map(truth)
        df_test["rel_err"] = (df_test["Cd_pred"] - df_test["Cd_true"]).abs() / df_test["Cd_true"].abs().clip(lower=1e-12)

        cd_mre = float(df_test["rel_err"].mean())

        # -- 5) quality: Cd_MRE 越小越好, 故 quality = baseline_MRE / player_MRE (不截断) --
        baseline_mre, direction = _load_baseline()
        # 分母下限取真值的分辨率地板：公布的 Cd 只到 0.001，任何预测与真值的差都有
        # ±0.0005 的取整不确定度，MRE 低于 mean(0.0005/|Cd_true|) 没有统计意义。
        # 原来用 1e-6 作下限，quality 上限高达 baseline/1e-6（曾出现 1676 分），改为此地板。
        mre_floor = float(np.mean(0.0005 / df_test["Cd_true"].abs().clip(lower=1e-12)))
        if direction == "higher_is_better":
            quality = max(cd_mre, mre_floor) / baseline_mre if baseline_mre > 0 else 0.0
        else:
            quality = baseline_mre / max(cd_mre, mre_floor)

        m["quality_score"] = round(float(quality), 6)
        m["overall_score"] = m["quality_score"]
        m["metric"] = {
            "n_test": n_test,
            "n_scored": int(len(df_test)),
            "Cd_MRE": round(cd_mre, 6),
            "Cd_MRE_floor": round(mre_floor, 6),
        }
        m["reference_value"] = baseline_mre
    except Exception:
        m["error_info"] = {"exception": traceback.format_exc()[:2000]}
    return m


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--submission-dir", required=True)
    ap.add_argument("--data-dir", default=os.path.join(_HERE, "..", "data"))
    a = ap.parse_args()
    print(json.dumps(evaluate(a.submission_dir, a.data_dir), ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
