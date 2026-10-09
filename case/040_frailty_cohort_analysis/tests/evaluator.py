"""
EXTRACTOR_SPEC:
  plan_file: trajectory_groups.csv
  required_columns: [hhidpn, group]
  notes: >
    疼痛轨迹分组。选手产物是一个 CSV：对 data/hrs_cohort.csv 里每个 hhidpn 各一行，
    给出该个体所属的轨迹组标签。

    列名必须恰好是：hhidpn, group

    若选手用了别的列名（id / ID / person_id 代替 hhidpn；class / cluster / traj /
    label / trajectory_group / 轨迹组 代替 group），映射到上面两列。
    若选手把分组结果写在宽表里（每组一列 0/1，或含后验概率矩阵），取其硬分组结果
    （概率最大的那一组）转成长表两列。组标签原样保留、不要重新编号或重新排序，
    也不要自行重新聚类。
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

PLAN_FILE = "trajectory_groups.csv"
ID_COL = "hhidpn"
GRP_COL = "group"
PAIN_COLS = ["p2006", "p2008", "p2010", "p2012"]

MIN_K, MAX_K = 2, 5
MIN_GROUP_N = 30
MIN_GROUP_PCT = 0.02
MIN_CENTROID_CONSISTENCY = 0.80
W_FIT = 0.5            # 轨迹概括程度权重
W_RISK = 2.0           # 风险区分度权重

_HERE = Path(__file__).resolve().parent
_TRUTH = _HERE / "_private" / "test_outcomes.csv"


def load_baseline():
    with open(_HERE / "baseline" / "reference_metrics.json", encoding="utf-8") as f:
        d = json.load(f)
    return float(d["reference_value"]), d.get("direction", "higher_is_better")


def _c_index(time: np.ndarray, event: np.ndarray, risk: np.ndarray) -> float:
    """Harrell's C-index; 风险分相同的可比对计 0.5。"""
    order = np.argsort(time)
    t, e, r = time[order], event[order], risk[order]
    num = den = 0.0
    for i in range(len(t)):
        if e[i] != 1:
            continue
        later = t > t[i]
        n = int(later.sum())
        if n == 0:
            continue
        den += n
        num += float((r[later] < r[i]).sum()) + 0.5 * float((r[later] == r[i]).sum())
    return num / den if den > 0 else 0.5


# 2026-08-25 修:去掉 quality 的 5.0 封顶 —— 好解一旦超过基准 5 倍就被压平,跟刷分的
#   混在一起分不出高低。前两轮批量去封顶漏了 min(max(q,0.0),5.0) 与 min(float(_q),5.0)
#   这两种写法(变量名不叫 quality),第三轮改用「看 m["quality_score"] 赋值右边」才扫净。
def evaluate(submission_dir: str, data_dir: str) -> Dict[str, Any]:
    m: Dict[str, Any] = {
        "validity_score": 0.0, "quality_score": 0.0, "overall_score": 0.0, "error_info": {},
    }
    try:
        plan = os.path.join(submission_dir, PLAN_FILE)
        if not os.path.exists(plan):
            m["error_info"] = {"fatal": [f"缺 {PLAN_FILE}"]}
            return m
        sub = pd.read_csv(plan, encoding="utf-8-sig")
        for c in (ID_COL, GRP_COL):
            if c not in sub.columns:
                m["error_info"] = {"fatal": [f"{PLAN_FILE} 缺列 {c}, 实际: {list(sub.columns)}"]}
                return m

        coh = pd.read_csv(os.path.join(data_dir, "hrs_cohort.csv"), encoding="utf-8-sig")
        tr = pd.read_csv(os.path.join(data_dir, "followup_outcomes_train.csv"),
                         encoding="utf-8-sig")
        te = pd.read_csv(_TRUTH, encoding="utf-8-sig")

        group_missing = sub[GRP_COL].isna()
        sub[ID_COL] = sub[ID_COL].astype(str).str.strip()
        coh[ID_COL] = coh[ID_COL].astype(str).str.strip()
        tr[ID_COL] = tr[ID_COL].astype(str).str.strip()
        te[ID_COL] = te[ID_COL].astype(str).str.strip()
        sub[GRP_COL] = sub[GRP_COL].astype(str).str.strip()

        if sub[ID_COL].duplicated().any():
            m["error_info"] = {"fatal": [f"{ID_COL} 重复 {int(sub[ID_COL].duplicated().sum())} 条"]}
            return m
        if group_missing.any() or (sub[GRP_COL] == "").any():
            m["error_info"] = {"fatal": ["group 列存在空值"]}
            return m
        need = set(coh[ID_COL])
        missing = need - set(sub[ID_COL])
        if missing:
            m["error_info"] = {"fatal": [f"名单缺 {len(missing)} 个个体, 例: {sorted(missing)[:5]}"]}
            return m

        gmap = dict(zip(sub[ID_COL], sub[GRP_COL]))
        g = coh[ID_COL].map(gmap).values.astype(object)
        P = coh[PAIN_COLS].values.astype(float)
        n = len(coh)

        # ── 硬约束 ──
        errs = []
        labels = sorted(set(g.tolist()))
        K = len(labels)
        if K < MIN_K or K > MAX_K:
            errs.append(f"轨迹组数 {K} 不在 [{MIN_K}, {MAX_K}] 内")
        sizes = {k: int((g == k).sum()) for k in labels}
        for k, sz in sizes.items():
            if sz < MIN_GROUP_N:
                errs.append(f"组 {k} 样本量 {sz} < {MIN_GROUP_N}")
            if sz / n < MIN_GROUP_PCT:
                errs.append(f"组 {k} 占比 {sz / n:.4f} < {MIN_GROUP_PCT}")
        if errs:
            m["error_info"] = {"hard_violations": errs[:15], "total": len(errs)}
            return m

        cent = {k: np.nanmean(P[g == k], axis=0) for k in labels}
        for k in labels:
            if np.isnan(cent[k]).any():
                errs.append(f"组 {k} 在某个波次上无任何可观测疼痛得分, 无法构成轨迹")
        if errs:
            m["error_info"] = {"hard_violations": errs[:15], "total": len(errs)}
            return m

        dist = np.stack([np.nanmean((P - cent[k]) ** 2, axis=1) for k in labels], axis=1)
        nearest = np.array(labels, dtype=object)[np.argmin(dist, axis=1)]
        consistency = float(np.mean(nearest == g))
        if consistency < MIN_CENTROID_CONSISTENCY:
            errs.append(f"仅 {consistency:.4f} 的个体归入疼痛均值最接近的组, "
                        f"低于 {MIN_CENTROID_CONSISTENCY}, 分组未按疼痛轨迹划分")
        if errs:
            m["error_info"] = {"hard_violations": errs[:15], "total": len(errs)}
            return m

        m["validity_score"] = 1.0

        # ── 质量: 轨迹概括程度(R²) + 对衰弱风险的区分度(留出集 C-index) ──
        grand = np.nanmean(P, axis=0)
        sst = float(np.nansum((P - grand) ** 2))
        ssw = float(sum(np.nansum((P[g == k] - cent[k]) ** 2) for k in labels))
        r2 = 1.0 - ssw / sst if sst > 0 else 0.0

        tr_g = tr[ID_COL].map(gmap).values.astype(object)
        inc = {}
        for k in labels:
            sel = tr_g == k
            inc[k] = float(tr["event"].values[sel].mean()) if sel.sum() > 0 else 0.0
        te_g = te[ID_COL].map(gmap).values.astype(object)
        risk = np.array([inc.get(x, 0.0) for x in te_g], dtype=float)
        c = _c_index(te["time"].values.astype(float),
                     te["event"].values.astype(int), risk)

        combined = W_FIT * r2 + W_RISK * (c - 0.5)
        base, direction = load_baseline()
        if direction == "lower_is_better":
            q = base / combined if combined > 0 else 0.0
        else:
            q = combined / base if base > 0 else 0.0
        m["quality_score"] = round(max(q, 0.0), 6)
        m["overall_score"] = m["quality_score"]
        m["player_objective"] = round(combined, 6)
        m["reference_value"] = base
        m["n_groups"] = K
        m["group_sizes"] = sizes
        m["trajectory_r2"] = round(r2, 6)
        m["holdout_c_index"] = round(c, 6)
        m["centroid_consistency"] = round(consistency, 6)
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
