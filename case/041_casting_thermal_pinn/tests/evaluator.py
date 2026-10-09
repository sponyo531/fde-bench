"""
EXTRACTOR_SPEC:
  plan_file: predictions.csv
  required_columns: [time_s, i, j, k, temperature]
  notes: >
    铸件凝固温度场跨零件推演。选手产物是中间厚度件 part2 在若干指定时刻、
    每个网格节点上的温度。

    输出一张 CSV，表头恰好为 time_s,i,j,k,temperature，每个 (时刻, 节点) 一行。
    若选手用了别的列名（t/time 代表时刻，x/y/z 或 ix/iy/iz 代表索引，
    T/temp/value/pred 代表温度），映射到上面五列。
    若选手交的是与输入同格式的多个 txt 温度场文件（如 t1.2.txt，前两行是网格头部、
    后续行为 i j k T mat），按文件名解析出时刻，把每行展开成一条记录，丢弃 mat 列。
    若选手交的是 npy/npz/json 里的三维数组，按 i 最外、k 最内的顺序展开成 i j k，
    索引从 1 开始计数。
    时刻写成与待预测时刻表一致的数值（如 1.2、2.4）。温度保留原始精度，
    不要四舍五入、不要裁剪到某个区间、不要自行补算缺失的节点或时刻。
"""
from __future__ import annotations

import argparse
import json
import os
import traceback
from pathlib import Path
from typing import Any, Dict, Tuple

import numpy as np
import pandas as pd

PLAN_FILE = "predictions.csv"
TARGET_PART = "part2"
T_LOW, T_HIGH = -10.0, 1750.0
MONO_TOL = 1.0          # ℃，容忍回归器的数值毛刺，不放宽业务方向
_HERE = Path(__file__).resolve().parent


# 2026-08-25 修:去掉 quality 的 5.0 封顶 —— 好解一旦超过基准 5 倍就被压平,
# 跟刷分的混在一起分不出高低。分母为零(完美解)时给一个有限哨兵值,避免 inf 破坏 JSON。
PERFECT_Q = 1e6

def load_baseline() -> Tuple[float, str]:
    with open(_HERE / "baseline" / "reference_metrics.json", encoding="utf-8") as f:
        d = json.load(f)
    return float(d["reference_value"]), d.get("direction", "lower_is_better")


def load_field(path: str) -> Tuple[np.ndarray, np.ndarray]:
    """读一个温度场文件, 返回 (温度数组, 材料数组), 形状 (nx, ny, nz)。"""
    with open(path, encoding="utf-8-sig") as f:
        lines = f.readlines()
    nx, ny, nz = (int(v) for v in lines[0].split())
    T = np.full((nx, ny, nz), np.nan)
    M = np.zeros((nx, ny, nz), dtype=int)
    for line in lines[2:]:
        p = line.split()
        if len(p) != 5:
            continue
        T[int(p[0]) - 1, int(p[1]) - 1, int(p[2]) - 1] = float(p[3])
        M[int(p[0]) - 1, int(p[1]) - 1, int(p[2]) - 1] = int(p[4])
    return T, M


def evaluate(submission_dir: str, data_dir: str) -> Dict[str, Any]:
    m: Dict[str, Any] = {
        "validity_score": 0.0, "quality_score": 0.0, "overall_score": 0.0, "error_info": {},
    }
    try:
        path = os.path.join(submission_dir, PLAN_FILE)
        if not os.path.exists(path):
            m["error_info"] = {"fatal": [f"缺 {PLAN_FILE}"]}
            return m
        sub = pd.read_csv(path, encoding="utf-8-sig")
        need = ["time_s", "i", "j", "k", "temperature"]
        lack = [c for c in need if c not in sub.columns]
        if lack:
            m["error_info"] = {"hard_violations": [f"predictions.csv 缺列 {lack}"]}
            return m

        # ── 待预测时刻表与初始温度场都从 data/ 独立重建 ──
        tg = pd.read_csv(os.path.join(data_dir, "prediction_targets.csv"), encoding="utf-8-sig")
        times = [float(t) for t in tg.loc[tg["part"].astype(str) == TARGET_PART, "time_s"]]
        T0, mat = load_field(os.path.join(data_dir, TARGET_PART, "initial_temp.txt"))
        nx, ny, nz = T0.shape
        n_nodes = nx * ny * nz

        errs = []
        for c in need:
            sub[c] = pd.to_numeric(sub[c], errors="coerce")

        n_bad = int(sub[need].isna().any(axis=1).sum())
        if n_bad:
            errs.append(f"{n_bad} 行存在空值或无法解析为数值的字段")
            sub = sub.dropna(subset=need)

        # 逐时刻张成稠密网格立方体
        grids = {}
        for t in times:
            block = sub[np.isclose(sub["time_s"].to_numpy(dtype=float), t, atol=1e-6)]
            g = np.full((nx, ny, nz), np.nan)
            ii = block["i"].to_numpy(dtype=int) - 1
            jj = block["j"].to_numpy(dtype=int) - 1
            kk = block["k"].to_numpy(dtype=int) - 1
            ok = (ii >= 0) & (ii < nx) & (jj >= 0) & (jj < ny) & (kk >= 0) & (kk < nz)
            if int((~ok).sum()):
                errs.append(f"t={t} 有 {int((~ok).sum())} 行的 i/j/k 越出 part2 网格范围")
            g[ii[ok], jj[ok], kk[ok]] = block["temperature"].to_numpy(dtype=float)[ok]
            n_dup = int(ok.sum()) - len(set(zip(ii[ok].tolist(), jj[ok].tolist(), kk[ok].tolist())))
            if n_dup > 0:
                errs.append(f"t={t} 有 {n_dup} 行重复的节点索引")
            miss = int(np.isnan(g).sum())
            if miss:
                errs.append(f"t={t} 有 {miss}/{n_nodes} 个节点没有给出温度")
            grids[t] = g

        if not errs:
            cast = mat == 2
            mold = mat == 1
            for t in times:
                g = grids[t]
                lo = float(np.nanmin(g))
                hi = float(np.nanmax(g))
                if lo < T_LOW or hi > T_HIGH:
                    errs.append(f"HC2 t={t} 温度越界: 最低 {lo:.2f}℃ 最高 {hi:.2f}℃，"
                                f"允许区间 [{T_LOW}, {T_HIGH}]")
                up = (g - T0)[cast]
                n_up = int((up > MONO_TOL).sum())
                if n_up:
                    errs.append(f"HC1 t={t} 有 {n_up} 个铸件节点高于初始温度，最大超出 {up.max():.2f}℃")
                dn = (T0 - g)[mold]
                n_dn = int((dn > MONO_TOL).sum())
                if n_dn:
                    errs.append(f"HC1 t={t} 有 {n_dn} 个砂型节点低于初始温度，最大低出 {dn.max():.2f}℃")

        if errs:
            m["error_info"] = {"hard_violations": errs[:12], "total": len(errs)}
            return m

        m["validity_score"] = 1.0

        # ── 目标值: 用私有仿真真值逐时刻算 RMSE, 再对时刻取平均 ──
        per_time = {}
        for t in times:
            ref, _ = load_field(str(_HERE / "_private" / TARGET_PART / (f"t{t:.1f}.txt")))
            rmse = float(np.sqrt(np.mean((grids[t] - ref) ** 2)))
            per_time[f"t{t:.1f}"] = round(rmse, 4)
        avg_rmse = float(np.mean(list(per_time.values())))

        base, direction = load_baseline()
        if direction == "lower_is_better":
            quality = base / avg_rmse if avg_rmse > 0 else PERFECT_Q
        else:
            quality = avg_rmse / base if base > 0 else 0.0

        m["quality_score"] = round((quality), 6)
        m["overall_score"] = m["quality_score"]
        m["player_objective"] = round(avg_rmse, 6)
        m["reference_value"] = base
        m["rmse_per_time"] = per_time
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
