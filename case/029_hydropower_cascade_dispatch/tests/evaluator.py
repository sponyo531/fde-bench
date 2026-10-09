"""
EXTRACTOR_SPEC:
  plan_file: solution.json
  required_columns: []
  notes: >
    梯级水电站 7 天联合调度。选手的产物是一个 JSON 文件 solution.json:

        {
          "Q_out": [[5行×7列, 全部5座电站逐日出库流量, 单位 m³/s]]
        }

    - Q_out 第一维电站序号 0~4（站点A/站点B/站点C/站点D/站点E），第二维天数 0~6。
    - 评估器从 Q_out 独立重算水位、出力、总发电量，不信任任何自报汇总值。

    Extractor: 从选手工作区找到最终调度结果（可能在 solution.json / *.json /
    脚本输出里，一份含 Q_out 的决策清单），规范化成上面的结构写到输出目录的
    solution.json；不执行选手代码、不改动数值。若选手输出的是逐站 CSV 或
    result.json（含 Q_out/P/E/total_energy 字段），只抽取 Q_out 写入 solution.json。
"""
from __future__ import annotations

import argparse
import json
import math
import os
import traceback
from pathlib import Path
from typing import Any, Dict, List

# ─── 固定参数（与用户原话 + station_params.json 一致）─────────────────────
STATIONS = ["站点A", "站点B", "站点C", "站点D", "站点E"]
P_MAX = [170, 260, 360, 66, 132]
K = [9.0, 9.2, 9.5, 8.8, 8.5]
Z_TAIL = [1740, 1000, 720, 660, 595]
Z_INIT = [1835, 1120, 840]
Z_NORMAL = [1845, 1130, 850]
Z_DEAD = [1802, 1090, 790]
Z_MAX = [1845, 1130, 845]
V_TOTAL = [7.6e8, 5.55e8, 53.9e8]
Q_LATERAL = [0, 50, 30, 10, 5]
Q_INFLOW = [800, 867, 933, 1000, 1067, 1133, 1200]
H_NET_RUNOFF = [28.0, 35.0]
ECO_MIN = 50.0
DAYS = 7
SEC_PER_DAY = 86400
RAMP_RATIO = 0.2
EVAL_TOL = 0.05

PLAN_FILE = "solution.json"
_HERE = Path(__file__).resolve().parent


def load_baseline() -> tuple:
    with open(_HERE / "baseline" / "reference_metrics.json", encoding="utf-8") as f:
        d = json.load(f)
    return float(d["reference_value"]), d.get("direction", "higher_is_better")


def _z_to_v(z: float, i: int) -> float:
    return (z - Z_DEAD[i]) / (Z_NORMAL[i] - Z_DEAD[i]) * V_TOTAL[i]


def _v_to_z(v: float, i: int) -> float:
    return Z_DEAD[i] + v / V_TOTAL[i] * (Z_NORMAL[i] - Z_DEAD[i])


def evaluate(submission_dir: str, data_dir: str) -> Dict[str, Any]:
    m: Dict[str, Any] = {
        "validity_score": 0.0, "quality_score": 0.0, "overall_score": 0.0, "error_info": {},
    }
    try:
        plan_path = os.path.join(submission_dir, PLAN_FILE)
        if not os.path.exists(plan_path):
            m["error_info"] = {"fatal": [f"缺 {PLAN_FILE} in {submission_dir}"]}
            return m

        with open(plan_path, "r", encoding="utf-8") as f:
            sol = json.load(f)

        Q_out = sol.get("Q_out")
        if not isinstance(Q_out, list):
            m["error_info"] = {"fatal": ["solution.json 缺 Q_out 字段"]}
            return m
        if len(Q_out) != 5 or any(len(r) != DAYS for r in Q_out):
            m["error_info"] = {"fatal": [f"Q_out 维度不是 5x{DAYS}"]}
            return m
        try:
            Q_out = [[float(x) for x in r] for r in Q_out]
        except (TypeError, ValueError):
            m["error_info"] = {"fatal": ["Q_out 含非数值"]}
            return m

        violations: List[str] = []

        V_init = [_z_to_v(Z_INIT[i], i) for i in range(3)]
        Z_calc = [[0.0] * DAYS for _ in range(3)]
        V = list(V_init)
        for t in range(DAYS):
            Q_in = [
                Q_INFLOW[t],
                Q_out[0][t] + Q_LATERAL[1],
                Q_out[1][t] + Q_LATERAL[2],
            ]
            for i in range(3):
                V[i] += (Q_in[i] - Q_out[i][t]) * SEC_PER_DAY
                Z_calc[i][t] = _v_to_z(V[i], i)
            q_ds3_expected = Q_out[2][t] + Q_LATERAL[3]
            q_gz_expected = Q_out[3][t] + Q_LATERAL[4]
            if abs(Q_out[3][t] - q_ds3_expected) > 1.0:
                violations.append(f"水量平衡：站点D第{t+1}天出库{Q_out[3][t]:.2f}应~{q_ds3_expected:.2f}")
            if abs(Q_out[4][t] - q_gz_expected) > 1.0:
                violations.append(f"水量平衡：站点E第{t+1}天出库{Q_out[4][t]:.2f}应~{q_gz_expected:.2f}")

        for i in range(3):
            for t in range(DAYS):
                z = Z_calc[i][t]
                if z < Z_DEAD[i] - EVAL_TOL:
                    violations.append(f"水位下限：{STATIONS[i]} 第{t+1}天水位{z:.3f}m < 死水位{Z_DEAD[i]}m")
                if z > Z_MAX[i] + EVAL_TOL:
                    violations.append(f"水位上限：{STATIONS[i]} 第{t+1}天水位{z:.3f}m > 上限{Z_MAX[i]}m")

        for i in range(5):
            for t in range(DAYS):
                if Q_out[i][t] < ECO_MIN - EVAL_TOL:
                    violations.append(f"生态流量：{STATIONS[i]} 第{t+1}天出库{Q_out[i][t]:.2f}<50")

        P_calc = [[0.0] * DAYS for _ in range(5)]
        for i in range(3):
            for t in range(DAYS):
                z_prev = Z_INIT[i] if t == 0 else Z_calc[i][t - 1]
                z_avg = z_prev * 0.5 + Z_calc[i][t] * 0.5
                H_net = max(z_avg - Z_TAIL[i], 1.0)
                p = K[i] * Q_out[i][t] * H_net / 10000.0
                P_calc[i][t] = min(p, P_MAX[i])
        for j, i in enumerate([3, 4]):
            H_net = H_NET_RUNOFF[j]
            for t in range(DAYS):
                p = K[i] * Q_out[i][t] * H_net / 10000.0
                P_calc[i][t] = min(p, P_MAX[i])

        for i in range(5):
            ramp_limit = P_MAX[i] * RAMP_RATIO
            for t in range(1, DAYS):
                diff = abs(P_calc[i][t] - P_calc[i][t - 1])
                if diff > ramp_limit + EVAL_TOL:
                    violations.append(f"爬坡约束：{STATIONS[i]} 第{t}→{t+1}天变化{diff:.3f}>{ramp_limit:.1f}")

        if violations:
            m["error_info"] = {"constraint": violations[:8]}
            return m

        m["validity_score"] = 1.0

        total_energy = sum(P_calc[i][t] for i in range(5) for t in range(DAYS)) * 24.0
        if not math.isfinite(total_energy):
            m["validity_score"] = 0.0
            m["error_info"] = {"exception": "total_energy is NaN/Inf"}
            return m

        baseline, direction = load_baseline()
        if direction == "higher_is_better":
            quality = total_energy / baseline if baseline > 0 else 0.0
        else:
            quality = baseline / total_energy if total_energy > 0 else 0.0

        m["quality_score"] = round(quality, 6)
        m["overall_score"] = m["quality_score"]
        m["player_objective"] = round(total_energy, 6)
        m["reference_value"] = baseline
        m["total_energy_wankwh"] = round(total_energy, 4)
        m["energy_by_station"] = {
            STATIONS[i]: round(sum(P_calc[i][t] * 24 for t in range(DAYS)), 4)
            for i in range(5)
        }
        return m

    except Exception as e:
        m["validity_score"] = 0.0
        m["error_info"] = {"exception": str(e), "traceback": traceback.format_exc()[-800:]}
        return m


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--submission-dir", required=True)
    ap.add_argument("--data-dir", default=str(_HERE.parent / "data"))
    args = ap.parse_args()
    print(json.dumps(evaluate(args.submission_dir, args.data_dir), ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
