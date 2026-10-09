"""
EXTRACTOR_SPEC:
  plan_file: solution.json
  schema: |
    {
      "caes_seq": [<float>, ...],     # 8760 项：压缩空气储能每小时控制量
      "battery_seq": [<float>, ...],  # 8760 项：锂电池每小时控制量
      "tp_seq": [<float>, ...]        # 8760 项：火电每小时控制量（可选, 缺省按常数 0.33）
    }
  notes: >
    全年风光火储运行优化。产物是三条长度为 8760 的控制序列，逐小时对应。
    caes_seq 取值允许 0 或 [-1,-0.33]（压缩储能）或 [0.86,1]（膨胀发电），其余值会按不达标处理；
    battery_seq 取值 [-1,1]（1 满放电, -1 满充电）；tp_seq 取值 [0.33,1]。
    三条序列都要有 8760 项, 每项是数值。选手若只交 caes_seq/battery_seq 也可, tp_seq 缺省按 0.33。
"""
from __future__ import annotations

import json
import os
import sys
from pathlib import Path
from typing import Any, Dict

import numpy as np
import pandas as pd

_HERE = Path(__file__).resolve().parent
PLAN_FILE = "solution.json"


class WhiteBoxSimulator:
    """风光火储综合能源系统逐小时仿真（纯 numpy 复刻，口径与线上一致）。"""

    def __init__(self, data_dir) -> None:
        self.wind_data = pd.read_csv(os.path.join(data_dir, "timeseries_wind_8760.csv")).iloc[:, -1].values
        self.pv_data = pd.read_csv(os.path.join(data_dir, "timeseries_pv_8760.csv")).iloc[:, -1].values
        self.eload_data = pd.read_csv(os.path.join(data_dir, "timeseries_load_8760.csv")).iloc[:, -1].values
        self.temp_data = pd.read_csv(os.path.join(data_dir, "timeseries_environment_8760.csv")).iloc[:, -1].values
        tp_df = pd.read_csv(os.path.join(data_dir, "timeseries_thermal_8760.csv"))
        self.tp_data = tp_df.drop_duplicates(subset=tp_df.columns[1]).iloc[:, -1].values
        bat_df = pd.read_csv(os.path.join(data_dir, "timeseries_battery_8760.csv"))
        self.bat_data = bat_df.iloc[:, -1].values
        caes_df = pd.read_csv(os.path.join(data_dir, "timeseries_caes_8760.csv"))
        self.caes_data = caes_df.iloc[:, -1].values

        self.pv_Pn = 2.6e8
        self.pv_eta = 0.95
        self.pv_Gstc = 1000.0
        self.pv_Tstc = 298.15
        self.pv_KT = 0.005
        self.pv_c = 0.25       # PV卖出电价 (元/kWh)

        self.wt_Pn = 3.0e8
        self.v_in = 3.0
        self.v_rated = 10.0
        self.v_out = 15.0
        self.wt_c = 0.35       # 风电卖出电价 (元/kWh)

        self.tp_P_max = 1.5e8
        self.tp_c = 0.4        # 火电度电成本 (元/kWh)

        self.bat_P_cap = 1e8
        self.bat_E_cap = 1.8e12
        self.bat_eta = 0.85
        self.bat_SOC_min = 0.1
        self.bat_SOC_max = 0.9
        self.bat_k = 0.0001
        self.bat_c1 = 0.2      # 充电成本 (元/kWh)
        self.bat_c2 = 0.4      # 放电收益 (元/kWh)

        self.caes_P_cap = 1.5e8
        self.caes_E_cap = 2.16e12
        self.caes_gas_soc_max, self.caes_gas_soc_min = 1.0, 0.6
        self.caes_hot_soc_max, self.caes_hot_soc_min = 0.95, 0.05
        self.caes_cold_soc_max, self.caes_cold_soc_min = 0.95, 0.05
        self.caes_k = 1.0
        self.caes_c1 = 0.2     # 充电成本 (元/kWh)
        self.caes_c2 = 0.4     # 放电收益 (元/kWh)

        self.eload_c = 0.5     # 用电收益 (元/kWh)

        self.grid_P_max = 5e8
        self.grid_P_min = -5e8
        self.grid_c_sale = 0.1  # 卖电电价
        self.grid_c_buy = 0.6   # 买电电价
        self.bus_k = 1e6        # 母线惩罚系数

    def _get_pv_power(self, t):
        G = self.pv_data[t + 1]
        if G <= 0:
            return 0.0
        T_air = self.temp_data[t + 1]
        T_c = T_air + G / 800 * 25
        P = self.pv_Pn * self.pv_eta * (G / self.pv_Gstc) * (1 - self.pv_KT * (T_c - self.pv_Tstc))
        return max(0, P)

    def _get_wt_power(self, t):
        v = self.wind_data[t + 1]
        if v < self.v_in or v > self.v_out:
            return 0.0
        elif v < self.v_rated:
            return self.wt_Pn * ((v**3 - self.v_in**3) / (self.v_rated**3 - self.v_in**3))
        else:
            return self.wt_Pn

    def simulate(self, seq_bat, seq_caes, seq_tp):
        bat_soc = 0.5
        bat_E = bat_soc * self.bat_E_cap
        caes_gas_soc = 0.8
        caes_hot_soc = 0.5
        caes_cold_soc = 0.5

        total_income = 0.0
        total_penalty = 0.0
        dt = 3600  # 1小时

        hard_constraint_triggered = False
        error_info: Dict[str, Any] = {}

        steps = min(len(seq_bat), len(self.pv_data), 8760)

        for t in range(steps):
            u_bat = seq_bat[t]
            u_caes = seq_caes[t]
            u_tp = seq_tp[t]

            P_bat = u_bat * self.bat_P_cap
            if P_bat > 0:
                bat_E -= (P_bat * dt / self.bat_eta)
            else:
                bat_E -= (P_bat * dt * self.bat_eta)
            bat_soc = bat_E / self.bat_E_cap

            if bat_soc > self.bat_SOC_max:
                total_penalty += self.bat_k * ((bat_soc - self.bat_SOC_max) * 100) ** 2
                hard_constraint_triggered = True
                error_info['bat_soc'] = f'Time {t}, SOC={bat_soc:.3f} > {self.bat_SOC_max}'
            elif bat_soc < self.bat_SOC_min:
                total_penalty += self.bat_k * ((self.bat_SOC_min - bat_soc) * 100) ** 2
                hard_constraint_triggered = True
                error_info['bat_soc'] = f'Time {t}, SOC={bat_soc:.3f} < {self.bat_SOC_min}'

            P_caes = u_caes * self.caes_P_cap
            if P_caes > 0:
                delta = (P_caes * dt) / self.caes_E_cap
                caes_gas_soc -= delta
                caes_hot_soc += delta
                caes_cold_soc -= delta
            elif P_caes < 0:
                delta = (-P_caes * dt * 0.75) / self.caes_E_cap
                caes_gas_soc += delta
                caes_hot_soc -= delta
                caes_cold_soc += delta

            if caes_gas_soc > self.caes_gas_soc_max or caes_gas_soc < self.caes_gas_soc_min:
                hard_constraint_triggered = True
                error_info['caes_gas'] = f'Time {t}, Gas SOC={caes_gas_soc:.2f}'
            if caes_hot_soc > self.caes_hot_soc_max or caes_hot_soc < self.caes_hot_soc_min:
                hard_constraint_triggered = True
                error_info['caes_hot'] = f'Time {t}, Hot SOC={caes_hot_soc:.2f}'
            if caes_cold_soc > self.caes_cold_soc_max or caes_cold_soc < self.caes_cold_soc_min:
                hard_constraint_triggered = True
                error_info['caes_cold'] = f'Time {t}, Cold SOC={caes_cold_soc:.2f}'

            P_PV_plan = self._get_pv_power(t)
            P_WT_plan = self._get_wt_power(t)
            P_TP = u_tp * self.tp_P_max
            P_Load_plan = self.eload_data[t + 1]

            P_res1 = P_PV_plan + P_WT_plan + P_TP + P_bat + P_caes - P_Load_plan

            if P_res1 >= 0:
                P_Grid_act = min(P_res1, self.grid_P_max)
                P_res1_after_grid = P_res1 - P_Grid_act
            else:
                P_Grid_act = max(P_res1, -self.grid_P_max)
                P_res1_after_grid = P_res1 - P_Grid_act

            if P_res1_after_grid > 0:
                P_WT_act = max(P_WT_plan - P_res1_after_grid, 0)
                remainder = P_res1_after_grid - (P_WT_plan - P_WT_act)
                P_PV_act = max(P_PV_plan - remainder, 0)
                P_res = remainder - (P_PV_plan - P_PV_act)
                P_Load_act = P_Load_plan
            elif P_res1_after_grid < 0:
                shortage = -P_res1_after_grid
                max_shed = P_Load_plan - 0.2 * P_Load_plan
                shed = min(shortage, max_shed)
                P_Load_act = P_Load_plan - shed
                P_res = shortage - shed
                P_PV_act = P_PV_plan
                P_WT_act = P_WT_plan
            else:
                P_PV_act = P_PV_plan
                P_WT_act = P_WT_plan
                P_Load_act = P_Load_plan
                P_res = 0.0

            pv_C = -P_PV_act * self.pv_c
            wt_C = -P_WT_act * self.wt_c
            tp_C = -P_TP * self.tp_c
            eload_C = P_Load_act * self.eload_c

            if P_Grid_act > 0:
                grid_C = P_Grid_act * self.grid_c_sale
            else:
                grid_C = P_Grid_act * self.grid_c_buy

            bat_C = -P_bat * self.bat_c2 if P_bat > 0 else -P_bat * self.bat_c1
            caes_C = -P_caes * self.caes_c2 if P_caes > 0 else -P_caes * self.caes_c1

            sum_C = (pv_C + wt_C + tp_C + eload_C + bat_C + caes_C + grid_C) / 3.6e6
            C_penality = (self.bus_k * P_res) ** 2
            inc = (sum_C - C_penality) * dt
            total_income += inc
            total_penalty += C_penality * dt

        return total_income, total_penalty, hard_constraint_triggered, error_info


def load_baseline() -> tuple:
    with open(_HERE / "baseline" / "reference_metrics.json", encoding="utf-8") as f:
        d = json.load(f)
    return float(d["reference_value"]), d.get("direction", "lower_is_better")


# 2026-08-25 修:去掉 quality 的 5.0 封顶 —— 好解一旦超过基准 5 倍就被压平,跟刷分的
#   混在一起分不出高低。前两轮批量去封顶漏了 min(max(q,0.0),5.0) 与 min(float(_q),5.0)
#   这两种写法(变量名不叫 quality),第三轮改用「看 m["quality_score"] 赋值右边」才扫净。
def evaluate(submission_dir: str, data_dir: str) -> Dict[str, Any]:
    m: Dict[str, Any] = dict(
        validity_score=0.0,
        quality_score=0.0,
        overall_score=0.0,
        reference_value=0.0,
    )
    try:
        reference_value, direction = load_baseline()
    except Exception as e:
        m["error_info"] = {"fatal": f"baseline 读取失败: {e}"}
        return m
    m["reference_value"] = reference_value

    plan_path = os.path.join(submission_dir, PLAN_FILE)
    if not os.path.exists(plan_path):
        m["error_info"] = {"hard_violations": [f"缺 {PLAN_FILE}"]}
        return m
    try:
        with open(plan_path, encoding="utf-8") as f:
            plan = json.load(f)
    except Exception as e:
        m["error_info"] = {"hard_violations": [f"{PLAN_FILE} 解析失败: {e}"]}
        return m

    caes = plan.get("caes_seq")
    bat = plan.get("battery_seq")
    tp = plan.get("tp_seq", np.full(8760, 0.33).tolist())

    errs = []
    if not isinstance(caes, list) or len(caes) < 8760:
        errs.append(f"caes_seq 缺失或不足 8760 项(实际 {len(caes) if isinstance(caes,list) else '非列表'})")
    if not isinstance(bat, list) or len(bat) < 8760:
        errs.append(f"battery_seq 缺失或不足 8760 项(实际 {len(bat) if isinstance(bat,list) else '非列表'})")
    if not isinstance(tp, list) or len(tp) < 8760:
        errs.append(f"tp_seq 缺失或不足 8760 项(实际 {len(tp) if isinstance(tp,list) else '非列表'})")
    if errs:
        m["error_info"] = {"hard_violations": errs[:8]}
        return m

    try:
        caes_a = np.array([float(x) for x in caes[:8760]])
        bat_a = np.array([float(x) for x in bat[:8760]])
        tp_a = np.array([float(x) for x in tp[:8760]])
    except (TypeError, ValueError) as e:
        m["error_info"] = {"hard_violations": [f"序列含非数值: {e}"]}
        return m

    # 取值边界：越界视为违反（hard），与线上口径一致
    if np.any((caes_a != 0) & (~((caes_a >= -1) & (caes_a <= -0.33))) & (~((caes_a >= 0.86) & (caes_a <= 1)))):
        m["error_info"] = {"hard_violations": ["caes_seq 存在取值区间外的值(须为 0/[-1,-0.33]/[0.86,1])"]}
        return m
    if np.any((bat_a < -1) | (bat_a > 1)):
        m["error_info"] = {"hard_violations": ["battery_seq 存在越界值(须在 [-1,1] 内)"]}
        return m
    if np.any((tp_a < 0.33) | (tp_a > 1)):
        m["error_info"] = {"hard_violations": ["tp_seq 存在越界值(须在 [0.33,1] 内)"]}
        return m

    sim = WhiteBoxSimulator(data_dir)
    total_income, total_penalty, hard, error_info = sim.simulate(bat_a, caes_a, tp_a)

    if hard:
        m["error_info"] = {"hard_violations": ["储能 SOC 越限"], "details": str(error_info)}
        return m

    m["error_info"] = {}
    m["validity_score"] = 1.0
    # 全年现金流(元)越高越好
    score = float(total_income)
    # 2026-08-23 修:原实现把原始目标值直接当 quality_score,与全库其余 case 的
    # 「相对 baseline 的倍数(≈1.0、cap 5.0)」口径不可比 —— 做任何跨 case 汇总都会被
    # 量级最大的那个 case 盖住。改为按 baseline 归一。
    _raw = float(score)
    _q = (_raw / reference_value) if reference_value > 0 else 0.0
    m["quality_score"] = round(max(float(_q), 0.0), 6)
    m["overall_score"] = m["quality_score"]
    m["player_objective"] = _raw
    m["total_income"] = float(total_income)
    m["total_penalty"] = float(total_penalty)
    return m


def main() -> None:
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--submission-dir", required=True)
    ap.add_argument("--data-dir", default=str(_HERE.parent / "data"))
    a = ap.parse_args()
    res = evaluate(a.submission_dir, a.data_dir)
    print(json.dumps(res, ensure_ascii=False, default=str))


if __name__ == "__main__":
    main()
