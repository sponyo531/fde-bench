"""
EXTRACTOR_SPEC:
  plan_file: solution.json
  required_columns: [num_shifts, ma_schedule, ca_schedule]
  notes: >
    两阶段排程（热工CA + 机加MA）产物。选手须输出一个 JSON 文件 solution.json，
    描述每台设备每个班次的生产计划。期望格式：
      {
        "num_shifts": 32,
        "products": ["100060", ...],                 // 可选，缺省时从 data 独立获取
        "ma_schedule": {                             // 机加设备排程
          "<machine_id_str>": {
            "<shift_int_as_str>": {"product": "<pid_or_null>", "qty": 0, "changeover": false}
          }
        },
        "ca_schedule": {                             // 热工设备排程
          "<machine_id_str>": {
            "<shift_int_as_str>": {"product": "<pid_or_null>", "qty": 0,
                                   "changeover": false, "maintenance": false}
          }
        }
      }

    常见选手产物差异及规整方式：
    - machine_id / product_id / shift 统一转成字符串键；product 可为 null 表示空闲。
    - 若产出为逐行长表 CSV（machine,shift,stage,product,qty,changeover,maintenance），
      按 stage 分别聚合成 ma_schedule / ca_schedule 的嵌套 dict。
    - qty 必须是整数件数；缺失的 (machine,shift) 组合视为空闲（product=null,qty=0）。
    - 初始库存一律从 data/production_data.xlsx 独立读取；提交中的同名字段会被忽略。
    evaluator 不执行选手代码，只读产物文件；所有约束与目标值均从 data 独立重算。
"""

import argparse
import json
import math
import os
import traceback
from collections import defaultdict
from pathlib import Path

PLAN_FILE = "solution.json"
_HERE = os.path.dirname(os.path.abspath(__file__))
_DATA = os.path.join(_HERE, "..", "data")

MAX_MA_CO_PER_SHIFT = 4
MAX_CA_EVENT_PER_SHIFT = 4
MAINT_LIMIT = 4
PERFECT_Q = 1e6


# ── 从 data 独立加载参考数据 ─────────────────────────────────────────────────
def load_reference_data(data_dir):
    import pandas as pd
    path = os.path.join(data_dir, "production_data.xlsx")
    xl = pd.ExcelFile(path)
    orders_df = pd.read_excel(xl, "订单数据")
    products_df = pd.read_excel(xl, "产品数据")
    ops_df = pd.read_excel(xl, "工序数据")
    modes_df = pd.read_excel(xl, "工序模式数据")
    machines_df = pd.read_excel(xl, "机器数据")
    inventory_df = pd.read_excel(xl, "初始库存数据")

    products = [str(p) for p in products_df["产品ID"]]

    ca_op_id, ma_op_id = {}, {}
    for _, r in ops_df.iterrows():
        pid = str(r["产品ID"])
        if r["工序名称"] == "热工(CA)":
            ca_op_id[pid] = str(r["工序ID"])
        elif r["工序名称"] == "机加(MA)":
            ma_op_id[pid] = str(r["工序ID"])

    ca_machines = set(str(m) for m in machines_df[machines_df["资源组ID"] == "压铸"]["机器ID"])
    ma_machines = set(str(m) for m in machines_df[machines_df["资源组ID"] == "机加"]["机器ID"])

    cap_ca_normal, cap_ca_event, cap_ma_normal = defaultdict(dict), defaultdict(dict), defaultdict(dict)
    ca_compat, ma_compat = defaultdict(set), defaultdict(set)
    for _, r in modes_df.iterrows():
        pid = str(r["产品ID"]); mid = str(r["机器ID"])
        cap = float(r["产能(件/小时)"]); qr = float(r["合格率"])
        if r["工序名称"] == "热工(CA)":
            cap_ca_normal[mid][pid] = math.ceil(24 * cap * qr)
            cap_ca_event[mid][pid] = math.ceil(20 * cap * qr)
            ca_compat[mid].add(pid)
        elif r["工序名称"] == "机加(MA)":
            cap_ma_normal[mid][pid] = math.ceil(24 * cap * qr)
            ma_compat[mid].add(pid)

    ca_init_inv, ma_init_inv = defaultdict(float), defaultdict(float)
    for _, r in inventory_df.iterrows():
        pid = str(r["产品ID"]); op = str(r["工序ID"]); qty = float(r["初始库存数量"])
        if pid in ca_op_id and op == ca_op_id[pid]:
            ca_init_inv[pid] = qty
        elif pid in ma_op_id and op == ma_op_id[pid]:
            ma_init_inv[pid] = qty

    max_dl = int(orders_df["交付截止日期(秒)"].max())
    num_shifts_ref = max(int(max_dl // 86400) + 2, 32)

    prod_weight = defaultdict(lambda: 1.0)
    demand_by_shift = defaultdict(lambda: defaultdict(float))
    for _, r in orders_df.iterrows():
        pid = str(r["产品ID"]); qty = float(r["订单数量"])
        pri = float(r["优先级"]); dl_t = int(float(r["交付截止日期(秒)"]) // 86400)
        demand_by_shift[pid][dl_t] += qty
        prod_weight[pid] = max(prod_weight[pid], pri)

    cum_demand = {}
    for pid in products:
        c = 0.0; cum_demand[pid] = {}
        for t in range(num_shifts_ref):
            c += demand_by_shift[pid].get(t, 0.0)
            cum_demand[pid][t] = c

    return dict(
        products=products, ca_machines=ca_machines, ma_machines=ma_machines,
        ca_compat=dict(ca_compat), ma_compat=dict(ma_compat),
        cap_ca_normal=dict(cap_ca_normal), cap_ca_event=dict(cap_ca_event),
        cap_ma_normal=dict(cap_ma_normal),
        ca_init_inv=dict(ca_init_inv), ma_init_inv=dict(ma_init_inv),
        prod_weight=dict(prod_weight), cum_demand=cum_demand,
        num_shifts_ref=num_shifts_ref,
    )


# ── 约束校验 + 独立重算目标值 ────────────────────────────────────────────────
def check_and_score(result, ref):
    violations = []
    # 2026-08-23 修:原实现直接采信选手自报的 num_shifts。这是可刷分的——
    # ① T 越小,逐班欠量 max(0, cum_demand[t] - cum) 累计的项数越少;
    # ② cum_demand 只有 0..num_shifts_ref-1 的键,T 超出后 .get(T-1, 0) 恒返回 0,
    #    末班欠量直接归零。排程期长度本来就由数据里的最晚交期算出(num_shifts_ref),
    #    不是选手可选的量。现在以数据为准,自报值只做一致性提示。
    T = int(ref["num_shifts_ref"])
    horizon_notes = []
    _T_self = result.get("num_shifts")
    if _T_self is not None:
        try:
            if int(_T_self) != T:
                horizon_notes.append(
                    f"自报 num_shifts={int(_T_self)} 与数据推出的排程期 {T} 不一致，已按 {T} 计算")
        except (TypeError, ValueError):
            horizon_notes.append(f"自报 num_shifts 无法解析({_T_self!r})，已按 {T} 计算")
    last = T - 1
    ma_sched = result.get("ma_schedule", {}) or {}
    ca_sched = result.get("ca_schedule", {}) or {}
    # 初始库存是输入事实，不是决策量。若信任提交自报值，可伪造巨额
    # 库存绕过冷却/非负约束，并把欠量刷到 0。
    ca_init = {str(k): float(v) for k, v in ref["ca_init_inv"].items()}
    ma_init = {str(k): float(v) for k, v in ref["ma_init_inv"].items()}

    def g(shifts, t):
        return shifts.get(str(t), shifts.get(t, {})) or {}

    # (1) 设备兼容性
    for mid, shifts in ma_sched.items():
        compat = ref["ma_compat"].get(str(mid), set())
        for t_str, info in shifts.items():
            pid = info.get("product")
            if pid is not None and str(pid) not in compat:
                violations.append(f"MA设备{mid}班次{t_str}生产不兼容产品{pid}")
    for mid, shifts in ca_sched.items():
        compat = ref["ca_compat"].get(str(mid), set())
        for t_str, info in shifts.items():
            pid = info.get("product")
            if pid is not None and str(pid) not in compat:
                violations.append(f"CA设备{mid}班次{t_str}生产不兼容产品{pid}")

    # (2) MA 产能离散性（t=0 受 CA 初始库存上限；t>0 严格 {0, cap}）
    for mid, shifts in ma_sched.items():
        caps = ref["cap_ma_normal"].get(str(mid), {})
        for t_str, info in shifts.items():
            t_int = int(t_str)
            pid = info.get("product"); qty = int(info.get("qty", 0))
            co = bool(info.get("changeover", False))
            if pid is None:
                if qty != 0:
                    violations.append(f"MA设备{mid}班次{t_str}无产品但qty={qty}")
                continue
            pid = str(pid); expected = caps.get(pid, 0)
            if co:
                if qty != 0:
                    violations.append(f"MA设备{mid}班次{t_str}换型但qty={qty}≠0")
            elif t_int == 0:
                ca_inv0 = ca_init.get(pid, ref["ca_init_inv"].get(pid, 0))
                upper = min(expected, ca_inv0)
                if qty > upper + 0.5:
                    violations.append(f"MA设备{mid}班次0产品{pid} qty={qty}>min(cap={expected},ca_inv={ca_inv0:.0f})")
            else:
                if qty != 0 and qty != expected:
                    violations.append(f"MA设备{mid}班次{t_str}产品{pid} qty={qty}∉{{0,{expected}}}")

    # (3) CA 产能离散性 + 换型/保养互斥
    for mid, shifts in ca_sched.items():
        cap_n = ref["cap_ca_normal"].get(str(mid), {})
        cap_e = ref["cap_ca_event"].get(str(mid), {})
        for t_str, info in shifts.items():
            pid = info.get("product"); qty = int(info.get("qty", 0))
            co = bool(info.get("changeover", False)); mnt = bool(info.get("maintenance", False))
            if pid is None:
                if qty != 0:
                    violations.append(f"CA设备{mid}班次{t_str}无产品但qty={qty}")
                continue
            pid = str(pid)
            if co and mnt:
                violations.append(f"CA设备{mid}班次{t_str}换型保养同时发生")
            expected = (cap_e if (co or mnt) else cap_n).get(pid, 0)
            if qty != expected:
                violations.append(f"CA设备{mid}班次{t_str}产品{pid} qty={qty}≠{expected}(event={co or mnt})")

    # (4) t=0 MA 产量 ≤ CA 初始库存（产品级汇总）
    ma_t0 = defaultdict(float)
    for mid, shifts in ma_sched.items():
        info = g(shifts, 0)
        pid = info.get("product"); qty = float(info.get("qty", 0))
        if pid and qty > 0:
            ma_t0[str(pid)] += qty
    for pid, qty in ma_t0.items():
        ca_inv0 = ca_init.get(pid, ref["ca_init_inv"].get(pid, 0))
        if qty > ca_inv0 + 0.5:
            violations.append(f"t=0产品{pid} MA产量{qty:.0f}>CA初始库存{ca_inv0:.0f}")

    # (5) MA 换型：每日≤4、首末班次=0
    ma_co = defaultdict(int)
    for mid, shifts in ma_sched.items():
        for t_str, info in shifts.items():
            if info.get("changeover"):
                ma_co[int(t_str)] += 1
    for t, c in ma_co.items():
        if c > MAX_MA_CO_PER_SHIFT:
            violations.append(f"MA班次{t}换型{c}次>4")
    if ma_co.get(0, 0) > 0:
        violations.append(f"MA首班次换型{ma_co[0]}次")
    if ma_co.get(last, 0) > 0:
        violations.append(f"MA末班次换型{ma_co[last]}次")

    # (6) CA 事件：每日≤4、首班次=0、末班次无换型
    ca_ev = defaultdict(int)
    for mid, shifts in ca_sched.items():
        for t_str, info in shifts.items():
            if info.get("changeover") or info.get("maintenance"):
                ca_ev[int(t_str)] += 1
    for t, c in ca_ev.items():
        if c > MAX_CA_EVENT_PER_SHIFT:
            violations.append(f"CA班次{t}事件{c}次>4")
    if ca_ev.get(0, 0) > 0:
        violations.append(f"CA首班次事件{ca_ev[0]}次")
    ca_co_last = sum(1 for mid, shifts in ca_sched.items() if g(shifts, last).get("changeover"))
    if ca_co_last > 0:
        violations.append(f"CA末班次换型{ca_co_last}次")

    # (7) 热工强制保养：连续生产同产品 >4 班次未保养
    for mid, shifts in ca_sched.items():
        consec = 0; cur = None
        for t in range(T):
            info = g(shifts, t)
            pid = info.get("product"); co = info.get("changeover"); mnt = info.get("maintenance")
            if pid is None:
                continue  # 空闲不打断
            p = str(pid)
            if co:
                cur = p; consec = 1
            elif mnt:
                cur = p; consec = 0
            else:
                if p == cur:
                    consec += 1
                    if consec > MAINT_LIMIT:
                        violations.append(f"CA设备{mid}班次{t}连续{consec}班未保养(强制保养违反)")
                else:
                    cur = p; consec = 1

    # (7b) 换型语义绑定：独立重算每台设备的“已装模产品(setup)”状态序列，
    #      换产品必须付出换型代价——相邻两个有产出班次产品不同却无换型 → 违规。
    #      语义：机器初始模具未定(setup=None)；某产品的首次投产视为免费初始装模，
    #      此后任何产品切换都必须在此之前发生一次 changeover 班次(该班 qty=0)。
    #      仅“有产出(qty>0)”的班次校验/建立 setup；空闲或 qty=0 不改变 setup。
    #      换型班次(co=True)把 setup 切换为该班标注的新产品(qty 已在别处校验=0)。
    #      这样“换产品”无法免费，必须消耗一个换型班次并受“每日≤4次”限制约束。
    def check_setup_continuity(sched, label, is_ca):
        for mid, shifts in sched.items():
            setup = None
            for t in range(T):
                info = g(shifts, t)
                pid = info.get("product"); qty = int(info.get("qty", 0) or 0)
                co = bool(info.get("changeover", False))
                mnt = bool(info.get("maintenance", False)) if is_ca else False
                if co:
                    # 换型：确立新的装模产品（换到哪个产品）
                    if pid is not None:
                        setup = str(pid)
                    continue
                if pid is None or qty <= 0:
                    # 空闲 / qty=0（含 CA 保养班若无产出）不改变、也不校验 setup
                    if mnt and pid is not None:
                        # 保养不换产品：产品须与当前 setup 一致（若已确立）
                        if setup is not None and str(pid) != setup:
                            violations.append(
                                f"{label}设备{mid}班次{t}保养产品{pid}≠当前装模{setup}(保养不换产品,须先换型)")
                    continue
                p = str(pid)
                if setup is None:
                    setup = p  # 首次投产：免费初始装模
                elif p != setup:
                    violations.append(
                        f"{label}设备{mid}班次{t}产出产品{p}≠当前装模{setup}且未换型(换产品必须先换型)")
                    setup = p  # 继续检查后续班次
    check_setup_continuity(ma_sched, "MA", is_ca=False)
    check_setup_continuity(ca_sched, "CA", is_ca=True)

    # (8) 库存非负 + 冷却：MA 在 t 只能消耗 t-1 及之前 CA 产出
    ca_prod = defaultdict(lambda: defaultdict(float))
    for mid, shifts in ca_sched.items():
        for t_str, info in shifts.items():
            pid = info.get("product"); qty = float(info.get("qty", 0))
            if pid and qty > 0:
                ca_prod[str(pid)][int(t_str)] += qty
    ma_prod = defaultdict(lambda: defaultdict(float))
    for mid, shifts in ma_sched.items():
        for t_str, info in shifts.items():
            pid = info.get("product"); qty = float(info.get("qty", 0))
            if pid and qty > 0:
                ma_prod[str(pid)][int(t_str)] += qty

    for pid in ref["products"]:
        ca_avail = ca_init.get(pid, ref["ca_init_inv"].get(pid, 0))
        neg = False
        for t in range(T):
            consume = ma_prod[pid].get(t, 0)
            ca_avail -= consume
            if ca_avail < -0.5:
                violations.append(f"CA库存负值(冷却): 产品{pid}班次{t} 可用={ca_avail+consume:.0f}<消耗={consume:.0f}")
                neg = True; break
            ca_avail += ca_prod[pid].get(t, 0)
        if neg:
            continue

    validity = 1 if not violations else 0

    score = None; fulfill = None
    if validity == 1:
        total_delay = 0.0; total_demand = 0.0; total_fulfilled = 0.0
        for pid in ref["products"]:
            w = ref["prod_weight"].get(pid, 1.0)
            cum = ma_init.get(pid, ref["ma_init_inv"].get(pid, 0))
            for t in range(T):
                cum += ma_prod[pid].get(t, 0)
                total_delay += w * max(0.0, ref["cum_demand"][pid].get(t, 0) - cum)
            cum_d = ref["cum_demand"][pid].get(T - 1, 0)
            total_demand += cum_d
            total_fulfilled += min(cum, cum_d)
        score = total_delay
        fulfill = total_fulfilled / total_demand if total_demand > 0 else 1.0

    return {
        "validity": validity,
        "weighted_delay": score,
        "fulfill_rate": fulfill,
        "num_shifts": T,
        "horizon_notes": horizon_notes,
        "violation_count": len(violations),
        "violations": violations[:15],
    }


def load_baseline():
    with open(os.path.join(_HERE, "baseline", "reference_metrics.json")) as f:
        return float((lambda _d:_d.get("reference_value",_d.get("baseline_cost")))(json.load(f)))


def evaluate(file_path, data_dir):
    metrics = {"validity_score": 0.0, "quality_score": 0.0, "overall_score": 0.0, "error_info": {}}
    try:
        ref = load_reference_data(data_dir)
        baseline_cost = load_baseline()

        if not os.path.exists(file_path):
            metrics["error_info"] = {"fatal": [f"File not found: {file_path}"]}
            return metrics
        with open(file_path, "r", encoding="utf-8") as f:
            sub = json.load(f)

        if "ma_schedule" not in sub or "ca_schedule" not in sub:
            metrics["error_info"] = {"schema": ["缺少 ma_schedule 或 ca_schedule 字段"]}
            return metrics

        res = check_and_score(sub, ref)

        if res["validity"] == 0:
            metrics["error_info"] = {"constraint": res["violations"]}
            metrics["violation_count"] = res["violation_count"]
            return metrics

        metrics["validity_score"] = 1.0
        player_value = res["weighted_delay"]
        # 最小化目标：quality = baseline / player（不加 min(,1) 截断）
        if player_value is None:
            metrics["error_info"] = {"fatal": ["weighted_delay 计算失败"]}
            return metrics
        if player_value <= 0:
            quality = 1.0 if baseline_cost <= 0 else PERFECT_Q
        else:
            quality = baseline_cost / player_value
        metrics["quality_score"] = round(quality, 4)
        metrics["overall_score"] = round(quality, 4)
        metrics["player_objective"] = round(player_value, 2)
        metrics["reference_value"] = baseline_cost
        metrics["fulfill_rate"] = round(res["fulfill_rate"], 4) if res["fulfill_rate"] is not None else None
        metrics["num_shifts"] = res["num_shifts"]
    except Exception as e:
        metrics["error_info"] = {"exception": str(e), "traceback": traceback.format_exc()}
    return metrics


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--submission-dir", type=Path, required=True)
    parser.add_argument("--data-dir", type=Path, default=None)
    args = parser.parse_args()
    data_dir = str(args.data_dir) if args.data_dir else _DATA
    result = evaluate(str(args.submission_dir / PLAN_FILE), data_dir)
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
