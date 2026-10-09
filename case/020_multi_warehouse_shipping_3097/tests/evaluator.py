"""
EXTRACTOR_SPEC:
  plan_file: solution.json
  required_columns: [orders, transfers]
  notes: >
    The agent must produce a JSON file `solution.json` describing the shipping plan
    for every one of the 3097 orders plus any inter-warehouse transfers. Expected shape:
      {
        "orders": [
          {"ORDER_CODE": "7398...", "MATERIAL_CODE": "CEAC...", "QTY": 100,
           "ship_date": "2024-04-28" | null, "warehouse": "WH_014" | null,
           "is_delayed": false, "otypenew": "kx|kp|yc|zc|np"},
          ...
        ],
        "transfers": [
          {"machine_id": "MAT_0096", "from_warehouse": "WH_014",
           "to_warehouse": "WH_001", "qty": 300.0,
           "transfer_type": "wms|aps", "trigger_date": "2024-04-26"},
          ...
        ]
      }

    - Every ORDER_CODE in orders.xlsx must appear exactly once in "orders".
    - A shipped order has is_delayed=false, a non-null ship_date (a valid send_date)
      and a non-null warehouse. A delayed order has is_delayed=true and null
      ship_date / warehouse.
    - "transfers" may be an empty list if the plan needs none.
    - The evaluator INDEPENDENTLY recomputes delayed_count, transfer_cost and all
      10 hard constraints from the raw data; any self-reported "metrics" block is
      ignored. If the agent produced CSVs (shipping_plan.csv / transfer_plan.csv)
      instead, convert them into the JSON shape above:
        ship rows -> orders[] (is_delayed=false), missing orders -> delayed,
        transfer rows -> transfers[]. Ship-date/warehouse columns map to
        ship_date/warehouse. Do NOT fabricate ship dates for delayed orders.

Multi-warehouse shipping scheduling evaluator (AgentCO-Bench interface)
======================================================================
Objective is LEXICOGRAPHIC:
    (1) minimize delayed_count   (primary)
    (2) minimize transfer_cost   (secondary tie-break)
where transfer_cost = sum(wms_transfer_qty)/2 + sum(aps_transfer_qty).

Final agreed business rules (independently enforced here):
  - WMS negative balances are cleaned to 0 (pre-committed / already shipped).
  - Factory (工厂库) usable supply = WMS(neg->0) + APS(pp_time < ship_date, STRICT).
  - External (外租库) usable supply = WMS(neg->0) + transfers_in (no APS).
  - Transfers factory->external only, <= db_limit(5) batches per machine_id.
  - JSJ (yuncang_type=='JSJ') orders must ship from an external warehouse.
  - Daily warehouse shipping <= limit_num_day (duplicate WHID_004 rows merged = 13000).
  - Delayed = order NOT shipped on a valid send_date <= its LATEST_SHIP_TIMESTAMP.

Ten hard constraints (any violation -> validity=0, quality=0):
  C1  ship_date in planning window (valid send_date)
  C2  JSJ order must ship from external warehouse
  C3  daily warehouse shipping <= limit_num_day
  C4  transfer direction factory->external only
  C5  transfer batches per machine_id <= db_limit
  C6  per-(warehouse,material) inventory conservation, non-negative at every prefix day
  C7  external warehouse ships only from WMS + transfers_in (no APS)
  C8  禁调品名 materials never transferred (none in this dataset -> auto-satisfied)
  C9  every shipped order's ship_date <= its LATEST_SHIP_TIMESTAMP (no late shipments)
  C10 成套: all rows of one ORDER_CODE ship-together-or-delay-together
      (each ORDER_CODE is a single row here -> auto-satisfied)

quality (no min(,1) truncation):
    absolute_score = p0 + tie_break, where
        p0        = 1 - delayed_count / total_orders        (primary, in [0,1])
        tie_break = quantized (1 - transfer_cost / transfer_upper), scaled strictly
                    below one unit of p0 so it only orders solutions with EQUAL
                    delayed_count.
    quality        = absolute_score / reference_value       (baseline self-ratio = 1.0)
"""

import argparse
import json
import os
import math
import traceback
from collections import defaultdict
from pathlib import Path

import pandas as pd

PLAN_FILE = "solution.json"
_HERE = Path(__file__).resolve().parent
_DATA = _HERE.parent / "data"

DB_LIMIT = None   # 由 config.xlsx 的 db_limit 决定，见 load_data
OVERDUE_STOCK_DAYS = None


# 禁调品名关键词。本数据集里 MATERIAL_NAME 无一命中，故 C8 恒为自动满足。
RESTRICTED_NAME = "烟管"

def load_data(data_dir):
    data_dir = Path(data_dir)
    hlmp = pd.read_excel(data_dir / "orders.xlsx")
    wms = pd.read_excel(data_dir / "inventory.xlsx")
    wh = pd.read_excel(data_dir / "warehouses.xlsx")
    aps = pd.read_excel(data_dir / "production.xlsx")
    send = pd.read_excel(data_dir / "shipping_dates.xlsx")
    conf = pd.read_excel(data_dir / "config.xlsx")

    # 调拨批次上限从配置表读取，不硬编码
    global DB_LIMIT, OVERDUE_STOCK_DAYS
    DB_LIMIT = int(conf["db_limit"].iloc[0])
    OVERDUE_STOCK_DAYS = int(conf["overdue_stock"].iloc[0])

    # otypenew (business priority label)
    hlmp["otypenew"] = "np"
    hlmp.loc[hlmp.order_sign_code.astype(str).str.startswith("A"), "otypenew"] = "zc"
    hlmp.loc[hlmp.order_sign_code == "YCBH", "otypenew"] = "yc"
    hlmp.loc[hlmp.order_sign_code == "KX", "otypenew"] = "kx"
    # kp: ~ORDER_CODE.startswith('ZQ') & is_pay=='1' -> 0 rows in this dataset

    hlmp["deadline"] = pd.to_datetime(hlmp.LATEST_SHIP_TIMESTAMP, errors="coerce")

    dates = sorted(pd.to_datetime(send.send_date).dt.strftime("%Y-%m-%d"))
    day_idx = {d: i for i, d in enumerate(dates)}
    ndays = len(dates)

    # merge duplicate warehouse rows
    wh_type, wh_limit = {}, defaultdict(float)
    for _, r in wh.iterrows():
        wh_type[r.store_name] = r.store_type
        wh_limit[r.store_name] += float(r.limit_num_day)
    fac = {n for n, t in wh_type.items() if t == "工厂库"}
    ext = {n for n, t in wh_type.items() if t == "外租库"}

    # WMS neg -> 0
    wms_inv = defaultdict(float)
    overdue_inv = defaultdict(float)
    for _, r in wms.iterrows():
        qty = max(0.0, float(r.num))
        key = (r.store_name, r.machine_id)
        wms_inv[key] += qty
        if float(r.age_order) >= OVERDUE_STOCK_DAYS:
            overdue_inv[key] += qty

    # APS cumulative usable-by-day (factory only, pp_time < ship_date STRICT)
    aps = aps[aps.num > 0].copy()
    aps["pp_time"] = pd.to_datetime(aps.pp_time)
    aps_cum = defaultdict(lambda: [0.0] * ndays)
    for _, r in aps.iterrows():
        row = aps_cum[(r.factory_store, r.machine_id)]
        for i, d in enumerate(dates):
            if r.pp_time < pd.Timestamp(d):
                row[i] += float(r.num)

    jsj_codes = set(hlmp[hlmp.yuncang_type == "JSJ"].ORDER_CODE)
    smoke_mats = set(hlmp[hlmp.MATERIAL_NAME.astype(str).str.contains(RESTRICTED_NAME)].MATERIAL_CODE)

    # 订单优先级：数据 priority 列，数值越小越紧急（2=款先真单 … 6=整车/天猫）。
    # 缺失按最低优先级兜底。权重 = 最大优先级值 + 1 - priority，越紧急权重越大。
    _pri = pd.to_numeric(hlmp.get("priority"), errors="coerce")
    _pmax = float(_pri.max()) if _pri.notna().any() else 1.0
    hlmp = hlmp.assign(_prio_w=(_pmax + 1.0 - _pri.fillna(_pmax)))

    order_info = {
        r.ORDER_CODE: dict(MATERIAL_CODE=r.MATERIAL_CODE, QTY=float(r.QTY),
                           deadline=r.deadline, otypenew=r.otypenew,
                           prio_w=float(r._prio_w))
        for _, r in hlmp.iterrows()
    }
    # 2024-08-23 修:C3(日发量上限) 与 C6(库存守恒) 原先用选手自报的 o["QTY"]，
    # 而不是源数据 order_info 里的真值。实测把 2306 张可发订单全标已发、QTY 一律写 0，
    # 就能得到 validity=1.0、零 violation、quality=1.1222(正好是理论上限)；连字段省掉
    # 不写也一样触发(o.get("QTY", 0) 缺省为 0)。下面统一改成按 ORDER_CODE 查真值，
    # 并对自报值做一致性校验(C3B)。
    def _true_qty(o):
        oc = o.get("ORDER_CODE")
        info = order_info.get(oc)
        if info is not None:
            return float(info["QTY"])
        return float(o.get("QTY", 0) or 0)

    def _true_mat(o):
        info = order_info.get(o.get("ORDER_CODE"))
        return info["MATERIAL_CODE"] if info is not None else o.get("MATERIAL_CODE")

    # 成套 groups (ORDER_CODE -> row count); here each is 1
    order_rowcount = hlmp.groupby("ORDER_CODE").size().to_dict()

    return dict(hlmp=hlmp, wms_inv=wms_inv, overdue_inv=overdue_inv,
                aps_cum=aps_cum, dates=dates, day_idx=day_idx,
                ndays=ndays, fac=fac, ext=ext, wh_limit=wh_limit, jsj_codes=jsj_codes,
                smoke_mats=smoke_mats, order_info=order_info, order_rowcount=order_rowcount,
                true_qty=_true_qty, true_mat=_true_mat)


def load_baseline():
    """基线的 absolute_score（与选手用同一 _absolute_score 公式算出）。"""
    with open(_HERE / "baseline" / "reference_metrics.json") as f:
        j = json.load(f)
    return float(j["reference_value"])


# ─── 评分：两层归一化到 [0,1] + 量化 + 大进制字典序合成为单一绝对分 ──────────────
# 尺子全部从数据算出，不依赖 baseline 数值。
R_TB = 1e-6          # tie-break 量化档距（调拨尺子量级远大于现实调拨量，取细档保分辨率）
M_TB = 1000001       # 进位基 = 1/R_TB + 1


def compute_transfer_upper(data):
    """调拨成本的理论上界。调拨只能「工厂库 → 外租库」，且受库存守恒约束，
    因此可调拨总量 ≤ 工厂库全部可用供给 = WMS(负数归零) + 规划期内可用的 APS 产出。
    成本 = wms_qty/2 + aps_qty ≤ 该总量，故它是一个合法且纯数据推出的上界。"""
    fac = data["fac"]
    ndays = data["ndays"]
    wms_fac = sum(v for (w, _m), v in data["wms_inv"].items() if w in fac)
    aps_fac = sum(row[ndays - 1] for (w, _m), row in data["aps_cum"].items() if w in fac)
    return max(wms_fac + aps_fac, 1.0)


def _absolute_score(delayed_weighted, transfer_cost, weight_total, transfer_upper):
    """把「少延重要单(主)」与「少调拨(次)」合成为单一绝对分。

    p0 = 1 - delayed_weighted / weight_total   in [0,1]  主目标（按优先级加权的准时率）
    p1 = 1 - transfer_cost / transfer_upper    in [0,1]  次目标（越大越好）

    延单按订单优先级加权：priority 越小越紧急、权重越大，因此延掉一张重要单
    比延掉一张普通单扣得多，体现「货不够优先保重要的单」。

    S_PRIMARY = 1 / weight_total               主目标一个最小台阶
    tie_break = p1_q * 0.49 * S_PRIMARY / M_TB 全量 < 主目标一台阶 => 主目标严格支配
    absolute  = p0 + tie_break
    """
    n = max(float(weight_total), 1.0)
    up = max(float(transfer_upper), 1.0)

    p0 = max(0.0, min(1.0, 1.0 - float(delayed_weighted) / n))
    p1 = max(0.0, min(1.0, 1.0 - float(transfer_cost) / up))

    s_primary = 1.0 / n
    tb_scale = 0.49 * s_primary / M_TB
    p1_q = int(round(p1 / R_TB))
    return p0 + p1_q * tb_scale


def check_and_score(sub, data):
    """Return (violations:list, recomputed:dict)."""
    violations = []
    orders = sub.get("orders", [])
    transfers = sub.get("transfers", [])

    fac, ext = data["fac"], data["ext"]
    wh_limit = data["wh_limit"]
    valid_dates = set(data["dates"])
    day_idx = data["day_idx"]
    ndays = data["ndays"]
    wms_inv = data["wms_inv"]
    aps_cum = data["aps_cum"]
    jsj_codes = data["jsj_codes"]
    smoke_mats = data["smoke_mats"]
    order_info = data["order_info"]
    _true_qty = data["true_qty"]
    _true_mat = data["true_mat"]

    c0 = 0
    all_wh = fac | ext
    for o in orders:
        delayed = bool(o.get("is_delayed", False))
        if delayed:
            if o.get("ship_date") not in (None, "") or o.get("warehouse") not in (None, ""):
                c0 += 1
        elif o.get("warehouse") not in all_wh:
            c0 += 1
    valid_mats = {v["MATERIAL_CODE"] for v in order_info.values()}
    for t in transfers:
        try:
            q = float(t.get("qty"))
        except (TypeError, ValueError):
            c0 += 1
            continue
        if (not math.isfinite(q) or q <= 0 or t.get("trigger_date") not in valid_dates
                or t.get("transfer_type") not in {"wms", "aps"}
                or t.get("machine_id") not in valid_mats):
            c0 += 1
    if c0:
        violations.append(f"C0_字段合法性: {c0} 条订单或调拨记录不合法")

    # ---- coverage: every ORDER_CODE present exactly once ----
    seen = defaultdict(int)
    for o in orders:
        seen[o.get("ORDER_CODE")] += 1
    missing = [c for c in order_info if seen.get(c, 0) == 0]
    dup = [c for c, n in seen.items() if n > 1]
    unknown = [c for c in seen if c not in order_info]
    if missing:
        violations.append(f"COVERAGE: {len(missing)} orders missing (e.g. {missing[:3]})")
    if dup:
        violations.append(f"COVERAGE: {len(dup)} orders duplicated (e.g. {dup[:3]})")
    if unknown:
        violations.append(f"COVERAGE: {len(unknown)} unknown ORDER_CODE (e.g. {unknown[:3]})")

    shipped = [o for o in orders if not o.get("is_delayed", False)]

    # ---- C1: ship_date in window ----
    c1 = 0
    for o in shipped:
        sd = o.get("ship_date")
        if sd not in valid_dates:
            c1 += 1
    if c1:
        violations.append(f"C1_窗口: {c1} 条发货日不在规划窗口")

    # ---- C2: JSJ -> external ----
    c2 = 0
    for o in shipped:
        if o.get("ORDER_CODE") in jsj_codes and o.get("warehouse") not in ext:
            c2 += 1
    if c2:
        violations.append(f"C2_JSJ外租库: {c2} 条JSJ订单未从外租库发货")

    # ---- C3: daily shipping <= limit ----
    daily = defaultdict(float)
    for o in shipped:
        if o.get("ship_date") and o.get("warehouse"):
            daily[(o["warehouse"], o["ship_date"])] += _true_qty(o)
    # C3B: 自报 QTY 若填了就必须与源数据一致(允许省略不填)
    c3b = 0
    for o in shipped:
        if "QTY" not in o or o.get("QTY") in (None, ""):
            continue
        info = order_info.get(o.get("ORDER_CODE"))
        if info is None:
            continue
        try:
            if abs(float(o["QTY"]) - float(info["QTY"])) > 1e-3:
                c3b += 1
        except (TypeError, ValueError):
            c3b += 1
    if c3b:
        violations.append(f"C3B_自报数量与源数据不符: {c3b} 条")

    c3 = 0
    for (w, d), q in daily.items():
        if q > wh_limit.get(w, float("inf")) + 1e-3:
            c3 += 1
    if c3:
        violations.append(f"C3_日发量上限: {c3} 个(仓库,日)超限")

    # ---- C4: transfer direction factory->external ----
    c4 = 0
    for t in transfers:
        if t.get("from_warehouse") not in fac or t.get("to_warehouse") not in ext:
            c4 += 1
    if c4:
        violations.append(f"C4_调拨方向: {c4} 条调拨非工厂库→外租库")

    # ---- C5: transfer batches per machine_id <= db_limit ----
    batch = defaultdict(int)
    for t in transfers:
        batch[t.get("machine_id")] += 1
    c5 = sum(1 for v in batch.values() if v > DB_LIMIT)
    if c5:
        violations.append(f"C5_调拨批次: {c5} 种型号调拨批次>{DB_LIMIT}")

    # ---- C8: 禁调品名 never transferred ----
    c8 = sum(1 for t in transfers if t.get("machine_id") in smoke_mats)
    if c8:
        violations.append(f"C8_禁调品名: {c8} 条禁调型号调拨")

    # ---- C9（修复）: transfer_type 标签必须有物理来源 ----
    # 原实现不校验标签：标 wms 成本减半，但工厂 WMS 里可能根本没有该型号现货。
    # 约束：同一 (工厂库, 型号) 的 wms 调拨总量 ≤ 该库 WMS 期初现货；
    #       aps 调拨总量 ≤ 该库 APS 计划期末累计可用量。
    c9 = 0
    wms_lab = defaultdict(float)
    aps_lab = defaultdict(float)
    for t in transfers:
        k = (t.get("from_warehouse"), t.get("machine_id"))
        if t.get("transfer_type") == "wms":
            wms_lab[k] += float(t.get("qty", 0))
        elif t.get("transfer_type") == "aps":
            aps_lab[k] += float(t.get("qty", 0))
    for k, q in wms_lab.items():
        if q > wms_inv.get(k, 0.0) + 1e-6:
            c9 += 1
    for k, q in aps_lab.items():
        row = aps_cum.get(k)
        if q > (row[ndays - 1] if row else 0.0) + 1e-6:
            c9 += 1
    if c9:
        violations.append(f"C9_调拨类型: {c9} 个(仓库,型号)的 wms/aps 标签量超出对应来源的可用量")

    # ---- C9: no late shipment (ship_date <= deadline) ----
    c9 = 0
    for o in shipped:
        oi = order_info.get(o.get("ORDER_CODE"))
        if oi is None or o.get("ship_date") is None:
            continue
        try:
            if pd.notna(oi["deadline"]) and pd.Timestamp(o["ship_date"]) > oi["deadline"]:
                c9 += 1
            elif pd.isna(oi["deadline"]):
                c9 += 1
        except Exception:
            c9 += 1
    if c9:
        violations.append(f"C9_晚点: {c9} 条发货日晚于最晚发货日")

    # ---- C10: 成套 (single-row orders here -> auto ok; guard anyway) ----
    c10 = 0
    grp = defaultdict(lambda: [0, 0])  # ORDER_CODE -> [shipped, delayed]
    for o in orders:
        if o.get("is_delayed", False):
            grp[o.get("ORDER_CODE")][1] += 1
        else:
            grp[o.get("ORDER_CODE")][0] += 1
    for c, (s, dd) in grp.items():
        if s > 0 and dd > 0:
            c10 += 1
    if c10:
        violations.append(f"C10_成套: {c10} 个订单部分发部分延")

    # ---- C6 + C7: per-(warehouse,material) time-phased conservation ----
    # APS strictness (pp_time < ship_date) is already baked into aps_cum: aps_cum[i]
    # is exactly the APS usable for shipping on dates[i]. So each day we first make
    # that day's usable APS present, then process transfers_in / transfers_out / ships.
    # Within a day the arrival order is aps -> transfer_in -> transfer_out -> ship.
    ship_ev = defaultdict(lambda: defaultdict(float))   # (wh,mat) -> day_i -> qty
    tin_ev = defaultdict(lambda: defaultdict(float))
    tout_ev = defaultdict(lambda: defaultdict(float))
    for o in shipped:
        if o.get("ship_date") in day_idx and o.get("warehouse"):
            ship_ev[(o["warehouse"], _true_mat(o))][day_idx[o["ship_date"]]] += _true_qty(o)
    for t in transfers:
        di = day_idx.get(t.get("trigger_date"))
        if di is None:
            continue
        tout_ev[(t.get("from_warehouse"), t.get("machine_id"))][di] += float(t.get("qty", 0))
        tin_ev[(t.get("to_warehouse"), t.get("machine_id"))][di] += float(t.get("qty", 0))

    keys = set(ship_ev) | set(tin_ev) | set(tout_ev)
    for (w, m) in list(wms_inv.keys()):
        keys.add((w, m))
    c6 = 0
    c6_maxdef = 0.0
    for (w, m) in keys:
        bal = wms_inv.get((w, m), 0.0)
        is_fac = w in fac
        acum = aps_cum.get((w, m), [0.0] * ndays) if is_fac else [0.0] * ndays
        prev_aps = 0.0
        for i in range(ndays):
            # 1) APS usable on dates[i] arrives (delta over previous day, strict pp<date)
            bal += acum[i] - prev_aps
            prev_aps = acum[i]
            # 2) transfer_in  3) transfer_out  4) ship
            bal += tin_ev.get((w, m), {}).get(i, 0.0)
            bal -= tout_ev.get((w, m), {}).get(i, 0.0)
            bal -= ship_ev.get((w, m), {}).get(i, 0.0)
            if bal < -1e-3:
                c6 += 1
                c6_maxdef = max(c6_maxdef, -bal)
    if c6:
        violations.append(f"C6_库存守恒: {c6} 次(仓库,型号,日)负库存，最大赤字 {c6_maxdef:.1f}")

    # ---- C7: external warehouse WMS+transfer_in only (no APS) ----
    # Independent balance ignoring APS for external warehouses.
    c7 = 0
    ext_keys = {(w, m) for (w, m) in keys if w in ext}
    for (w, m) in ext_keys:
        bal = wms_inv.get((w, m), 0.0)
        for i in range(ndays):
            bal += tin_ev.get((w, m), {}).get(i, 0.0)
            bal -= tout_ev.get((w, m), {}).get(i, 0.0)   # external never sends, but guard
            bal -= ship_ev.get((w, m), {}).get(i, 0.0)
            if bal < -1e-3:
                c7 += 1
                break
    if c7:
        violations.append(f"C7_外租库无APS: {c7} 个(外租库,型号)在无APS下库存不足")

    c11 = 0
    overdue_inv = data["overdue_inv"]
    age_keys = set(overdue_inv) | set(tin_ev) | set(tout_ev) | set(ship_ev)
    for w, mat in age_keys:
        old_bal = overdue_inv.get((w, mat), 0.0)
        for i in range(ndays):
            incoming = tin_ev.get((w, mat), {}).get(i, 0.0)
            if incoming > 0 and old_bal > 1e-3:
                c11 += 1
            consumed = (tout_ev.get((w, mat), {}).get(i, 0.0)
                        + ship_ev.get((w, mat), {}).get(i, 0.0))
            old_bal = max(0.0, old_bal - consumed)
    if c11:
        violations.append(f"C11_库龄优先: {c11} 次超期库存未清完即调入同仓同型号")

    # ---- recompute objective (independent) ----
    delayed_count = sum(1 for o in orders if o.get("is_delayed", False))
    # 加权延单：按订单优先级加权（priority 越小越紧急、权重越大），
    # 体现用户「货不够优先保重要的单」的诉求。
    oi = data["order_info"]
    delayed_weighted = sum(
        float(oi.get(str(o.get("ORDER_CODE")), {}).get("prio_w", 1.0))
        for o in orders if o.get("is_delayed", False))
    wms_t = sum(float(t.get("qty", 0)) for t in transfers if t.get("transfer_type") == "wms")
    aps_t = sum(float(t.get("qty", 0)) for t in transfers if t.get("transfer_type") == "aps")
    transfer_cost = wms_t / 2.0 + aps_t

    recomputed = dict(
        delayed_count=delayed_count,
        delayed_weighted=round(delayed_weighted, 4),
        shipped_count=len(shipped),
        transfer_cost=round(transfer_cost, 2),
        total_transfer_qty=round(wms_t + aps_t, 2),
        constraint_counts=dict(C0=c0, C1=c1, C2=c2, C3=c3, C4=c4, C5=c5, C6=c6,
                               C7=c7, C8=c8, C9=c9, C10=c10, C11=c11),
    )
    return violations, recomputed


def evaluate(file_path, data_dir):
    metrics = {"validity_score": 0.0, "quality_score": 0.0, "overall_score": 0.0, "error_info": {}}
    try:
        data = load_data(data_dir)
        reference_value = load_baseline()

        if not os.path.exists(file_path):
            metrics["error_info"] = {"fatal": [f"File not found: {file_path}"]}
            return metrics
        with open(file_path, "r", encoding="utf-8") as f:
            sub = json.load(f)

        if not isinstance(sub, dict) or "orders" not in sub:
            metrics["error_info"] = {"schema": ["missing 'orders' array"]}
            return metrics

        violations, rec = check_and_score(sub, data)
        metrics.update(rec)
        metrics["reference_value"] = reference_value

        if violations:
            metrics["error_info"] = {"constraint": violations[:8]}
            metrics["validity_score"] = 0.0
            metrics["quality_score"] = 0.0
            metrics["overall_score"] = 0.0
            return metrics

        metrics["validity_score"] = 1.0

        player_delayed = rec["delayed_count"]
        player_transfer = rec["transfer_cost"]

        # 流派 A：两层归一化 + 大进制字典序合成为单一 absolute_score，再对基线的
        # absolute_score 做一次除法归一（尺子从数据算出，不依赖 baseline 数值）。
        total_orders = len(data["order_info"])
        transfer_upper = compute_transfer_upper(data)
        weight_total = sum(float(v.get("prio_w", 1.0)) for v in data["order_info"].values())
        absolute = _absolute_score(rec.get("delayed_weighted", player_delayed),
                                   player_transfer, weight_total, transfer_upper)
        quality = absolute / reference_value if reference_value > 0 else 0.0

        metrics["quality_score"] = round(quality, 8)
        metrics["overall_score"] = round(quality, 8)
        metrics["absolute_score"] = round(absolute, 12)
        metrics["total_orders"] = total_orders
        metrics["transfer_upper"] = round(transfer_upper, 2)

    except Exception as e:
        metrics["error_info"] = {"exception": str(e), "traceback": traceback.format_exc()}
    return metrics


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--submission-dir", type=Path, required=True)
    parser.add_argument("--data-dir", type=Path, default=None)
    args = parser.parse_args()
    data_dir = str(args.data_dir) if args.data_dir else str(_DATA)
    result = evaluate(str(args.submission_dir / PLAN_FILE), data_dir)
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
