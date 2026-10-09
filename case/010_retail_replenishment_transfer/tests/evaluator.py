"""
EXTRACTOR_SPEC:
  plan_file: solution.json
  required_columns: [days]
  notes: >
    The agent must produce solution.json shaped as:
      {
        "days": [
          {"date": "2025-03-13",
           "transfers": [{"from_type": "store|wh", "from_id": "309827", "to_store_id": "1000027",
                          "item_id": "101363341300144", "qty": 5.0}],
           "purchases": [{"supplier_id": "1091786101", "to_store_id": "1000027",
                          "item_id": "101363341300144", "qty": 3.0}]},
          ... one entry per date in 2025-03-13..2025-03-29 (17 days, missing dates ok = no
          activity that day) ...
        ]
      }
    - from_type "store" or "wh" (warehouse, e.g. "WH_LOCAL"). purchases always come from a
      supplier_id (external, unconstrained inventory).
    - qty must be numeric and non-negative.
    - The evaluator INDEPENDENTLY recomputes fulfillment/lateness/cost and all hard
      constraints from data/ (the 5 raw 匿名零售 Excel files); any self-reported summary
      the agent includes is ignored.
    - If the agent produced per-day CSVs or a flat transfer list instead, convert into
      the days[] shape above (group by date; infer from_type by looking up the from_id
      in store/warehouse identifiers found in the raw data).
    - Do NOT fabricate transfers/purchases the agent did not decide, and do NOT re-solve.
"""

from __future__ import annotations

import argparse
import json
import os
import traceback
from collections import defaultdict
from datetime import datetime, timedelta
from pathlib import Path

import numpy as np
import pandas as pd

PLAN_FILE = "solution.json"
_HERE = Path(__file__).resolve().parent
_DATA = _HERE.parent / "data"

FILE_MAIN = "master_data.xlsx"
FILE_INV = "inventory_data.xlsx"
FILE_WH = "warehouse_inventory.xlsx"
FILE_FEE = "shipping_fee_rules.xlsx"
FILE_FCST = "forecast_data.xlsx"

WH_ID = "WH_LOCAL"
PLANNING_START = datetime(2025, 3, 13)
NUM_DAYS = 17  # 2025-03-13 .. 2025-03-29, inclusive
PLANNING_DATES = [(PLANNING_START + timedelta(days=i)).strftime("%Y-%m-%d") for i in range(NUM_DAYS)]
DAY_INDEX = {d: i for i, d in enumerate(PLANNING_DATES)}

DEFAULT_LT_DAYS = 3       # mode of 仓到店LT WH_LEAD_TIME (195164/207435 = 94.1% of rows)
DEFAULT_WEIGHT_KG = 1.0   # per the sheet's own note: "如果重量为空,则默认取该货品重量值为1KG"
STORE_TO_STORE_LT_DAYS = 1  # same-city store-to-store transfer, no LT field in raw data

# ── data loading (independent recompute from raw Excel, never from agent output) ──────

def _int_id(v) -> str:
    """Stringify a possibly float64-stored large integer ID without introducing the
    precision loss that a bare str(float) would (e.g. str(1.0143542640826712e+16)
    would print in scientific notation and lose trailing digits)."""
    if isinstance(v, float):
        return str(int(round(v)))
    return str(v).strip()


def _read_lt_sheets(xlsx_path: Path) -> dict[tuple[str, str], int]:
    """(item_id, store_id) -> WH_LEAD_TIME days, merged from both duplicate-free sheets."""
    lt = {}
    # Both sheets carry a Chinese label row followed by the real English header row;
    # sheet 01's header row is row index 1, sheet 02's is also row index 1 (its row 0
    # is the Chinese label row, contrary to the bare "header=0" a naive read would use).
    # Sheet 01's ITEM ID and sheet 02's LOC ID are float64-typed in the raw file (same
    # class of precision risk documented for 本地仓 WH_LOCAL仓库库存's ITEM column); go through
    # _int_id rather than a bare str() to avoid losing trailing digits.
    df1 = pd.read_excel(xlsx_path, sheet_name="仓到店LT 配送时长 0313_01", header=1)
    for _, r in df1.iterrows():
        if pd.isna(r["ITEM ID"]) or pd.isna(r["LOC ID"]):
            continue  # a handful of rows (5 in this sheet) genuinely have a blank ITEM ID
        item = _int_id(r["ITEM ID"])
        store = _int_id(r["LOC ID"])
        lt[(item, store)] = int(r["WH_LEAD_TIME"])
    df2 = pd.read_excel(xlsx_path, sheet_name="仓到店LT 配送时长 0313_02", header=1)
    for _, r in df2.iterrows():
        if pd.isna(r["ITEM ID"]) or pd.isna(r["LOC ID"]):
            continue
        item = _int_id(r["ITEM ID"])
        store = _int_id(r["LOC ID"])
        lt[(item, store)] = int(r["WH_LEAD_TIME"])
    return lt


def _read_dsd(xlsx_path: Path) -> dict[tuple[str, str], str]:
    """(store_id, item_id) -> supplier_id for DSD-flagged pairs."""
    df = pd.read_excel(xlsx_path, sheet_name="DSD 主档信息 0310", header=1)
    dsd = {}
    for _, r in df.iterrows():
        store = _int_id(r["LOC"])
        item = _int_id(r["ITEM"])
        dsd[(store, item)] = _int_id(r["主供应商ID"])
    return dsd


def _read_weights(xlsx_path: Path) -> dict[str, float]:
    df = pd.read_excel(xlsx_path, sheet_name="商品供应商重量 体积 0310", header=2)
    weights = {}
    for _, r in df.iterrows():
        item = r.get("item ")
        if pd.isna(item):
            continue
        item = _int_id(item)
        w = r.get("WEIGHT")
        if pd.isna(w) or w == "":
            continue
        # keep the heaviest known weight per item if duplicated across suppliers
        w = float(w)
        if item not in weights or w > weights[item]:
            weights[item] = w
    return weights


def _read_moq_mov(xlsx_path: Path) -> dict[str, dict[str, float | None]]:  # noqa: ARG001
    """Placeholder — 主档 '区域仓MOQ+MOV 0310' sheet 存在但当前评估器不使用
    (非 DSD 供应商 MOQ/MOV 未纳入约束, 见 gt.json 结构性缺失字段说明)。
    保留 sheet 为客户原始数据的一部分；此函数已从数据加载链剔除, 未来若纳入
    MOQ/MOV 约束再恢复。"""
    return {}


def _read_shipping_fees(xlsx_path: Path) -> dict[tuple[str, str], tuple[float, float]]:
    """(origin_province, dest_province) -> (first_kg_fee, extra_kg_fee). Same-city rows
    (TC/同城) are keyed with origin==dest. If a province pair appears multiple times
    (different couriers), keep the cheapest first-kg fee (a reasonable, data-derived
    choice; the agent's own solver is free to pick differently for its own plan)."""
    df = pd.read_excel(xlsx_path, sheet_name="---基础快递费用---", header=0)
    cols = list(df.columns)
    c_type, c_from, c_to, c_first, c_extra = cols[1], cols[2], cols[3], cols[5], cols[6]
    fees = {}
    for _, r in df.iterrows():
        origin, dest = str(r[c_from]).strip(), str(r[c_to]).strip()
        first, extra = float(r[c_first]), float(r[c_extra])
        key = (origin, dest)
        if key not in fees or first < fees[key][0]:
            fees[key] = (first, extra)
    return fees


def _read_inventory(inv_xlsx: Path, wh_xlsx: Path) -> dict[tuple[str, str], float]:
    """(loc_id, item_id) -> available qty at planning day 0, per the formula given in
    库存数据 sheet '可用库存计算规则 0313': 店存 = 1+2+4-3-5-6-7
    (STOCK_ON_HAND + IN_TRANSIT_QTY + TSF_EXPECTED_QTY - TSF_RESERVED_QTY - RTV_QTY
     - NON_SELLABLE_QTY - CUSTOMER_RESV). Warehouse snapshot lacks IN_TRANSIT/EXPECTED
    columns in some rows; missing fields treated as 0."""
    avail: dict[tuple[str, str], float] = defaultdict(float)

    def _avail_row(r) -> float:
        soh = float(r.get("STOCK_ON_HAND", 0) or 0)
        transit = float(r.get("IN_TRANSIT_QTY", 0) or 0)
        expected = float(r.get("TSF_EXPECTED_QTY", 0) or 0)
        reserved = float(r.get("TSF_RESERVED_QTY", 0) or 0)
        rtv = float(r.get("RTV_QTY", 0) or 0)
        non_sellable = float(r.get("NON_SELLABLE_QTY", 0) or 0)
        cust_resv = float(r.get("CUSTOMER_RESV", 0) or 0)
        return soh + transit + expected - reserved - rtv - non_sellable - cust_resv

    for sheet in ("P1", "P2", "P3", "P4", "P5", "P6", "P7"):
        df = pd.read_excel(inv_xlsx, sheet_name=sheet, header=0)
        # P5's "ITEM ID" header has a trailing space unlike the other six sheets;
        # normalize all column names before indexing rather than hardcode per-sheet.
        df.columns = [str(c).strip() for c in df.columns]
        for _, r in df.iterrows():
            item = _int_id(r["ITEM ID"])
            loc = _int_id(r["LOC ID"])
            avail[(loc, item)] += _avail_row(r)

    wh_df = pd.read_excel(wh_xlsx, sheet_name="warehouse_inventory", header=0)
    for _, r in wh_df.iterrows():
        item = r.get("ITEM")
        if pd.isna(item):
            continue
        # ITEM column is float64 in the raw file; _int_id round-trips through int to
        # avoid 1.0147532e+15-style precision loss before stringifying.
        item = _int_id(item)
        avail[(WH_ID, item)] += _avail_row(r)

    return dict(avail)


def _read_forecast(fcst_xlsx: Path) -> dict[tuple[str, str, int], float]:
    """(store_id, item_id, day_index) -> forecast qty, merged across the 8 sheets that
    use inconsistent column names ("stroe "/"store", "item "/"item")."""
    demand: dict[tuple[str, str, int], float] = defaultdict(float)
    xl = pd.ExcelFile(fcst_xlsx)
    for sheet in xl.sheet_names:
        df = pd.read_excel(xl, sheet_name=sheet, header=0)
        df.columns = [str(c).strip() for c in df.columns]
        rename = {}
        for c in df.columns:
            if c in ("stroe", "store"):
                rename[c] = "store"
            elif c == "item":
                rename[c] = "item"
        df = df.rename(columns=rename)
        for _, r in df.iterrows():
            wd = r["work_date"]
            if pd.isna(wd):
                continue
            date_str = pd.Timestamp(wd).strftime("%Y-%m-%d")
            if date_str not in DAY_INDEX:
                continue
            store = _int_id(r["store"])
            item = _int_id(r["item"])
            qty = float(r["ai_forecast_qty"] or 0)
            demand[(store, item, DAY_INDEX[date_str])] += qty
    return dict(demand)


def _read_store_master(main_xlsx: Path) -> dict[str, dict]:
    df = pd.read_excel(main_xlsx, sheet_name="门店地址基础信息 0310", header=0)
    out = {}
    for _, r in df.iterrows():
        sid = r.get("店铺号ID")
        if pd.isna(sid):
            continue
        out[_int_id(sid)] = {
            "city": str(r.get("城市", "")).strip(),
            "district": str(r.get("区", "")).strip(),
            "store_type": str(r.get("店铺类型", "")).strip(),
        }
    return out


def load_data(data_dir: str | Path) -> dict:
    data_dir = Path(data_dir)
    main_xlsx = data_dir / FILE_MAIN
    inv_xlsx = data_dir / FILE_INV
    wh_xlsx = data_dir / FILE_WH
    fee_xlsx = data_dir / FILE_FEE
    fcst_xlsx = data_dir / FILE_FCST

    stores = _read_store_master(main_xlsx)
    dsd = _read_dsd(main_xlsx)
    lt = _read_lt_sheets(main_xlsx)
    weights = _read_weights(main_xlsx)
    fees = _read_shipping_fees(fee_xlsx)
    forecast = _read_forecast(fcst_xlsx)
    inventory0 = _read_inventory(inv_xlsx, wh_xlsx)

    optimization_stores = sorted({s for (s, _, _) in forecast})
    optimization_items = sorted({i for (_, i, _) in forecast})

    total_forecast_by_pair: dict[tuple[str, str], float] = defaultdict(float)
    for (s, i, _d), q in forecast.items():
        total_forecast_by_pair[(s, i)] += q

    return dict(
        stores=stores, dsd=dsd, lt=lt, weights=weights, fees=fees,
        forecast=forecast, inventory0=inventory0,
        optimization_stores=optimization_stores, optimization_items=optimization_items,
        total_forecast_by_pair=total_forecast_by_pair,
    )


def load_baseline() -> float:
    with open(_HERE / "baseline" / "reference_metrics.json", encoding="utf-8") as f:
        d = json.load(f)
    return float(d["reference_value"])


# ── 流派 A 评分：三层归一化到 [0,1] + 量化+大进制字典序合成为单一绝对分 ──────
NUM_DAYS_FOR_UPPER = 17          # 迟到理论上界的天数系数
COST_UPPER = 1e9                 # 成本保守上界(baseline≈2.4e7, 40× 缓冲)
R_TB = 0.001                     # tie-break 每层量化到 1000 档
M_TB = 1001                      # 进位基 = 1/R_TB + 1


def _absolute_score(fulfillment: float, late: float, cost: float, total_demand: float) -> float:
    """把三级字典序目标合成一个绝对分, 尺子全部从数据/常数算出, 不依赖 baseline。

    构造:
      p0 = fulfillment_rate                           ∈ [0, 1]
      p1 = 1 − late / (total_demand × NUM_DAYS)       ∈ [0, 1](越大越好)
      p2 = 1 − cost / COST_UPPER                      ∈ [0, 1](越大越好)

    合成为:
      S_PRIMARY = 1 / total_demand                    (P0 一个满足单位的最小台阶)
      TB_SCALE  = 0.49 × S_PRIMARY / (M_TB × M_TB)    (保证全 tie 块 < P0 一台阶)
      tie_break = (p1_q × M_TB + p2_q) × TB_SCALE
      absolute  = p0 + tie_break

    P0 一个台阶 = S_PRIMARY, 而 tie_break 全量最大 = 0.49×S_PRIMARY, 故 P0 严格支配;
    tie_break 内部 P1 用 M_TB 大进制乘子锁死、严格支配 P2。
    """
    td = max(float(total_demand), 1.0)
    late_upper = td * NUM_DAYS_FOR_UPPER

    p0 = max(0.0, min(1.0, float(fulfillment)))
    p1 = max(0.0, min(1.0, 1.0 - float(late) / late_upper))
    p2 = max(0.0, min(1.0, 1.0 - float(cost) / COST_UPPER))

    s_primary = 1.0 / td
    tb_scale = 0.49 * s_primary / (M_TB * M_TB)

    p1_q = int(round(p1 / R_TB))
    p2_q = int(round(p2 / R_TB))
    tie_break = (p1_q * M_TB + p2_q) * tb_scale

    return p0 + tie_break


def get_lt_days(data: dict, item: str, store: str, diag: dict) -> int:
    v = data["lt"].get((item, store))
    if v is None:
        diag["lt_defaults_used"] += 1
        return DEFAULT_LT_DAYS
    return v


def get_weight_kg(data: dict, item: str) -> float:
    return data["weights"].get(item, DEFAULT_WEIGHT_KG)


def shipping_cost(data: dict, from_type: str, from_id: str, to_store: str, item: str, qty: float) -> float:
    """Rounded-up first-kg + extra-kg fee from the raw 快递费用 rate table. All stores
    and the warehouse are in the anonymized local region, so same-city (REGION_LOCAL,REGION_LOCAL) rate applies unless
    the source is an out-of-province supplier (then look up supplier's declared city
    -> best-effort province match; falls back to same-city rate if unresolved, since
    supplier province/city normalization is a structural data gap, see information.md)."""
    weight = get_weight_kg(data, item) * qty
    origin = "REGION_LOCAL"  # all internal sources (stores, wh WH_LOCAL) are in the anonymized local region
    dest = "REGION_LOCAL"
    first, extra = data["fees"].get((origin, dest), (8.8, 1.3))
    if weight <= 1.0:
        return float(np.ceil(first))
    return float(np.ceil(first + (weight - 1.0) * extra))


def _simulate(data: dict, submission: dict, diag: dict) -> tuple[list[str], dict]:
    """Roll forward day-by-day: apply arrivals scheduled for today, then deduct today's
    outbound transfers/purchases, enforcing every hard constraint independently. Returns
    (violations, recomputed_metrics)."""
    violations: list[str] = []

    balance: dict[tuple[str, str], float] = defaultdict(float)
    for (loc, item), qty in data["inventory0"].items():
        balance[(loc, item)] = qty

    pending_arrivals: dict[int, list[tuple[str, str, float]]] = defaultdict(list)  # day -> [(loc,item,qty)]
    satisfied: dict[tuple[str, str, int], float] = defaultdict(float)
    late_units = 0.0
    total_cost = 0.0
    dsd_hits = 0

    days_by_date = {d["date"]: d for d in submission.get("days", []) if isinstance(d, dict)}

    for day_i, date in enumerate(PLANNING_DATES):
        for (loc, item, qty) in pending_arrivals.get(day_i, []):
            balance[(loc, item)] += qty

        # Natural store demand consumes stock on the day it occurs, before today's
        # transfer/purchase decisions are checked against the resulting balance. This
        # is what makes balance a real day-by-day inventory rollforward instead of an
        # arrivals-only ledger that only grows — without it, every (store,item) that
        # ever receives a transfer inflates indefinitely and both the capacity
        # weak-check and the source-inventory check would be checking a fictitious,
        # ever-rising balance rather than what the store would actually be holding.
        #
        # "Lateness" (P1) is defined here as accumulated unmet demand persisting past
        # the day it was due — i.e. shortage_qty × days_still_unmet, mirroring
        # multi_warehouse_shipping_3097's late_penalty concept of "quantity that missed
        # its due date". A transfer dispatched the same day demand occurs and arriving
        # after STORE_TO_STORE_LT_DAYS/DEFAULT_LT_DAYS therefore is NOT automatically
        # "late" — it is late only if the (store,item) shortage from that day is still
        # unresolved on a later day, at which point it starts accruing. This avoids the
        # structural mismatch of comparing a transfer's arrival day directly against
        # the demand day it happens to be closest to (LT>=1 makes same-day dispatch
        # always arrive after the demand day, which said nothing about real delay).
        for (store, item, d) in list(data["forecast"].keys()):
            if d != day_i:
                continue
            demand = data["forecast"][(store, item, d)]
            avail = max(0.0, balance.get((store, item), 0.0))
            sat = min(avail, demand)
            satisfied[(store, item, d)] = sat
            balance[(store, item)] = balance.get((store, item), 0.0) - sat

        day = days_by_date.get(date, {})
        transfers = day.get("transfers", []) or []
        purchases = day.get("purchases", []) or []

        outbound_today: dict[tuple[str, str], float] = defaultdict(float)

        for t in transfers:
            from_type = t.get("from_type")
            from_id = str(t.get("from_id", "")).strip()
            to_store = str(t.get("to_store_id", "")).strip()
            item = str(t.get("item_id", "")).strip()
            qty = float(t.get("qty", 0) or 0)
            if qty <= 0:
                continue
            if from_type not in ("store", "wh"):
                violations.append(f"HC0_来源类型: {date} from_type={from_type!r} 非法")
                continue
            # 修复：时效按声明的 from_type 取（:423），库存却按 from_id 扣（:468），
            # 把仓库 9961 声明成 store 就能动用仓库库存却拿 1 天店到店时效。
            # 声明类型必须与 ID 的真实身份一致。
            if from_type == "wh" and from_id != WH_ID:
                violations.append(f"HC0_来源类型: {date} from_id={from_id} 不是仓库，却声明 from_type=wh")
                continue
            if from_type == "store" and (from_id == WH_ID or from_id not in data["stores"]):
                violations.append(f"HC0_来源类型: {date} from_id={from_id} 不是门店，却声明 from_type=store")
                continue

            outbound_today[(from_id, item)] += qty

            if (to_store, item) in data["dsd"]:
                dsd_hits += 1
                required_supplier = data["dsd"][(to_store, item)]
                if from_type != "wh" and from_id != required_supplier:
                    violations.append(
                        f"HC4_DSD: {date} 门店{to_store}商品{item}为DSD商品，"
                        f"必须来自本地仓库或指定供应商{required_supplier}，实际来源{from_type}:{from_id}"
                    )

            lt = get_lt_days(data, item, to_store, diag) if from_type == "wh" else STORE_TO_STORE_LT_DAYS
            arrival_day = day_i + lt
            if arrival_day < NUM_DAYS:
                pending_arrivals[arrival_day].append((to_store, item, qty))

            total_cost += shipping_cost(data, from_type, from_id, to_store, item, qty)

        for p in purchases:
            supplier_id = str(p.get("supplier_id", "")).strip()
            to_store = str(p.get("to_store_id", "")).strip()
            item = str(p.get("item_id", "")).strip()
            qty = float(p.get("qty", 0) or 0)
            if qty <= 0:
                continue

            if (to_store, item) in data["dsd"]:
                dsd_hits += 1
                required_supplier = data["dsd"][(to_store, item)]
                if supplier_id != required_supplier:
                    violations.append(
                        f"HC4_DSD: {date} 门店{to_store}商品{item}为DSD商品，"
                        f"必须来自指定供应商{required_supplier}，实际来自{supplier_id}"
                    )

            same_city_store_stock = sum(
                max(0.0, balance.get((s, item), 0.0) - outbound_today.get((s, item), 0.0))
                for s in data["optimization_stores"] if s != to_store
            )
            same_city_wh_stock = max(0.0, balance.get((WH_ID, item), 0.0) - outbound_today.get((WH_ID, item), 0.0))
            if same_city_store_stock + same_city_wh_stock > 1e-6:
                violations.append(
                    f"HC2_HC3_内部优先同城穷尽: {date} 门店{to_store}商品{item}向供应商{supplier_id}补货"
                    f"{qty}，但同城仍有可用库存(门店{same_city_store_stock:.1f}+仓库{same_city_wh_stock:.1f})未用完"
                )

            lt = STORE_TO_STORE_LT_DAYS + 6  # supplier lead time unknown from data for
            # non-DSD purchases (only DSD rows carry PICKUP_LEAD_TIME); use a
            # conservative placeholder longer than internal LT so purchases are never
            # rewarded with unrealistically fast arrival. See information.md.
            arrival_day = day_i + lt
            if arrival_day < NUM_DAYS:
                pending_arrivals[arrival_day].append((to_store, item, qty))

            total_cost += shipping_cost(data, "supplier", supplier_id, to_store, item, qty)

        for (loc, item), qty in outbound_today.items():
            if qty > balance.get((loc, item), 0.0) + 1e-6:
                violations.append(
                    f"HC1_来源库存超用: {date} {loc}商品{item} 调出{qty:.1f} > 可用{balance.get((loc, item), 0.0):.1f}"
                )
            balance[(loc, item)] = balance.get((loc, item), 0.0) - qty

        # accumulate lateness: any (store,item) demand from today or an earlier day
        # that is still unmet as of today keeps costing one more day of delay. This
        # is computed AFTER this day's transfers/purchases above, using
        # the same `satisfied` dict already populated for every earlier day_i.
        for (store, item, d) in list(data["forecast"].keys()):
            if d > day_i:
                continue
            demand = data["forecast"][(store, item, d)]
            sat = satisfied.get((store, item, d), 0.0)
            shortfall = demand - sat
            if shortfall > 1e-9:
                late_units += shortfall

    total_demand = sum(data["forecast"].values())
    total_satisfied = sum(satisfied.values())
    fulfillment_rate = total_satisfied / total_demand if total_demand > 0 else 0.0

    recomputed = dict(
        fulfillment_rate=round(fulfillment_rate, 8),
        total_satisfied=round(total_satisfied, 4),
        total_demand=round(total_demand, 4),
        late_units=round(late_units, 4),
        total_cost=round(total_cost, 2),
        dsd_constraint_checked_pairs=dsd_hits,
        lt_defaults_used=diag["lt_defaults_used"],
    )
    return violations, recomputed


def evaluate(submission_dir: str | Path, data_dir: str | Path) -> dict:
    metrics = {"validity_score": 0.0, "quality_score": 0.0, "overall_score": 0.0, "error_info": {}}
    try:
        data = load_data(data_dir)
        baseline = load_baseline()

        sol_path = Path(submission_dir) / PLAN_FILE
        if not sol_path.exists():
            metrics["error_info"] = {"fatal": [f"缺 {PLAN_FILE}"]}
            return metrics
        with open(sol_path, "r", encoding="utf-8") as f:
            submission = json.load(f)

        if not isinstance(submission, dict) or "days" not in submission:
            metrics["error_info"] = {"schema": ["缺少 'days' 数组"]}
            return metrics

        diag = {"lt_defaults_used": 0}
        violations, rec = _simulate(data, submission, diag)
        metrics.update(rec)

        if violations:
            metrics["error_info"] = {"constraint": violations[:8]}
            metrics["violation_count"] = len(violations)
            return metrics

        metrics["validity_score"] = 1.0

        player_fulfillment = rec["fulfillment_rate"]
        player_late = rec["late_units"]
        player_cost = rec["total_cost"]

        # 流派 A：绝对分合成 + 一次除法归一(尺子全部从数据/常数算, 不依赖 baseline 数值)
        total_demand = max(rec["total_demand"], 1.0)
        player_absolute = _absolute_score(player_fulfillment, player_late, player_cost, total_demand)
        reference_value = baseline  # 已是 baseline 用同一公式算出的 absolute_score

        quality = player_absolute / reference_value if reference_value > 0 else 0.0

        metrics["quality_score"] = round(quality, 8)
        metrics["overall_score"] = round(quality, 8)
        metrics["player_absolute_score"] = round(player_absolute, 8)
        metrics["reference_value"] = reference_value

    except Exception as e:
        metrics["error_info"] = {"exception": str(e), "traceback": traceback.format_exc()}
    return metrics


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--submission-dir", type=Path, required=True)
    parser.add_argument("--data-dir", type=Path, default=None)
    args = parser.parse_args()
    data_dir = args.data_dir if args.data_dir else _DATA
    result = evaluate(args.submission_dir, data_dir)
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
