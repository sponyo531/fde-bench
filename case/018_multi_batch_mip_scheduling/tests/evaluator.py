"""
EXTRACTOR_SPEC:
  plan_file: solution.json
  required_columns: [source_ids, batches]
  notes: >
    The agent must produce a single JSON file `solution.json` describing the
    production schedule for ALL 5 batches (sourceId). Expected format:
      {
        "source_ids": ["BATCH-24A-01", "BATCH-24A-02", "BATCH-24A-03",
                       "BATCH-24A-04", "BATCH-24B-01"],
        "batches": {
          "<sourceId>": {
            "schedule": [
              {
                "工单编码": <int>,            // work order id (from 订单列表.工单编码)
                "产品编码": "<str>",
                "线体编码": "<str or null>",  // production line; null when 是否未排=Y
                "订单编码": "<str>",
                "系列编码": "<str>",
                "要求交付日期": "YYYY-MM-DD",
                "班次": "day",
                "排产日期": "YYYY-MM-DD",     // '2999-01-01' when unscheduled
                "排产数量": <int>,
                "是否未排": "Y|N",            // N = scheduled (counted), Y = unscheduled
                "行号": <float or null>,
                "跟随行号": <float or null>,
                "生产节拍": <int>,
                "线体优先级": <int or null>
              }, ...
            ]
          }, ...
        }
      }

    The evaluator recomputes ALL metrics and constraints independently from
    data/scheduling_data.xlsx (takt, line priority, daily capacity, workdays,
    order qty are all read from the Excel, NOT trusted from the solution rows).

    Common agent output patterns to handle / normalize into solution.json:
    - Per-batch CSV files (e.g. result_BATCH-24A-01.csv ... with columns
      工单编码,产品编码,线体编码,订单编码,系列编码,要求交付日期,班次,排产日期,排产数量,
      是否未排,行号,跟随行号) -> collect each batch's rows into batches[sid].schedule.
    - A row is "scheduled" when 是否未排 == "N"; unscheduled rows have
      是否未排=="Y" and 排产日期=="2999-01-01".
    - 工单编码 may be int or a string like "123_余" for a split-remainder row; keep as-is.
    - 排产日期 / 要求交付日期 may carry a time component -> truncate to 10 chars (YYYY-MM-DD).
    Write solution.json to the output directory.
"""

import argparse
import json
import os
import traceback
from pathlib import Path
from collections import defaultdict

import numpy as np

PLAN_FILE = "solution.json"
_HERE = os.path.dirname(os.path.abspath(__file__))
_DATA = os.path.join(_HERE, "..", "data")
DATA_FILE_NAME = "scheduling_data.xlsx"

SOURCE_IDS = [
    "BATCH-24A-01",
    "BATCH-24A-02",
    "BATCH-24A-03",
    "BATCH-24A-04",
    "BATCH-24B-01",
]

# 评分组合：主目标比值 + 递归有界 tie-break（消除大权重）。
# 用户明确要求"优先级从高到低"(少未排 > 交期 > 整单同天 > 线体适配 > 系列集中 > 产能均衡),
# 且强调"不要因下面的集中等目标导致未排"——即严格字典序,高优先级不可被低优先级累积翻盘。
#
# 组合方式（与 multi_warehouse_shipping 干净样板同构：primary + bounded tie_break）：
#   主目标 = schedule_rate（排产率, [0,1], 压倒性主目标, 连续值不量化）。
#   下面 5 层(on_time/same_day/line_prio/series/balance)做成【递归有界 tie-break】：
#     - 给每层子分一个分辨率台阶 R_TB(=0.02)，量化成整数位 n_k∈[0, 1/R_TB]；
#     - 逐层嵌套成一个大进制定点数 V = n1·M^4 + n2·M^3 + n3·M^2 + n4·M + n5
#       (M = 1/R_TB + 1，保证每层"进位"严格压过其下所有层之和——即严格字典序)；
#     - 整个 tie-break 块再乘一个极小有界系数 TB_SCALE，使【全块最大值 < 主目标一个
#       最小台阶 S_PRIMARY 的 0.49 倍】。S_PRIMARY = 1/最大需求量(schedule_rate 的最小
#       非零步长)。故"主目标改善一个台阶 > tie-break 全部拉满"，schedule_rate 高一档的解
#       无论下面 5 层多差，batch_score 必更高；层内同理逐层不可被下层翻盘。
#   所有系数均为极小定点小数（无 1e5 大权重），且最小层台阶(~6e-13)远高于 float64 分辨率，
#   不会数值下溢。
R_TB       = 0.02                       # tie-break 每层子分的量化台阶(50 档)
M_TB       = int(round(1.0 / R_TB)) + 1 # 定点进制基(每层数字范围 [0, 1/R_TB])
_V_MAX     = (M_TB - 1) * (M_TB**4 + M_TB**3 + M_TB**2 + M_TB + 1)  # tie-break 整数上界
S_PRIMARY  = 1.0 / 2452.0               # schedule_rate 最小非零步长(=1/最大批次需求量)
TB_SCALE   = 0.49 * S_PRIMARY / _V_MAX  # 全块 < 0.49×主目标一个台阶 ⇒ 严格字典序


# ─── 数据加载（从原始 Excel 独立加载，不信任 solution 自报值）──────────────────
def load_reference_data(data_dir):
    import pandas as pd
    xl_path = os.path.join(data_dir, DATA_FILE_NAME)
    xl = pd.ExcelFile(xl_path)
    orders_df   = pd.read_excel(xl, "订单列表")
    products_df = pd.read_excel(xl, "产品列表")
    lines_df    = pd.read_excel(xl, "线体信息")
    calendar_df = pd.read_excel(xl, "工作日历")
    orders_df["要求交付日期"] = pd.to_datetime(orders_df["要求交付日期"]).dt.strftime("%Y-%m-%d")
    calendar_df["工作日期"]   = pd.to_datetime(calendar_df["工作日期"]).dt.strftime("%Y-%m-%d")
    return orders_df, products_df, lines_df, calendar_df


def load_baseline():
    with open(os.path.join(_HERE, "baseline", "reference_metrics.json")) as f:
        return float((lambda _d:_d.get("reference_value",_d.get("baseline_cost")))(json.load(f)))


def _is_nan(v):
    try:
        return v is None or (isinstance(v, float) and np.isnan(v))
    except Exception:
        return False


# ─── 硬约束验证（6条,独立重算）────────────────────────────────────────────────
def check_hard_constraints(schedule_rows, orders_df, products_df, lines_df, calendar_df, sid):
    violations = []

    p = products_df[products_df["sourceId"] == sid]
    l = lines_df[lines_df["sourceId"] == sid]
    c = calendar_df[calendar_df["sourceId"] == sid]
    o = orders_df[orders_df["sourceId"] == sid]

    valid_prod_line = set(zip(p["产品编码"].astype(str), p["线体编码"].astype(str)))
    valid_workdays  = set(zip(c["线体编码_排班"].astype(str), c["工作日期"].astype(str)))
    line_cap_map    = {str(k): float(v) for k, v in zip(l["线体编码"], l["日产能"])}
    order_qty_map   = {int(k): int(v) for k, v in zip(o["工单编码"].astype(int), o["订单数量"].astype(int))}

    takt_ref = {}  # (产品编码, 线体编码) -> takt，从产品列表独立读取
    for _, row in p.iterrows():
        takt_ref[(str(row["产品编码"]), str(row["线体编码"]))] = int(row["生产节拍"])
    zero_takt_prods = {str(row["产品编码"]) for _, row in p.iterrows() if int(row["生产节拍"]) == 0}

    scheduled = [r for r in schedule_rows if r.get("是否未排", "Y") == "N"]
    all_rows  = schedule_rows

    # 约束1：线体可用性（节拍=0跟随行例外）
    for r in scheduled:
        is_follow = not _is_nan(r.get("跟随行号"))
        prod_code = str(r.get("产品编码", ""))
        if is_follow and prod_code in zero_takt_prods:
            continue
        key = (prod_code, str(r.get("线体编码", "")))
        if key not in valid_prod_line:
            violations.append(f"[线体可用性] 工单{r.get('工单编码')}: 产品{prod_code} 不可在线体{r.get('线体编码')}生产")

    # 约束2：工作日历
    for r in scheduled:
        date_str = str(r.get("排产日期", ""))[:10]
        line_str = str(r.get("线体编码", ""))
        if (line_str, date_str) not in valid_workdays:
            is_follow = not _is_nan(r.get("跟随行号"))
            prod_code = str(r.get("产品编码", ""))
            if is_follow and prod_code in zero_takt_prods:
                continue
            violations.append(f"[工作日历] 工单{r.get('工单编码')}: 线体{line_str} 在{date_str}无排班")

    # 约束3：产能上限≤110%（节拍从产品列表独立读取）
    cap_usage = defaultdict(float)
    for r in scheduled:
        prod_code = str(r.get("产品编码", ""))
        line_code = str(r.get("线体编码", ""))
        takt = takt_ref.get((prod_code, line_code), 0)
        if takt > 0:
            date_str = str(r.get("排产日期", ""))[:10]
            cap_usage[(line_code, date_str)] += int(r.get("排产数量", 0)) * takt
    for (lc, dt), usage in cap_usage.items():
        cap = line_cap_map.get(lc, 0)
        # 修复：原实现 cap<=0（线体无产能记录或产能为 0）时直接跳过检查，等于无限产能。
        # 线体不在产能表里 / 产能为 0 却排了产，按违规处理。
        if cap <= 0:
            if usage > 1e-6:
                violations.append(f"[产能超限] 线体{lc} {dt}: 线体无有效日产能({cap})却排产 {usage:.1f}")
            continue
        if usage > cap * 1.10 + 1e-6:
            violations.append(f"[产能超限] 线体{lc} {dt}: 使用{usage:.1f} > 上限{cap*1.10:.1f}")

    # 约束4：不超量排产（跳过 _余 拆单记录）
    wo_usage = defaultdict(int)
    for r in scheduled:
        wo_str = str(r.get("工单编码", ""))
        if "_余" in wo_str:
            continue
        try:
            wo_code = int(wo_str)
        except ValueError:
            continue
        wo_usage[wo_code] += int(r.get("排产数量", 0))
    for wo_code, used in wo_usage.items():
        max_qty = order_qty_map.get(wo_code, 0)
        if used > max_qty:
            violations.append(f"[超量排产] 工单{wo_code}: 排产{used} > 订单量{max_qty}")

    # 约束5：排产量非负
    for r in scheduled:
        qty = int(r.get("排产数量", 0))
        if qty < 0:
            violations.append(f"[负排产量] 工单{r.get('工单编码')}: 排产量={qty}")

    # 约束6：跟随约束（同天同线体）
    row_sched_map = {}
    for r in all_rows:
        row_no = r.get("行号")
        if not _is_nan(row_no):
            key = (str(r.get("订单编码", "")), float(row_no))
            row_sched_map[key] = (
                str(r.get("排产日期", ""))[:10],
                str(r.get("线体编码", "")),
                r.get("是否未排", "Y"),
            )
    # 跟随关系一律取订单列表的权威值，不读选手自报——原先 r.get("跟随行号") 让
    # 「漏填字段」变成了得分策略：整列省略 ⇒ 每行都 _is_nan ⇒ 零违规 ⇒ 满分，
    # 而老实补全字段的模型反倒被判 98 处违规判 0。同一份数据不该因为选手少写一
    # 个字段就换一套约束。口径与第 252 行「需求量从订单列表独立读取,不信任
    # solution 自报」保持一致。
    # 用「工单编码」而不是「行号」做主键。行号在 schema 里是 required=False 的
    # identifier，agent 的产物里本来就没有——抽取器可以选择从 data/ 的订单表 join
    # 进来，也可以不 join，两种都合规。但评估器原先靠行号定位，于是同一份 agent
    # 产物，join 了的判出 3 条违规、没 join 的零违规拿满分：一条硬约束的成立与否
    # 取决于抽取器做没做一个可选动作。工单编码在该批次 426 个值全唯一（比
    # (订单编码,行号) 的 405 组还细），且三个抽取器都保留了它。
    auth_follow = {}          # 工单编码 -> 被跟随的工单编码
    _row_to_wo = {}
    for _, _o in o.iterrows():
        _rn, _oc = _o.get("行号"), str(_o.get("订单编码", ""))
        if not _is_nan(_rn):
            _row_to_wo[(_oc, float(_rn))] = str(_o.get("工单编码", ""))
    for _, _o in o.iterrows():
        _f, _oc = _o.get("跟随行号"), str(_o.get("订单编码", ""))
        if _is_nan(_f):
            continue
        _target = _row_to_wo.get((_oc, float(_f)))
        if _target:
            auth_follow[str(_o.get("工单编码", ""))] = _target

    _wo_sched = {}            # 工单编码 -> (排产日期, 线体编码, 是否未排)
    for r in all_rows:
        _wo = str(r.get("工单编码", ""))
        if _wo:
            _wo_sched[_wo] = (str(r.get("排产日期", ""))[:10],
                              str(r.get("线体编码", "")),
                              r.get("是否未排", "Y"))

    for r in all_rows:
        order_code = str(r.get("订单编码", ""))
        follow = auth_follow.get(str(r.get("工单编码", "")))
        if not follow:
            continue
        follow_scheduled = r.get("是否未排", "Y") == "N"
        if follow not in _wo_sched:
            if follow_scheduled:
                violations.append(f"[跟随约束] 工单{r.get('工单编码')}: 被跟随工单{follow}在solution中找不到,但跟随行已排")
            continue
        target_date, target_line, target_is_unsched = _wo_sched[follow]
        if target_is_unsched == "Y" and follow_scheduled:
            violations.append(f"[跟随约束] 工单{r.get('工单编码')}: 被跟随行{follow}未排,跟随行不应排产")
        if follow_scheduled and target_is_unsched == "N":
            my_date = str(r.get("排产日期", ""))[:10]
            my_line = str(r.get("线体编码", ""))
            if my_date != target_date or my_line != target_line:
                violations.append(
                    f"[跟随约束] 工单{r.get('工单编码')}: 排在{my_date}/{my_line},"
                    f"被跟随行{follow}排在{target_date}/{target_line}"
                )

    return violations, len(violations)


# ─── 质量指标计算（6项加权,独立重算)──────────────────────────────────────────
def compute_quality(schedule_rows, orders_df, products_df, lines_df, calendar_df, sid):
    p = products_df[products_df["sourceId"] == sid]
    l = lines_df[lines_df["sourceId"] == sid]
    o = orders_df[orders_df["sourceId"] == sid]

    prod_prio_map = {}
    takt_ref      = {}
    max_prio      = 1
    for _, row in p.iterrows():
        k = (str(row["产品编码"]), str(row["线体编码"]))
        prod_prio_map[k] = int(row["线体优先级"])
        takt_ref[k]      = int(row["生产节拍"])
        max_prio = max(max_prio, int(row["线体优先级"]))
    cap_map = {str(k): float(v) for k, v in zip(l["线体编码"], l["日产能"])}
    # 需求量：从订单列表独立读取,每工单的订单数量(不信任 solution 自报)
    order_qty_map = {int(k): int(v) for k, v in zip(o["工单编码"].astype(int), o["订单数量"].astype(int))}
    total_demand_qty = sum(order_qty_map.values())
    # 2026-08-23:交期一律取源数据。准时率原先拿「排产日期」去比**提交里自报的**
    # 「要求交付日期」,把那一列全填 2999-01-01 就能让准时率恒为 1.0。
    order_due_map = {int(k): str(v)[:10]
                     for k, v in zip(o["工单编码"].astype(int), o["要求交付日期"].astype(str))}

    total   = len(schedule_rows)
    sched   = [r for r in schedule_rows if r.get("是否未排", "Y") == "N"]
    n_sched = len(sched)

    if total == 0:
        return {k: 0 for k in ["schedule_rate", "on_time_rate", "same_day_line_rate",
                               "line_priority_score", "series_score", "balance_score", "batch_score"]}

    # 1. 排产率（按数量口径）：Σ已排产量 / Σ需求量。
    #    过载(需求>可排产能)时必然 <1，杜绝"每工单象征性排1件"刷满100%的漏洞。
    #    分子按工单聚合并以该工单订单数量封顶（防止超量记录/拆单 _余 记录虚增分子），
    #    与硬约束4"不超量排产"口径一致；分母为全部工单订单数量之和。
    sched_qty_by_wo = defaultdict(int)
    for r in sched:
        wo_str = str(r.get("工单编码", ""))
        base_wo = wo_str.split("_")[0]  # "123_余" -> "123"
        try:
            wo_code = int(base_wo)
        except ValueError:
            continue
        sched_qty_by_wo[wo_code] += int(r.get("排产数量", 0))
    scheduled_qty = 0
    for wo_code, used in sched_qty_by_wo.items():
        scheduled_qty += min(used, order_qty_map.get(wo_code, 0))
    schedule_rate = scheduled_qty / total_demand_qty if total_demand_qty > 0 else 0.0

    # 2. 准时率
    def _due_of(r):
        """交期只认源数据；工单编码解析不出来时退回自报值（并在下面计入不一致计数）。"""
        try:
            wc = int(str(r.get("工单编码", "")).split("-")[0].split("_")[0])
        except (TypeError, ValueError):
            return str(r.get("要求交付日期", "2999-01-01"))[:10]
        return order_due_map.get(wc, str(r.get("要求交付日期", "2999-01-01"))[:10])

    on_time = sum(
        1 for r in sched
        if str(r.get("排产日期", ""))[:10] <= _due_of(r)
    )
    on_time_rate = on_time / n_sched if n_sched > 0 else 0
    # 自报交期与源数据不一致的行数（仅诊断，不判违规）
    due_mismatch = sum(
        1 for r in sched
        if r.get("要求交付日期") and str(r.get("要求交付日期"))[:10] != _due_of(r)
    )

    # 3. 整单同天同线体率
    order_slots = defaultdict(set)
    for r in sched:
        order_slots[str(r.get("订单编码", ""))].add(
            (str(r.get("排产日期", ""))[:10], str(r.get("线体编码", "")))
        )
    n_orders = len(order_slots)
    same_dl = sum(1 for v in order_slots.values() if len(v) == 1)
    same_day_line_rate = same_dl / n_orders if n_orders > 0 else 0

    # 4. 线体适配分（归一化,独立从产品列表读优先级）
    prio_vals = []
    for r in sched:
        k = (str(r.get("产品编码", "")), str(r.get("线体编码", "")))
        prio_vals.append(prod_prio_map.get(k, max_prio))
    avg_prio = float(np.mean(prio_vals)) if prio_vals else float(max_prio)
    line_priority_score = (max_prio - avg_prio) / (max_prio - 1) if max_prio > 1 else 1.0

    # 5. 系列集中度
    slot_series = defaultdict(set)
    for r in sched:
        slot_series[(str(r.get("线体编码", "")), str(r.get("排产日期", ""))[:10])].add(
            str(r.get("系列编码", ""))
        )
    avg_series = float(np.mean([len(v) for v in slot_series.values()])) if slot_series else 1.0
    series_score = min(1.0 / avg_series, 1.0) if avg_series > 0 else 0.0

    # 6. 前紧后松+均衡（独立从线体信息读产能）
    slot_usage = defaultdict(float)
    for r in sched:
        k = (str(r.get("产品编码", "")), str(r.get("线体编码", "")))
        takt = takt_ref.get(k, 0)
        if takt > 0:
            dt = str(r.get("排产日期", ""))[:10]
            slot_usage[(str(r.get("线体编码", "")), dt)] += int(r.get("排产数量", 0)) * takt
    load_ratios = []
    for (lc, dt), usage in slot_usage.items():
        cap = cap_map.get(lc, 0)
        if cap > 0:
            load_ratios.append(min(usage / cap, 1.1))
    if load_ratios:
        std_load = float(np.std(load_ratios))
        balance_score = max(0.0, 1.0 - std_load / 0.3)
    else:
        balance_score = 0.5

    # 主目标比值 + 递归有界 tie-break：schedule_rate 为压倒性主目标(连续,不量化);
    # 下面 5 层量化到 R_TB 台阶后逐层嵌套成一个大进制定点数,再乘极小有界系数 TB_SCALE,
    # 保证全块 < 主目标一个最小台阶 ⇒ 严格字典序,且无大权重、不下溢(见文件头推导)。
    def _q(x):  # 量化到 [0, 1/R_TB] 的整数位
        x = 0.0 if x < 0 else (1.0 if x > 1 else x)
        return int(round(x / R_TB))
    tie_int = (
        _q(on_time_rate)        * M_TB**4
      + _q(same_day_line_rate)  * M_TB**3
      + _q(line_priority_score) * M_TB**2
      + _q(series_score)        * M_TB
      + _q(balance_score)
    )
    tie_break   = TB_SCALE * tie_int
    batch_score = schedule_rate + tie_break

    return {
        "schedule_rate":       round(schedule_rate, 4),
        "scheduled_qty":       int(scheduled_qty),
        "demand_qty":          int(total_demand_qty),
        "on_time_rate":        round(on_time_rate, 4),
        "same_day_line_rate":  round(same_day_line_rate, 4),
        "line_priority_score": round(line_priority_score, 4),
        "avg_line_priority":   round(avg_prio, 4),
        "series_score":        round(series_score, 4),
        "avg_series_per_slot": round(avg_series, 4),
        "balance_score":       round(balance_score, 4),
        "tie_break":           tie_break,
        "batch_score":         batch_score,
        "total":               total,
        "scheduled":           n_sched,
        "unscheduled":         total - n_sched,
    }


# ─── AgentCO 接口 ─────────────────────────────────────────────────────────────
def evaluate(file_path, data_dir):
    metrics = {"validity_score": 0.0, "quality_score": 0.0, "overall_score": 0.0, "error_info": {}}
    try:
        baseline_total = load_baseline()

        if not os.path.exists(file_path):
            metrics["error_info"] = {"fatal": [f"File not found: {file_path}"]}
            return metrics
        with open(file_path, "r", encoding="utf-8") as f:
            sub = json.load(f)

        batches = sub.get("batches")
        if not batches or not isinstance(batches, dict):
            metrics["error_info"] = {"fatal": ["Missing or empty 'batches'"]}
            return metrics

        orders_df, products_df, lines_df, calendar_df = load_reference_data(data_dir)

        all_validity  = []
        all_scores    = []
        batch_details = {}
        all_violations = []

        for sid in SOURCE_IDS:
            if sid not in batches:
                all_validity.append(0)
                batch_details[sid] = {"validity": 0, "error": "batch missing"}
                all_violations.append(f"{sid}: batch missing")
                continue
            schedule_rows = batches[sid].get("schedule", [])
            violations, n_viol = check_hard_constraints(
                schedule_rows, orders_df, products_df, lines_df, calendar_df, sid
            )
            validity = 1 if n_viol == 0 else 0
            all_validity.append(validity)
            if validity == 0:
                batch_details[sid] = {"validity": 0, "violation_count": n_viol, "violations": violations[:10]}
                all_violations.extend([f"{sid}: {v}" for v in violations[:3]])
            else:
                m = compute_quality(schedule_rows, orders_df, products_df, lines_df, calendar_df, sid)
                all_scores.append(m["batch_score"])
                batch_details[sid] = {"validity": 1, "metrics": m}

        overall_validity = 1 if all(v == 1 for v in all_validity) else 0
        # 总分=5批次 batch_score 均值。tie-break 层数值极小(≤~5e-4),
        # 不能用 round(,4) 抹掉分辨率,保留高精度。
        player_total = float(np.mean(all_scores)) if len(all_scores) == len(SOURCE_IDS) else 0.0

        metrics["batch_details"] = batch_details
        metrics["valid_batches"] = int(sum(all_validity))
        metrics["baseline_cost"] = baseline_total

        if overall_validity == 0:
            # 任一批次违反硬约束 -> validity=0, quality=0
            metrics["validity_score"] = 0.0
            metrics["quality_score"]  = 0.0
            metrics["overall_score"]  = 0.0
            metrics["error_info"] = {"constraint": all_violations[:10] or ["one or more batches invalid"]}
            return metrics

        metrics["validity_score"] = 1.0
        # 最大化目标: quality = 选手总分 / baseline 总分, 不加 min(,1) 截断
        quality = player_total / baseline_total if baseline_total > 0 else 0.0
        metrics["quality_score"] = round(quality, 6)
        metrics["overall_score"] = round(quality, 6)
        metrics["player_total_score"] = round(player_total, 8)

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
