"""
EXTRACTOR_SPEC:
  plan_file: solution.json
  required_columns: [scheduled_ops]
  notes: >
    The agent must produce an APS schedule as a JSON file `solution.json` describing,
    for every schedulable operation, which device it runs on and its start/end
    datetimes. An operation is schedulable iff
      WOP.STA=0  AND  its order ORD.STA=1  AND  its TON node has a TUS(ENA=1)
      resource that maps (via RES.GRP) to at least one ENA=1 concrete device
    (i.e. WOP.STA=0 ops belonging to STA=1 orders; ORD.STA=6 / other non-1 orders
    are skipped). This yields exactly 23 schedulable operations for the shipped
    dataset. Expected format:
      {
        "scheduled_ops": [
          {
            "wop_oid":      "WOP_0008",                          // WOP.OID of a schedulable operation
            "resource":     "RES_0045",                          // concrete device CODE (RES.CODE, ENA=1)
            "start_dt":     "2026-05-27T08:00:00",                // ISO8601, within work window
            "end_dt":       "2026-05-27T08:33:00",                // ISO8601
            "duration_min": 33.0                                  // MAK(min) x QTY, optional (recomputed)
          }
        ]
      }

    # 2026-08-23：原 schema 这里还写着一个 "exceptions" 数组（列出排不了的工序及原因），
    # 但 evaluator 从头到尾不读它 —— 填了不加分、不填不判违规，是个只在文档里存在的接口。
    # 已删除。排不出来的工序直接不写进 scheduled_ops 即可（覆盖率按已排工序算）；
    # 注意反过来也不要把「无 ENA=1 设备」的不可排工序硬塞进 scheduled_ops，
    # 那些行会被静默忽略、白占篇幅。

    Common agent output patterns to handle:
    - a CSV schedule (schedule_result.csv / 排程结果.csv) with columns for wop/order id,
      resource/device, start, end -> build the scheduled_ops list. Keep the de-identified
      WOP.OID (for example `WOP_0008`) as `wop_oid`; if the agent only wrote an ORD.OID or a CODE, map it back via the data
      tables so `wop_oid` matches a schedulable WOP.
    - datetimes as "2026-05-27 08:00" (space, no seconds) -> normalize to ISO8601.
    - resource given as a group code (`RES_0006`) instead of a concrete device code
      (`RES_0045`) -> this is a C2 violation; do NOT silently fix it, keep as given.
    - only schedulable ops are required; locked STA=1 / exception ops may be omitted.

    Use Bash with python3 to transform data if needed, then write solution.json
    to the output directory.
"""

import argparse
import json
import math
import os
import re
import traceback
from collections import defaultdict
from datetime import datetime, timedelta
from pathlib import Path

PLAN_FILE = "solution.json"
_HERE = os.path.dirname(os.path.abspath(__file__))
_DATA = os.path.join(_HERE, "..", "data")

# ── Work calendar (hardcoded from clarifications) ──
SCHEDULE_START = datetime(2026, 5, 27, 8, 0, 0)
WORK_START_H = 8
WORK_END_H = 18
WORK_DAYS = {0, 1, 2, 3, 4, 5}  # Mon-Sat, Sunday off


# ─────────────────────────────────────────────────────────────────────────────
# Data loading — read the required sheets directly from aps_data.xlsx.
# ─────────────────────────────────────────────────────────────────────────────

def _load_tables(data_dir):
    """Read the required sheets directly from the user's original upload
    aps_data.xlsx (each sheet is one table: ORD/TON/TUS/RES/WOP)."""
    import pandas as pd
    xlsx = os.path.join(data_dir, "aps_data.xlsx")
    sheets = ["ORD", "TON", "TUS", "RES", "WOP"]
    xl = pd.ExcelFile(xlsx)
    tabs = {s: pd.read_excel(xl, s) for s in sheets}
    return tabs["ORD"], tabs["TON"], tabs["TUS"], tabs["RES"], tabs["WOP"]


def _nan(v):
    if v is None:
        return True
    if isinstance(v, float) and math.isnan(v):
        return True
    return str(v).strip().lower() in ("nan", "none", "nat", "")


def _parse_mak(s):
    """'1.9MP' -> 1.9 (min/pc), '0.51HP' -> 30.6 (min/pc)."""
    if _nan(s):
        return None
    m = re.match(r"([\d.]+)\s*(MP|HP)", str(s).strip())
    if not m:
        return None
    v = float(m.group(1))
    return v * 60 if m.group(2) == "HP" else v


def _ton_seq(ton_oid):
    """工序顺序键: TON.OID 末尾数字段(尾部连续数字)升序即工序执行顺序。
    (information.md: TON.PCS 列恒为 0 不可用, 改用 OID 末尾序号)。
    e.g. `TON_0699_SEQ_0001` -> 1 and `TON_0004_SEQ_0012` -> 12.
    Returns None if no trailing digit run (then C6 skips that op)."""
    if _nan(ton_oid):
        return None
    m = re.search(r"(\d+)\s*$", str(ton_oid).strip())
    return int(m.group(1)) if m else None


def _parse_ole(v):
    """交期 -> datetime or None (empty / >=9000 => no due date)."""
    if _nan(v):
        return None
    try:
        import pandas as pd
        dt = pd.to_datetime(v)
        if dt.year >= 9000:
            return None
        return dt.to_pydatetime()
    except Exception:
        return None


# ─────────────────────────────────────────────────────────────────────────────
# Independent recompute of the schedulable set (never trust the submission for
# which ops exist, their durations, their valid devices, or their due dates).
# ─────────────────────────────────────────────────────────────────────────────

def _valid_devices_by_group(res_df):
    """resource-group name -> set of ENA=1 concrete device CODEs (RES.GRP not null)."""
    grp_devs = defaultdict(set)
    direct = set()
    for _, r in res_df.iterrows():
        code = str(r["CODE"]) if not _nan(r.get("CODE")) else ""
        grp = r.get("GRP")
        ena = r.get("ENA", 0)
        if not code or ena != 1:
            continue
        if not _nan(grp):
            grp_devs[str(grp)].add(code)   # concrete device -> its group
        else:
            direct.add(code)               # a group head that is itself enabled
    return grp_devs, direct


def compute_schedulable(ord_df, ton_df, tus_df, res_df, wop_df):
    """
    Returns:
      sched: {wop_oid: {devices:set, dur_min:float, ole:datetime|None, ord_oid, group}}
      all_valid_devices: set of all ENA=1 concrete device CODEs
    Rule: WOP.STA=0 AND its order ORD.STA=1 AND the TON node has a TUS(ENA=1)
    resource that maps (via RES.GRP) to >=1 ENA=1 concrete device.
    Orders that are not STA=1 (e.g. STA=6) are locked/skipped per the user spec, so
    their WOP.STA=0 ops are NOT schedulable. Duration = MAK(min) x WOP.QTY.
    """
    grp_devs, direct = _valid_devices_by_group(res_df)
    all_valid_devices = set()
    for devs in grp_devs.values():
        all_valid_devices |= devs
    all_valid_devices |= direct

    # TUS(ENA=1) grouped by TON node
    tus_by_ton = defaultdict(list)
    for _, t in tus_df[tus_df["ENA"] == 1].iterrows():
        tus_by_ton[str(t["TON"])].append(t)

    # ORD OID -> OLE, and ORD OID -> STA (only STA=1 orders are schedulable)
    ord_ole = {str(r["OID"]): _parse_ole(r.get("OLE")) for _, r in ord_df.iterrows()}
    ord_sta = {str(r["OID"]): r.get("STA") for _, r in ord_df.iterrows()}

    sched = {}
    for _, w in wop_df[wop_df["STA"] == 0].iterrows():
        ord_oid = str(w["ORD"]) if not _nan(w.get("ORD")) else ""
        # ORD.STA=1 filter: op belongs to a live (STA=1) order, else it is locked/skipped.
        if ord_sta.get(ord_oid) != 1:
            continue
        ton_oid = str(w["TON"]) if not _nan(w.get("TON")) else ""
        opts = tus_by_ton.get(ton_oid, [])
        if not opts:
            continue
        devices = set()
        mak = None
        group = None
        for t in opts:
            res_name = str(t["RES"]) if not _nan(t.get("RES")) else ""
            devs = grp_devs.get(res_name, set())
            if not devs and res_name in direct:
                devs = {res_name}
            if devs:
                devices |= devs
                if group is None:
                    group = res_name
            mk = _parse_mak(t.get("MAK"))
            if mk is not None and mak is None:
                mak = mk
        if not devices or mak is None:
            continue
        qty = float(w["QTY"]) if not _nan(w.get("QTY")) else 1.0
        wop_oid = str(w["OID"])
        sched[wop_oid] = {
            "devices": devices,
            "dur_min": round(mak * qty, 4),
            "ole": ord_ole.get(ord_oid),
            "ord_oid": ord_oid,
            "group": group,
            "ton_oid": ton_oid,          # for C6 process-precedence ordering
            "seq": _ton_seq(ton_oid),    # TON.OID trailing sequence number (工序顺序键)
        }
    return sched, all_valid_devices


# ─────────────────────────────────────────────────────────────────────────────
# Work-calendar helpers.
# ─────────────────────────────────────────────────────────────────────────────

def _work_minutes_from_start(dt):
    """Working minutes from SCHEDULE_START to dt (0 if dt<=start)."""
    if dt <= SCHEDULE_START:
        return 0.0
    total = 0.0
    cur = SCHEDULE_START
    while cur < dt:
        if cur.weekday() not in WORK_DAYS:
            cur = (cur + timedelta(days=1)).replace(hour=WORK_START_H, minute=0, second=0, microsecond=0)
            continue
        day_start = cur.replace(hour=WORK_START_H, minute=0, second=0, microsecond=0)
        day_end = cur.replace(hour=WORK_END_H, minute=0, second=0, microsecond=0)
        seg_s = max(cur, day_start)
        seg_e = min(dt, day_end)
        if seg_e > seg_s:
            total += (seg_e - seg_s).total_seconds() / 60.0
        cur = (cur + timedelta(days=1)).replace(hour=WORK_START_H, minute=0, second=0, microsecond=0)
    return total


def _valid_start(dt):
    return dt.weekday() in WORK_DAYS and WORK_START_H <= dt.hour < WORK_END_H


def _iso(s):
    try:
        return datetime.fromisoformat(str(s).rstrip("Z").replace(" ", "T")) if "T" not in str(s) and " " in str(s) \
            else datetime.fromisoformat(str(s).rstrip("Z"))
    except Exception:
        try:
            import pandas as pd
            return pd.to_datetime(s).to_pydatetime()
        except Exception:
            return None


# ─────────────────────────────────────────────────────────────────────────────
# Evaluate.
# ─────────────────────────────────────────────────────────────────────────────

def load_baseline():
    """基线的 absolute_score（与选手用同一 _absolute_score 公式算出）。"""
    with open(os.path.join(_HERE, "baseline", "reference_metrics.json")) as f:
        _d = json.load(f)
    return float(_d["reference_value"])


# ─── 评分：两层归一化到 [0,1] + 量化 + 大进制字典序合成为单一绝对分 ──────────────
# 尺子全部从数据/工作日历算出，不依赖 baseline 数值。
DELAY_RES_DAYS = 1.0 / 1440.0     # 主目标最小台阶：一个工作分钟（以天计）
R_TB = 0.001                      # tie-break 量化到 1000 档
M_TB = 1001                       # 进位基


def _worst_case_end(total_dur_min):
    """把全部可排工序串行摆在一台设备上、按 08:00-18:00 / 周一至周六 工作日历推进，
    得到的完工时刻。它是"最坏但仍然合法"的完工参考点：任何不空转、不重叠的排程都
    不会比它更晚，因此可作为逾期天数归一化的理论上界锚点（纯数据+日历推出）。"""
    remaining = float(total_dur_min)
    cur = SCHEDULE_START
    guard = 0
    while remaining > 1e-9 and guard < 100000:
        guard += 1
        if cur.weekday() not in WORK_DAYS:
            cur = (cur + timedelta(days=1)).replace(hour=WORK_START_H, minute=0, second=0, microsecond=0)
            continue
        day_end = cur.replace(hour=WORK_END_H, minute=0, second=0, microsecond=0)
        avail = (day_end - cur).total_seconds() / 60.0
        if avail <= 1e-9:
            cur = (cur + timedelta(days=1)).replace(hour=WORK_START_H, minute=0, second=0, microsecond=0)
            continue
        if remaining <= avail:
            return cur + timedelta(minutes=remaining)
        remaining -= avail
        cur = (cur + timedelta(days=1)).replace(hour=WORK_START_H, minute=0, second=0, microsecond=0)
    return cur


def compute_delay_upper(sched):
    """逾期天数的理论上界 = Σ_订单 max(0, 最坏完工时刻 − 该订单交期)。
    本数据集所有订单交期(2025-12~2026-01)都早于排程起点(2026-05-27)，逾期不可避免；
    上界刻画的是"全部工序串行排完、所有订单都拖到最后才完工"的最差情形。"""
    total_dur = sum(v["dur_min"] for v in sched.values())
    worst_end = _worst_case_end(total_dur)
    ole_by_ord = {}
    for v in sched.values():
        ole_by_ord[v["ord_oid"]] = v["ole"]
    upper = 0.0
    for ole in ole_by_ord.values():
        if ole is not None:
            upper += max(0.0, (worst_end - ole).total_seconds() / 86400.0)
    return max(upper, 1.0)


def _absolute_score(total_delay_days, utilization, delay_upper):
    """把「少逾期(主)」与「高设备利用率(次)」合成为单一绝对分。

    p0 = 1 − total_delay_days / delay_upper   ∈ [0,1]  主目标（越大越好）
    p1 = utilization                          ∈ [0,1]  次目标（本身即归一化利用率）

    S_PRIMARY = DELAY_RES_DAYS / delay_upper          主目标一个最小台阶(1 工作分钟)
    tie_break = p1_q × 0.49 × S_PRIMARY / M_TB        全量 < 主目标一台阶 ⇒ 主目标严格支配
    absolute  = p0 + tie_break
    """
    upper = max(float(delay_upper), 1.0)
    p0 = max(0.0, min(1.0, 1.0 - float(total_delay_days) / upper))
    p1 = max(0.0, min(1.0, float(utilization)))

    s_primary = DELAY_RES_DAYS / upper
    tb_scale = 0.49 * s_primary / M_TB
    p1_q = int(round(p1 / R_TB))
    return p0 + p1_q * tb_scale


def compute_utilization(valid_ops, sched):
    """
    设备综合利用率 (equipment utilization): fraction of available machine time that
    is actually spent processing, measured over the schedule's makespan window and
    only over the concrete devices that are actually used.

        util = Σ(op processing minutes) / (num_used_devices × makespan_work_minutes)

    where
      - Σ(op processing minutes) = sum of each covered op's independently recomputed
        MAK×QTY duration (dur_min);
      - num_used_devices = number of distinct concrete devices carrying ≥1 op;
      - makespan_work_minutes = working minutes (per the 08:00–18:00, Mon–Sat calendar)
        between the earliest op start and the latest op end.

    For a legal, non-overlapping schedule this lies in (0, 1]. Higher = the used
    devices are kept busier relative to the span they are held for. No min(·,1) cap.
    Returns 0.0 if the schedule has no positive makespan (single instant / empty).
    """
    starts = [p["start"] for p in valid_ops if p["start"] is not None]
    ends = [p["end"] for p in valid_ops if p["end"] is not None]
    used_devices = {p["resource"] for p in valid_ops if p["resource"]}
    if not starts or not ends or not used_devices:
        return 0.0
    earliest = min(starts)
    latest = max(ends)
    makespan_min = _work_minutes_from_start(latest) - _work_minutes_from_start(earliest)
    if makespan_min <= 0:
        return 0.0
    busy_min = sum(sched[p["wop_oid"]]["dur_min"] for p in valid_ops)
    return busy_min / (len(used_devices) * makespan_min)


def evaluate(file_path, data_dir):
    metrics = {"validity_score": 0.0, "quality_score": 0.0, "overall_score": 0.0, "error_info": {}}
    try:
        ord_df, ton_df, tus_df, res_df, wop_df = _load_tables(data_dir)
        sched, all_valid_devices = compute_schedulable(ord_df, ton_df, tus_df, res_df, wop_df)
        reference_value = load_baseline()

        if not os.path.exists(file_path):
            metrics["error_info"] = {"fatal": [f"File not found: {file_path}"]}
            return metrics
        with open(file_path, "r", encoding="utf-8") as f:
            sub = json.load(f)

        ops = sub.get("scheduled_ops")
        if not ops or not isinstance(ops, list):
            metrics["error_info"] = {"fatal": ["Missing or empty 'scheduled_ops'"]}
            return metrics

        # Parse submitted ops (only keep those that are actually schedulable).
        parsed = []
        for op in ops:
            wid = str(op.get("wop_oid", ""))
            res = str(op.get("resource", ""))
            sdt = _iso(op.get("start_dt", ""))
            edt = _iso(op.get("end_dt", ""))
            parsed.append({"wop_oid": wid, "resource": res, "start": sdt, "end": edt})

        violations = []

        # ── C4: coverage (independent schedulable set) ──
        covered = {p["wop_oid"] for p in parsed if p["wop_oid"] in sched}
        missing = set(sched.keys()) - covered
        if missing:
            violations.append(f"C4_coverage: {len(covered)}/{len(sched)} schedulable ops covered "
                              f"(missing {len(missing)}, e.g. {list(missing)[:3]})")

        # For the remaining constraint checks, use only ops that map to a real schedulable WOP.
        valid_ops = [p for p in parsed if p["wop_oid"] in sched]

        # basic well-formedness
        for p in valid_ops:
            if p["start"] is None or p["end"] is None:
                violations.append(f"bad datetime for {p['wop_oid']}")
                break

        # ── C2: resource legality (device must be in that op's allowed device set) ──
        c2 = 0
        for p in valid_ops:
            allowed = sched[p["wop_oid"]]["devices"]
            if p["resource"] not in allowed:
                c2 += 1
        if c2:
            violations.append(f"C2_resource: {c2} ops on illegal device")

        # ── C3: work-window start ──
        c3 = sum(1 for p in valid_ops if p["start"] is not None and not _valid_start(p["start"]))
        if c3:
            violations.append(f"C3_work_window: {c3} ops start outside work window")

        # ── C1: no device overlap ──
        c1 = 0
        by_res = defaultdict(list)
        for p in valid_ops:
            if p["start"] is not None and p["end"] is not None and p["resource"]:
                by_res[p["resource"]].append(p)
        for res, plist in by_res.items():
            plist.sort(key=lambda x: x["start"])
            for i in range(len(plist) - 1):
                if plist[i + 1]["start"] < plist[i]["end"]:
                    c1 += 1
        if c1:
            violations.append(f"C1_overlap: {c1} overlapping pairs")

        # ── C5: duration correctness (1% tolerance), against independent MAK x QTY ──
        c5 = 0
        for p in valid_ops:
            if p["start"] is None or p["end"] is None:
                continue
            actual = _work_minutes_from_start(p["end"]) - _work_minutes_from_start(p["start"])
            exp = sched[p["wop_oid"]]["dur_min"]
            if exp > 0 and abs(actual - exp) / exp > 0.01:
                c5 += 1
        if c5:
            violations.append(f"C5_duration: {c5} ops with wrong duration")

        # ── C6: process precedence (工序前后置) ──
        # Within each work order (ORD), the schedulable operations form a process
        # chain ordered by TON.OID trailing sequence number (ascending). The user
        # explicitly required: an upstream operation must be 100% complete (its end)
        # before the next (downstream) operation may start, i.e. start_downstream >=
        # end_upstream. Only checked between ops that are BOTH in the schedulable set
        # and share the same ORD; cross-order pairs are unconstrained. In the shipped
        # dataset every ORD contributes exactly one schedulable op (the second STA=0
        # op of the 8 multi-op orders has no ENA=1 device and is not schedulable), so
        # no chain has length>1 and C6 does not trigger here — but it is enforced
        # correctly whenever a same-order downstream op is scheduled before its
        # upstream op finishes.
        c6 = 0
        chain_by_ord = defaultdict(list)
        for p in valid_ops:
            if p["start"] is None or p["end"] is None:
                continue
            info = sched[p["wop_oid"]]
            if info.get("seq") is None:
                continue
            chain_by_ord[info["ord_oid"]].append(p)
        for oid, plist in chain_by_ord.items():
            if len(plist) < 2:
                continue
            plist.sort(key=lambda x: sched[x["wop_oid"]]["seq"])
            for i in range(len(plist) - 1):
                up, dn = plist[i], plist[i + 1]
                # downstream must not start before upstream is fully complete
                if dn["start"] < up["end"]:
                    c6 += 1
        if c6:
            violations.append(f"C6_precedence: {c6} downstream ops start before "
                              f"upstream op completes (工序前后置违反)")

        if violations:
            metrics["error_info"] = {"constraint": violations[:6]}
            return metrics

        metrics["validity_score"] = 1.0

        # ── QUALITY: primary = minimize total delay; secondary = maximize utilization ──
        # Independently recompute total delay days (primary) and equipment utilization
        # (secondary). We never trust numbers self-reported in the submission.
        ord_completion = {}
        for p in valid_ops:
            info = sched[p["wop_oid"]]
            oid = info["ord_oid"]
            end = p["end"]
            if oid not in ord_completion or end > ord_completion[oid][0]:
                ord_completion[oid] = (end, info["ole"])

        total_delay_days = 0.0
        delayed_orders = 0
        for oid, (end, ole) in ord_completion.items():
            if ole is not None and end > ole:
                total_delay_days += (end - ole).total_seconds() / 86400.0
                delayed_orders += 1

        player_util = compute_utilization(valid_ops, sched)

        # 流派 A：两层归一化 + 大进制字典序合成为单一 absolute_score，再对基线的
        # absolute_score 做一次除法归一（尺子从数据/日历算出，不依赖 baseline 数值）。
        delay_upper = compute_delay_upper(sched)
        absolute = _absolute_score(total_delay_days, player_util, delay_upper)
        quality = absolute / reference_value if reference_value > 0 else 0.0

        metrics["quality_score"] = round(quality, 8)
        metrics["overall_score"] = round(quality, 8)
        metrics["absolute_score"] = round(absolute, 10)
        metrics["reference_value"] = reference_value
        metrics["delay_upper_days"] = round(delay_upper, 4)
        metrics["total_delay_days"] = round(total_delay_days, 4)
        metrics["equipment_utilization"] = round(player_util, 6)
        metrics["delayed_orders"] = delayed_orders
        metrics["schedulable_ops"] = len(sched)
        metrics["covered_ops"] = len(covered)

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
