"""
EXTRACTOR_SPEC:
  plan_file: solution.csv
  required_columns: [crewId, taskId, isDDH]
  notes: >
    航空机组排班。选手的产物是一份 CSV `solution.csv`，严格三列：

        crewId,taskId,isDDH
        CREW_000001,FLIGHT_000001,0
        CREW_000001,POSITION_000001,1

    - crewId 取自 data.xlsx 的 crew.crewId
    - taskId 取自 flight.id（飞行任务）或 busInfo.id（地面置位）
    - isDDH：1 表示该任务是置位，0 表示正常执行
    - 占位任务（groundDuty）不出现在这份 CSV 里

    CSV 的行序即该机长的任务执行顺序无关——评估器按任务自身的时间字段重建时序，
    不依赖行序。

    Extractor: 从选手工作区找到最终排班结果（可能叫 roster_Result.csv / solution.csv /
    result.csv，或藏在 xlsx / 脚本输出里），规范化成上面的三列写到输出目录的
    solution.csv。列名大小写不一致（crewid/CrewID/机长）时映射到规范列名；
    isDDH 若用 True/False 或 是/否 表示，映射成 1/0。不执行选手代码、不改动任何 ID。

    评估器从 data/data.xlsx 独立重建六张表，校验全部硬约束与规则，并独立重算
    日均飞时、未覆盖航班、新增过夜站点、外站过夜天数、置位次数与各类违规次数，
    绝不采信选手自报的任何分数或统计量。
"""

from __future__ import annotations

import argparse
import json
import os
import traceback
from collections import defaultdict
from datetime import timedelta

import pandas as pd

_HERE = os.path.dirname(os.path.abspath(__file__))
PLAN_FILE = "solution.csv"

# ── 计划期与规则常量 ──
PLAN_START = pd.Timestamp("2025-05-29 00:00:00")
PLAN_END = pd.Timestamp("2025-06-04 23:59:59")

UNCOVERED_MAX_RATIO = 0.20     # 未覆盖航班上限占比
MAX_FLIGHT_TASKS_PER_DUTY = 4
MAX_TOTAL_TASKS_PER_DUTY = 6
MAX_FLIGHT_SEC_PER_DUTY = 8 * 3600
MAX_DUTY_SEC_PER_DUTY = 12 * 3600
MAX_TOTAL_DUTY_SEC = 60 * 3600
MIN_REST_SEC = 12 * 3600
TAIL_CHANGE_MIN_SEC = 3 * 3600     # 换飞机尾号最小间隔
BUS_ADJACENT_MIN_SEC = 2 * 3600    # 大巴与相邻飞行任务最小间隔
DUTY_SPLIT_REST_SEC = MIN_REST_SEC # 相邻任务间隔达到此值即切分值勤日
MAX_CYCLE_CALENDAR_DAYS = 4
CYCLE_REST_FULL_DAYS = 2

# 扣分权重
W_UNCOVERED = 5.0
W_NEW_LAYOVER = 10.0
W_OUTSTATION_DAY = 0.5
W_DEADHEAD = 0.5
W_VIOLATION = 10.0

# combined_score 的平移下界：全部航班未覆盖(4036×5=20180)即已远低于此值，
# 任何可行解都在其之上，平移后不改变解之间的优劣次序。
SCORE_FLOOR = -25000.0


def load_baseline():
    with open(os.path.join(_HERE, "baseline", "reference_metrics.json"), encoding="utf-8") as f:
        d = json.load(f)
    return float(d["reference_value"]), d.get("direction", "higher_is_better")


def _sec(ts):
    """时间戳 → 整数秒（§4 整数化，避免浮点毛刺）。"""
    return int(pd.Timestamp(ts).timestamp())


def load_data(data_dir):
    xl = pd.ExcelFile(os.path.join(data_dir, "data.xlsx"))
    flight = xl.parse("flight")
    crew = xl.parse("crew")
    bus = xl.parse("busInfo")
    ground = xl.parse("groundDuty")
    match = xl.parse("crewLegMatch")
    layover = xl.parse("layoverStation")

    for df, cols in ((flight, ["std", "sta"]), (bus, ["td", "ta"]),
                     (ground, ["startTime", "endTime"])):
        for c in cols:
            df[c] = pd.to_datetime(df[c], errors="coerce")

    flights = {}
    for r in flight.itertuples(index=False):
        flights[str(r.id)] = {
            "id": str(r.id), "dep": str(r.depaAirport), "arr": str(r.arriAirport),
            "start": _sec(r.std), "end": _sec(r.sta),
            "tail": str(r.aircraftNo),
            # flyTime 单位为分钟（已按数据核实：与 sta-std 比值中位数 0.97）
            "fly_sec": int(round(float(r.flyTime) * 60)),
            "kind": "flight",
        }
    buses = {}
    for r in bus.itertuples(index=False):
        buses[str(r.id)] = {
            "id": str(r.id), "dep": str(r.depaAirport), "arr": str(r.arriAirport),
            "start": _sec(r.td), "end": _sec(r.ta),
            "tail": None, "fly_sec": 0, "kind": "bus",
        }
    crews = {}
    for r in crew.itertuples(index=False):
        crews[str(r.crewId)] = {"base": str(r.base), "stay": str(r.stayStation)}

    grounds = defaultdict(list)
    for r in ground.itertuples(index=False):
        grounds[str(r.crewId)].append({
            "id": str(r.id), "airport": str(r.airport),
            "start": _sec(r.startTime), "end": _sec(r.endTime),
            "is_duty": int(r.isDuty) == 1, "kind": "ground",
        })
    for k in grounds:
        grounds[k].sort(key=lambda t: t["start"])

    qual = defaultdict(set)
    for r in match.itertuples(index=False):
        qual[str(r.crewId)].add(str(r.legId))

    layover_set = {str(a) for a in layover["airport"].dropna().tolist()}
    return flights, buses, crews, grounds, qual, layover_set


def _full_calendar_days_between(end_sec, start_sec):
    """两个时刻之间完整日历日的个数（不含端点所在日）。"""
    if start_sec <= end_sec:
        return 0
    d_end = pd.Timestamp(end_sec, unit="s").normalize()
    d_start = pd.Timestamp(start_sec, unit="s").normalize()
    return max(0, int((d_start - d_end).days) - 1)


def _midnights_crossed(a_sec, b_sec):
    """a→b 跨越的零点数。"""
    if b_sec <= a_sec:
        return 0
    da = pd.Timestamp(a_sec, unit="s").normalize()
    db = pd.Timestamp(b_sec, unit="s").normalize()
    return int((db - da).days)


def evaluate(submission_dir, data_dir):
    m = {"validity_score": 0.0, "quality_score": 0.0, "overall_score": 0.0, "error_info": {}}
    try:
        path = os.path.join(submission_dir, PLAN_FILE)
        if not os.path.exists(path):
            m["error_info"] = {"fatal": [f"缺 {PLAN_FILE}"]}
            return m

        flights, buses, crews, grounds, qual, layover_set = load_data(data_dir)
        baseline, direction = load_baseline()

        sub = pd.read_csv(path, dtype=str)
        need = {"crewId", "taskId", "isDDH"}
        if not need.issubset(set(sub.columns)):
            m["error_info"] = {"schema": [f"缺列，需 {sorted(need)}，实得 {list(sub.columns)}"]}
            return m
        sub = sub.dropna(subset=["crewId", "taskId"])

        # ── 解析提交，逐行落到机长任务表 ──
        assign = defaultdict(list)
        schema_err = []
        seen_pair = set()
        flight_exec = {}        # 航班 → 执行它的机长（isDDH=0）
        n_deadhead = 0
        for i, r in enumerate(sub.itertuples(index=False)):
            cid, tid = str(r.crewId).strip(), str(r.taskId).strip()
            try:
                ddh = int(float(str(r.isDDH).strip()))
            except Exception:
                schema_err.append(f"第{i+2}行 isDDH 无法解析: {r.isDDH!r}")
                continue
            if cid not in crews:
                schema_err.append(f"第{i+2}行 crewId 不存在: {cid}")
                continue
            task = flights.get(tid) or buses.get(tid)
            if task is None:
                schema_err.append(f"第{i+2}行 taskId 不存在于 flight/busInfo: {tid}")
                continue
            if (cid, tid) in seen_pair:
                schema_err.append(f"重复的 (crewId, taskId): ({cid}, {tid})")
                continue
            seen_pair.add((cid, tid))
            if ddh == 1:
                n_deadhead += 1
            else:
                if task["kind"] != "flight":
                    schema_err.append(f"第{i+2}行 非置位任务却指向大巴班次: {tid}")
                    continue
                if tid in flight_exec:
                    schema_err.append(f"航班 {tid} 被分配给多名机长: {flight_exec[tid]} 与 {cid}")
                    continue
                flight_exec[tid] = cid
            assign[cid].append({**task, "is_ddh": ddh == 1})

        if schema_err:
            m["error_info"] = {"schema": schema_err[:8], "n_schema_err": len(schema_err)}
            return m

        # ── HC1 航班覆盖率 ──
        total_flights = len(flights)
        n_uncovered = total_flights - len(flight_exec)
        if n_uncovered > UNCOVERED_MAX_RATIO * total_flights:
            m["error_info"] = {"constraint": [
                f"HC1 未覆盖航班 {n_uncovered}/{total_flights} "
                f"超过 {UNCOVERED_MAX_RATIO:.0%} 上限"]}
            return m

        violations = []      # 规则违规明细（计入扣分）
        n_violation = 0
        total_fly_sec = 0
        # 计分用的飞行时长。对违反单值勤日 8 小时上限的值勤日，
        # 只能按该上限计入收益；否则把大量重叠航班塞进一个值勤日，
        # 虽然会产生违规扣分，仍可通过不受限的分子把日均飞时放大到
        # 数百/上千小时，形成 evaluator hack。合法解的该值与原始值完全相同。
        scored_fly_sec = 0
        # 总飞行值勤日历日 = 各机长各自值勤日历日之和（跨零点计 2 日、同一机长同一日历日不重复）
        total_duty_days = 0
        new_layover_airports = set()
        outstation_days = 0

        for cid, tasks in assign.items():
            base = crews[cid]["base"]
            stay = crews[cid]["stay"]
            gds = grounds.get(cid, [])
            tasks.sort(key=lambda t: t["start"])

            # ── HC2 任务不重叠（含与占位任务；占位-占位之间不查）──
            timeline = [(t["start"], t["end"], t["id"], t["kind"]) for t in tasks]
            timeline += [(g["start"], g["end"], g["id"], "ground") for g in gds]
            timeline.sort()
            for a, b in zip(timeline, timeline[1:]):
                if a[3] == "ground" and b[3] == "ground":
                    continue
                if b[0] < a[1]:
                    violations.append(f"HC2 {cid} 任务时间重叠: {a[2]} 与 {b[2]}")
                    n_violation += 1

            # ── HC4 资质 ──
            for t in tasks:
                if t["kind"] == "flight" and not t["is_ddh"]:
                    if t["id"] not in qual.get(cid, set()):
                        violations.append(f"HC4 {cid} 无资质执飞 {t['id']}")
                        n_violation += 1

            # ── HC5 地点衔接 ──
            if tasks:
                if tasks[0]["dep"] != stay:
                    violations.append(
                        f"HC5 {cid} 首任务起点 {tasks[0]['dep']} != stayStation {stay}")
                    n_violation += 1
                for a, b in zip(tasks, tasks[1:]):
                    if a["arr"] != b["dep"]:
                        violations.append(
                            f"HC5 {cid} 地点断链: {a['id']} 到达 {a['arr']} 后 {b['id']} 从 {b['dep']} 出发")
                        n_violation += 1

            # ── HC7 连接时间：换尾号 3h、大巴与相邻飞行 2h ──
            for a, b in zip(tasks, tasks[1:]):
                gap = b["start"] - a["end"]
                if a["kind"] == "flight" and b["kind"] == "flight":
                    if a["tail"] != b["tail"] and gap < TAIL_CHANGE_MIN_SEC:
                        violations.append(
                            f"HC7 {cid} 换尾号间隔不足: {a['id']}->{b['id']} {gap}s < {TAIL_CHANGE_MIN_SEC}s")
                        n_violation += 1
                elif "bus" in (a["kind"], b["kind"]):
                    if gap < BUS_ADJACENT_MIN_SEC:
                        violations.append(
                            f"HC7 {cid} 大巴衔接间隔不足: {a['id']}->{b['id']} {gap}s < {BUS_ADJACENT_MIN_SEC}s")
                        n_violation += 1

            # ── 切分值勤日：相邻任务间隔 >= 12h 视为跨值勤日 ──
            duties = []
            cur = []
            for t in tasks:
                if cur and t["start"] - cur[-1]["end"] >= DUTY_SPLIT_REST_SEC:
                    duties.append(cur)
                    cur = []
                cur.append(t)
            if cur:
                duties.append(cur)

            crew_total_duty_sec = 0
            crew_duty_days = set()  # 该机长的飞行值勤日历日
            flight_duties = []      # 只保留含飞行任务的值勤日（飞行值勤日）
            for d in duties:
                n_f = sum(1 for t in d if t["kind"] == "flight")
                if n_f > MAX_FLIGHT_TASKS_PER_DUTY:
                    violations.append(f"HC8 {cid} 值勤日飞行任务 {n_f} > {MAX_FLIGHT_TASKS_PER_DUTY}")
                    n_violation += 1
                if len(d) > MAX_TOTAL_TASKS_PER_DUTY:
                    violations.append(f"HC8 {cid} 值勤日总任务 {len(d)} > {MAX_TOTAL_TASKS_PER_DUTY}")
                    n_violation += 1

                fly_sec = sum(t["fly_sec"] for t in d
                              if t["kind"] == "flight" and not t["is_ddh"])
                if fly_sec > MAX_FLIGHT_SEC_PER_DUTY:
                    violations.append(
                        f"HC9 {cid} 值勤日飞行时间 {fly_sec/3600:.2f}h > 8h")
                    n_violation += 1

                has_flight = any(t["kind"] == "flight" and not t["is_ddh"] for t in d)
                if has_flight:
                    last_fly_end = max(t["end"] for t in d
                                       if t["kind"] == "flight" and not t["is_ddh"])
                    duty_sec = last_fly_end - d[0]["start"]
                    if duty_sec > MAX_DUTY_SEC_PER_DUTY:
                        violations.append(
                            f"HC9 {cid} 值勤日值勤时间 {duty_sec/3600:.2f}h > 12h")
                        n_violation += 1
                    crew_total_duty_sec += duty_sec
                    flight_duties.append(d)
                    total_fly_sec += fly_sec
                    scored_fly_sec += min(fly_sec, MAX_FLIGHT_SEC_PER_DUTY)
                    # 总飞行值勤日历日：跨零点计 2 日，同日不重复
                    d0 = pd.Timestamp(d[0]["start"], unit="s").normalize()
                    d1 = pd.Timestamp(last_fly_end, unit="s").normalize()
                    nd = int((d1 - d0).days)
                    for k in range(nd + 1):
                        crew_duty_days.add((d0 + timedelta(days=k)).date())

            if crew_total_duty_sec > MAX_TOTAL_DUTY_SEC:
                violations.append(
                    f"HC9 {cid} 计划期飞行值勤时间 {crew_total_duty_sec/3600:.2f}h > 60h")
                n_violation += 1

            # ── HC10 飞行值勤日前最小休息 12h（休息占位可计入休息）──
            for prev, nxt in zip(flight_duties, flight_duties[1:]):
                rest = nxt[0]["start"] - prev[-1]["end"]
                if rest < MIN_REST_SEC:
                    violations.append(
                        f"HC10 {cid} 飞行值勤日前休息 {rest/3600:.2f}h < 12h")
                    n_violation += 1

            # ── HC11 飞行周期：跨度<=4 日历日，周期前 >=2 完整日历日休息且在基地 ──
            cycles = []
            cyc = []
            for d in flight_duties:
                if cyc:
                    gap_full_days = _full_calendar_days_between(cyc[-1][-1]["end"], d[0]["start"])
                    if gap_full_days >= CYCLE_REST_FULL_DAYS:
                        cycles.append(cyc)
                        cyc = []
                cyc.append(d)
            if cyc:
                cycles.append(cyc)
            for ci, c in enumerate(cycles):
                s = pd.Timestamp(c[0][0]["start"], unit="s").normalize()
                e = pd.Timestamp(max(t["end"] for d in c for t in d), unit="s").normalize()
                span = int((e - s).days) + 1
                if span > MAX_CYCLE_CALENDAR_DAYS:
                    violations.append(
                        f"HC11 {cid} 飞行周期跨 {span} 个日历日 > {MAX_CYCLE_CALENDAR_DAYS}")
                    n_violation += 1
                if ci > 0:
                    prev_end = max(t["end"] for d in cycles[ci - 1] for t in d)
                    prev_arr = None
                    for d in cycles[ci - 1][::-1]:
                        for t in d[::-1]:
                            if t["end"] == prev_end:
                                prev_arr = t["arr"]
                                break
                        if prev_arr:
                            break
                    if prev_arr is not None and prev_arr != base:
                        violations.append(
                            f"HC11 {cid} 飞行周期前的休息不在基地: 休息于 {prev_arr}, 基地 {base}")
                        n_violation += 1

            # ── HC6 置位只允许在值勤日首尾；地面置位须来自 busInfo（已由 taskId 校验保证）──
            for d in duties:
                for pos, t in enumerate(d):
                    if t["is_ddh"] and 0 < pos < len(d) - 1:
                        violations.append(
                            f"HC6 {cid} 置位 {t['id']} 出现在值勤日中间（位置 {pos+1}/{len(d)}）")
                        n_violation += 1

            # ── 新增过夜站点 / 外站过夜天数 ──
            for d in flight_duties:
                for ap in (d[0]["dep"], d[-1]["arr"]):
                    if ap not in layover_set:
                        new_layover_airports.add(ap)
            if tasks and stay not in layover_set:
                new_layover_airports.add(stay)

            if tasks:
                # 历史停留为外站：计划期开始 → 首任务开始 的跨零点天数
                if stay != base:
                    outstation_days += _midnights_crossed(_sec(PLAN_START), tasks[0]["start"])
                # 值勤日之间的过夜
                for prev, nxt in zip(duties, duties[1:]):
                    ap = prev[-1]["arr"]
                    if ap != base:
                        cross = _midnights_crossed(prev[-1]["end"], nxt[0]["start"])
                        outstation_days += cross if cross > 0 else 1
                # 计划期结束时停在非基地
                if tasks[-1]["arr"] != base:
                    outstation_days += _midnights_crossed(tasks[-1]["end"], _sec(PLAN_END))

            total_duty_days += len(crew_duty_days)

        m["validity_score"] = 1.0

        # ── 独立重算 combined_score ──
        n_duty_days = total_duty_days
        raw_avg_daily_fly_h = (total_fly_sec / 3600.0 / n_duty_days) if n_duty_days > 0 else 0.0
        avg_daily_fly_h = (scored_fly_sec / 3600.0 / n_duty_days) if n_duty_days > 0 else 0.0
        n_new_layover = len(new_layover_airports)
        score = (avg_daily_fly_h * 1000.0
                 - W_UNCOVERED * n_uncovered
                 - W_NEW_LAYOVER * n_new_layover
                 - W_OUTSTATION_DAY * outstation_days
                 - W_DEADHEAD * n_deadhead
                 - W_VIOLATION * n_violation)

        # combined_score 可能为负（扣分压过收益），直接做比值会让语义颠倒
        # （越差的负分除以负 baseline 反而更大）。故统一平移到非负区间后再比。
        # SCORE_FLOOR 取一个显著低于任何可行解的下界，平移后单调性与原分一致。
        shifted_player = max(0.0, score - SCORE_FLOOR)
        shifted_base = max(1e-9, baseline - SCORE_FLOOR)
        if direction == "lower_is_better":
            quality = shifted_base / shifted_player if shifted_player > 0 else 0.0
        else:
            quality = shifted_player / shifted_base
        quality = max(0.0, quality)

        m["quality_score"] = round(quality, 6)
        m["overall_score"] = m["quality_score"]
        m["player_objective"] = round(score, 6)
        m["reference_value"] = round(baseline, 6)
        m["metric"] = {
            "avg_daily_flight_hours": round(avg_daily_fly_h, 4),
            "total_flight_hours": round(total_fly_sec / 3600.0, 2),
            "scored_total_flight_hours": round(scored_fly_sec / 3600.0, 2),
            "raw_avg_daily_flight_hours": round(raw_avg_daily_fly_h, 4),
            "total_duty_calendar_days": n_duty_days,
            "uncovered_flights": n_uncovered,
            "new_layover_stations": n_new_layover,
            "outstation_overnight_days": outstation_days,
            "deadhead_count": n_deadhead,
            "violation_count": n_violation,
        }
        if violations:
            m["error_info"] = {"rule_violation": violations[:10],
                               "n_violation": n_violation}
    except Exception as e:
        m["error_info"] = {"exception": str(e), "traceback": traceback.format_exc()}
    return m


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--submission-dir", required=True)
    ap.add_argument("--data-dir", default=os.path.join(_HERE, "..", "data"))
    a = ap.parse_args()
    print(json.dumps(evaluate(a.submission_dir, a.data_dir), ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
