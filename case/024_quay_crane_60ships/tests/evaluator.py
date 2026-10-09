"""
EXTRACTOR_SPEC:
  plan_file: solution.json
  required_fields: [num_cranes, total_move_seconds, cranes]
  notes: >
    选手需产出港口岸桥调度方案 JSON，文件名 solution.json，放在提交目录根。
    结构：
      {
        "num_cranes": <int>,                 // 使用的岸桥总数（= cranes 数组长度）
        "total_move_seconds": <float>,       // 所有 move 事件耗时之和（仅供参考，评估器独立重算）
        "cranes": [
          {
            "crane_id": <int>,               // 岸桥编号；不可穿越校验按 crane_id 升序 == 位置升序
            "events": [                      // 该岸桥按时间排列的事件序列（work/move；空闲段可省略，评估器自动补 wait）
              {
                "type": "work"|"move"|"wait",
                "start_time": "YYYY-MM-DD HH:MM:SS",
                "end_time":   "YYYY-MM-DD HH:MM:SS",
                "berth":      <int|null>,     // work/wait 事件所在泊位；move 事件为 null
                "from_berth": <int|null>,     // move 事件出发泊位；否则 null
                "to_berth":   <int|null>,     // move 事件目标泊位；否则 null
                "vessel_id":  "<str>|null"    // work 事件服务的船舶ID；否则 null
              }
            ]
          }
        ]
      }
    常见需规整的情况：
    - crane_id 从 0 开始或乱序 -> 保持不变即可，评估器按 crane_id 排序做不穿越校验；
      但请确保 crane_id 越小的桥物理位置越靠左（泊位号越小）。
    - 事件用 event_type 而非 type、用 vessel 而非 vessel_id -> 请统一改成上述键名。
    - 时间戳带 'T' 或毫秒 -> 规整为 "YYYY-MM-DD HH:MM:SS"。
    - 若产物为 CSV(schedule.csv: crane_id,event_type,start_time,end_time,from_berth,to_berth,vessel_id)
      -> 转成上述 JSON：work 事件 berth=to_berth(或该行泊位)，move 事件填 from/to_berth。
    - num_cranes / total_move_seconds 缺失可由 cranes 推出，但建议补全。
    评估器只读该 JSON，不执行任何选手代码。

评估口径：
  - 硬约束 C1..C7 全部满足 -> validity_score=1，否则 0 且 quality=0。
  - score = num_cranes*100_000_000 + total_move_seconds（越小越好，评估器独立重算）。
    权重 1e8 保证字典序（先岸桥数、后移动耗时）：单桥活动时间轴长度 = 计划时域
    ≈ 690000 秒，任意可行解 total_move_seconds ≤ 岸桥数×时域，即便 40 台岸桥也
    < 40×690000 ≈ 2.76e7 < 1e8，故减少 1 台岸桥必然优于任意移动耗时增量，移动项
    永远无法翻盘岸桥数。（原实现权重 1e6 < 可行 total_move 上限(单台时域 6.9e5、
    多台可达数百万秒),移动耗时能翻盘岸桥数,已修。）
  - baseline_cost = 我方最优解 score（tests/baseline/reference_metrics.json）。
  - 最小化目标：quality_score = baseline_cost / player_score（选手更优时 >1，不截断）。
"""
import argparse
import csv
import json
import math
import os
import traceback
from datetime import datetime
from pathlib import Path

PLAN_FILE = "solution.json"
MOVE_TIME_PER_GRID = 880
_HERE = os.path.dirname(os.path.abspath(__file__))
_DATA = os.path.join(_HERE, "..", "data")


def parse_dt(s):
    return datetime.strptime(s, "%Y-%m-%d %H:%M:%S")


def load_data(data_dir):
    tact = {}
    # clean CSV assets retain their original bytes except for the approved
    # identifier/date mapping and carry a UTF-8 BOM; utf-8-sig keeps headers
    # aligned with the raw schema without touching the protected data file.
    with open(os.path.join(data_dir, "tact_times.csv"), encoding="utf-8-sig") as f:
        for r in csv.DictReader(f):
            tact[(r["op_type"], int(r["size_ft"]))] = (float(r["seconds_per_lift"]), int(r["max_per_lift"]))
    vessels = {}
    with open(os.path.join(data_dir, "vessel_workload.csv"), encoding="utf-8-sig") as f:
        for r in csv.DictReader(f):
            vid = r["vessel_id"]
            l20, l40 = int(r["load_20"]), int(r["load_40"])
            u20, u40 = int(r["unload_20"]), int(r["unload_40"])
            spl20, mpl20 = tact[("L", 20)]; spl40, mpl40 = tact[("L", 40)]
            spu20, mpu20 = tact[("U", 20)]; spu40, mpu40 = tact[("U", 40)]
            req = (math.ceil(l20 / mpl20) * spl20 + math.ceil(l40 / mpl40) * spl40 +
                   math.ceil(u20 / mpu20) * spu20 + math.ceil(u40 / mpu40) * spu40)
            vessels[vid] = {
                "berth_no": int(r["berth_no"]),
                "start_time": parse_dt(r["start_time"]),
                "end_time": parse_dt(r["end_time"]),
                "required_work_seconds": req,
            }
    return vessels


def _event_end_berth(ev):
    """事件结束时岸桥所在泊位：work/wait 取 berth，move 取 to_berth。"""
    if ev["type"] == "move":
        return ev.get("to_berth")
    return ev.get("berth")


def fill_idle_gaps(events):
    """在相邻事件之间的空档处自动补 wait 事件（岸桥停在前一事件的结束泊位）。

    C3「时间轴无缝覆盖」是产物格式约定，不是物理约束——岸桥空闲本身合法，
    补 wait 不影响目标（num_cranes/move 耗时）与 C6 位置（wait 停在原地）。
    故评估器代选手补齐空闲段，只把真正的物理约束（事件重叠、C7 单状态）交给校验。
    """
    evs = sorted(events, key=lambda e: parse_dt(e["start_time"]))
    filled = []
    for i, ev in enumerate(evs):
        if i > 0:
            prev = evs[i - 1]
            prev_e = parse_dt(prev["end_time"])
            cur_s = parse_dt(ev["start_time"])
            gap = (cur_s - prev_e).total_seconds()
            if gap > 1:  # 空档 > 1s，补一条 wait
                berth = _event_end_berth(prev)
                filled.append({
                    "type": "wait",
                    "start_time": prev["end_time"],
                    "end_time": ev["start_time"],
                    "berth": berth,
                    "from_berth": None,
                    "to_berth": None,
                    "vessel_id": None,
                })
        filled.append(ev)
    return filled


def get_crane_position(events, t_sec):
    evs = sorted(events, key=lambda e: parse_dt(e["start_time"]))
    for ev in evs:
        s = parse_dt(ev["start_time"]).timestamp()
        e = parse_dt(ev["end_time"]).timestamp()
        if s <= t_sec <= e:
            if ev["type"] in ("work", "wait"):
                return float(ev["berth"])
            elif ev["type"] == "move":
                frac = (t_sec - s) / (e - s) if e > s else 0
                return ev["from_berth"] + frac * (ev["to_berth"] - ev["from_berth"])
    return None


def validate(result, vessels):
    """独立重算所有约束。返回 (validity, violations, diagnostics)。"""
    violations = []
    diagnostics = {}
    cranes = result.get("cranes", [])

    # --- 预处理：自动补齐空闲段为 wait 事件（C3 无缝覆盖是格式约定，非物理约束）---
    for crane in cranes:
        crane["events"] = fill_idle_gaps(crane["events"])

    # --- C3, C7: 单桥时间轴不重叠且首尾衔接无间断（补 wait 后仅剩重叠会违规）---
    for crane in cranes:
        cid = crane.get("crane_id")
        events = sorted(crane["events"], key=lambda e: parse_dt(e["start_time"]))
        for i, ev in enumerate(events):
            s = parse_dt(ev["start_time"]); e = parse_dt(ev["end_time"])
            if e <= s:
                violations.append(f"C3/C7 crane={cid} event={i}: end_time <= start_time")
            if i > 0:
                prev_e = parse_dt(events[i - 1]["end_time"])
                gap = (s - prev_e).total_seconds()
                if gap < -1:  # 仅重叠（负间隔）判违规；空档已由 fill_idle_gaps 补齐
                    violations.append(
                        f"C7 crane={cid}: overlap between event {i-1} and {i}: {gap:.0f}s")

    # --- C4: 移动耗时正确性 ---
    for crane in cranes:
        cid = crane.get("crane_id")
        for ev in crane["events"]:
            if ev["type"] == "move":
                if ev.get("from_berth") is None or ev.get("to_berth") is None:
                    violations.append(f"C4 crane={cid}: move event missing from/to_berth")
                    continue
                dist = abs(int(ev["to_berth"]) - int(ev["from_berth"]))
                expected = dist * MOVE_TIME_PER_GRID
                s = parse_dt(ev["start_time"]); e = parse_dt(ev["end_time"])
                actual = (e - s).total_seconds()
                if abs(actual - expected) > 1:
                    violations.append(
                        f"C4 crane={cid}: move {ev['from_berth']}->{ev['to_berth']} expected={expected}s actual={actual:.0f}s")

    # --- C5 泊位匹配 & C2 时间窗口 & 累计有效作业时间 ---
    vessel_work_seconds = {vid: 0.0 for vid in vessels}
    for crane in cranes:
        cid = crane.get("crane_id")
        for ev in crane["events"]:
            if ev["type"] == "work":
                vid = ev.get("vessel_id")
                if not vid:
                    violations.append(f"C5 crane={cid}: work event missing vessel_id"); continue
                if vid not in vessels:
                    violations.append(f"C5 crane={cid}: unknown vessel_id={vid}"); continue
                v = vessels[vid]
                if int(ev["berth"]) != v["berth_no"]:
                    violations.append(
                        f"C5 crane={cid} vessel={vid}: work berth={ev['berth']} != vessel berth={v['berth_no']}")
                s = parse_dt(ev["start_time"]); e = parse_dt(ev["end_time"])
                eff_s = max(s, v["start_time"]); eff_e = min(e, v["end_time"])
                if eff_e > eff_s:
                    vessel_work_seconds[vid] += (eff_e - eff_s).total_seconds()
                if e > v["end_time"]:
                    violations.append(f"C2 crane={cid} vessel={vid}: work ends {e} after end_time {v['end_time']}")
                if s < v["start_time"]:
                    violations.append(f"C2 crane={cid} vessel={vid}: work starts {s} before start_time {v['start_time']}")

    # --- C1 作业量完成 ---
    # 作业量按【整数秒】口径判定：allocated / required 均四舍五入到秒后比较。
    # information 要求“100% 完成装卸量”，故不设人工容差；round 到秒仅消除时间戳可能的
    # 亚秒浮点表示误差，不放松“必须干满 required 秒”的业务要求。
    vessel_status = {}
    for vid, v in vessels.items():
        allocated = round(vessel_work_seconds[vid])
        req = round(v["required_work_seconds"])
        ok = allocated >= req
        vessel_status[vid] = {"required": req, "allocated": allocated, "ok": ok}
        if not ok:
            violations.append(
                f"C1 vessel={vid}: required={req:.0f}s allocated={allocated:.0f}s shortfall={req-allocated:.0f}s")

    # --- C6 全局不可穿越 ---
    all_times = set()
    for crane in cranes:
        for ev in crane["events"]:
            all_times.add(parse_dt(ev["start_time"]).timestamp())
            all_times.add(parse_dt(ev["end_time"]).timestamp())
            if ev["type"] == "move":
                s = parse_dt(ev["start_time"]).timestamp(); e = parse_dt(ev["end_time"]).timestamp()
                all_times.add((s + e) / 2)
    crossing = 0
    sorted_cranes = sorted(cranes, key=lambda c: c["crane_id"])
    for t in sorted(all_times):
        positions = []
        for crane in sorted_cranes:
            pos = get_crane_position(crane["events"], t)
            if pos is not None:
                positions.append((crane["crane_id"], pos))
        for i in range(len(positions) - 1):
            if positions[i][1] > positions[i + 1][1] + 0.01:
                crossing += 1
                if crossing <= 3:
                    violations.append(
                        f"C6 crossing: crane_{positions[i][0]} pos={positions[i][1]:.2f} > "
                        f"crane_{positions[i+1][0]} pos={positions[i+1][1]:.2f} at t={datetime.fromtimestamp(t)}")
    if crossing > 3:
        violations.append(f"C6: ... and {crossing-3} more crossing violations")

    diagnostics["vessels_ok"] = sum(1 for s in vessel_status.values() if s["ok"])
    diagnostics["vessels_total"] = len(vessels)
    diagnostics["crossing_violations"] = crossing
    return len(violations) == 0, violations, diagnostics


def compute_score(result):
    """独立重算 num_cranes 与 total_move_seconds（不信任选手自报）。"""
    cranes = result.get("cranes", [])
    num_cranes = len(cranes)
    total_move = 0.0
    for crane in cranes:
        for ev in crane["events"]:
            if ev["type"] == "move":
                s = parse_dt(ev["start_time"]); e = parse_dt(ev["end_time"])
                total_move += (e - s).total_seconds()
    return num_cranes * 100_000_000 + total_move, num_cranes, total_move


def load_baseline():
    with open(os.path.join(_HERE, "baseline", "reference_metrics.json")) as f:
        return float((lambda _d:_d.get("reference_value",_d.get("baseline_cost")))(json.load(f)))


def evaluate(file_path, data_dir):
    metrics = {"validity_score": 0.0, "quality_score": 0.0, "overall_score": 0.0, "error_info": {}}
    try:
        vessels = load_data(data_dir)
        baseline_cost = load_baseline()

        if not os.path.exists(file_path):
            metrics["error_info"] = {"fatal": [f"File not found: {file_path}"]}
            return metrics
        with open(file_path, "r", encoding="utf-8") as f:
            result = json.load(f)

        if not isinstance(result, dict) or "cranes" not in result or not isinstance(result["cranes"], list):
            metrics["error_info"] = {"schema": ["Missing or invalid 'cranes' array"]}
            return metrics
        if not result["cranes"]:
            metrics["error_info"] = {"schema": ["'cranes' is empty"]}
            return metrics

        validity, violations, diag = validate(result, vessels)
        if not validity:
            metrics["error_info"] = {
                "constraint": violations[:10],
                "vessels_ok": diag.get("vessels_ok", 0),
                "vessels_total": diag.get("vessels_total", 60),
                "crossing_violations": diag.get("crossing_violations", 0),
            }
            return metrics

        metrics["validity_score"] = 1.0
        player_score, num_cranes, total_move = compute_score(result)
        quality = baseline_cost / player_score if player_score > 0 else 0.0
        metrics["quality_score"] = round(quality, 6)
        metrics["overall_score"] = round(quality, 6)
        metrics["player_score"] = player_score
        metrics["num_cranes"] = num_cranes
        metrics["total_move_seconds"] = total_move
        metrics["baseline_cost"] = baseline_cost
        metrics["vessels_ok"] = diag.get("vessels_ok", 0)
        metrics["vessels_total"] = diag.get("vessels_total", 60)
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
