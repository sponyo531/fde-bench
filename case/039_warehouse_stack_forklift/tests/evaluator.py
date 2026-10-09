"""
EXTRACTOR_SPEC:
  plan_file: solution.json
  schema: |
    {
      "operations": [
        {"operation_id": "OP-00001", "type": "inbound", "sku": "SKU_0127",
         "batch": "20250630", "source": "PACKING", "target": "ZONE_A_LOC_041", "region": "ZONE_A",
         "tons": 2.0, "start_minute": 149, "end_minute": 152,
         "forklift": "FLT_001", "task_id": "IN_TASK_0001", "trip_id": "OP-00001"},
        ...
      ]
    }
  notes: >
    仓库垛位落位与叉车派工方案。选手产物只需包含当日全部作业行 operations；
    evaluator 会从期初库存独立重算期末库存，不要求选手重复提交。

    type 取值只能是 inbound / outbound / passive_move / active_move；若选手写成
    "入库"/"出库"/"移库"/"倒垛"/"IN"/"OUT"/"move" 等，映射到这四个英文值
    (移库/倒垛若无法区分主被动，一律映射成 active_move)。
    若选手用了别的键名 (op_id/id 代替 operation_id，from/to 代替 source/target，
    weight/qty 代替 tons，start/end 代替 start_minute/end_minute，
    driver/forklift_id 代替 forklift，area/zone 代替 region)，映射到上面的 schema。
    时间一律换算成"距计划期起点的分钟数"整数；tons 保留原始小数不要四舍五入。
    缺 trip_id 的行用它自己的 operation_id 补齐；缺 region 的行按 target/source 的
    垛位编号前两位 (ZONE_A/ZONE_B/ZONE_C) 补齐。
    不要自行重新求解、不要执行选手代码、不要改动任何数量与时刻。
"""
from __future__ import annotations

import argparse
import json
import os
import traceback
from collections import defaultdict
from pathlib import Path
from typing import Any, Dict, List, Tuple

PLAN_FILE = "solution.json"
DATA_FILE = "warehouse_snapshot.json"
_HERE = Path(__file__).resolve().parent

# 吨数一律换算成整数微吨(1e-6 t = 1 g)后比较, 避免浮点毛刺(数据中吨数最多 6 位小数)
UT = 1_000_000
TON_TOL_UT = 1_000          # 覆盖量/结存比对容差 0.001 t
VIRTUAL_IN = "PACKING"
VIRTUAL_OUT = "OUTBOUND"
OP_TYPES = ("inbound", "outbound", "passive_move", "active_move")
MOVE_TYPES = ("passive_move", "active_move")
MAX_VIOLATION_ROWS = 20
PERFECT_Q = 1e6


def _ut(value: Any) -> int:
    return int(round(float(value) * UT))


def _region_of(location: str) -> str:
    text = str(location or "")
    for prefix in ("ZONE_A", "ZONE_B", "ZONE_C"):
        if text.startswith(prefix):
            return prefix
    return ""


def load_baseline() -> Tuple[float, str]:
    with open(_HERE / "baseline" / "reference_metrics.json", encoding="utf-8") as f:
        ref = json.load(f)
    return float(ref["reference_value"]), str(ref.get("direction", "lower_is_better"))


def load_data(data_dir: str) -> Dict[str, Any]:
    with open(os.path.join(data_dir, DATA_FILE), encoding="utf-8-sig") as f:
        return json.load(f)


def combine(mixed_bays: int, overweight_tons: float, turnover_tons: float) -> float:
    """把三级字典序目标压成一个越低越好的实数。

    主目标 mixed_bays 占整数位; 第二级 (结束仍超重吨数) 与第三级 (主动搬运吨数)
    依次嵌套进小数部分并各自封顶, 保证 combined 落在 [mixed_bays, mixed_bays+1) 内,
    低优先级永远翻不了盘。两级都取 0.1 t 分辨率, 步长 1e-8 在 float64 与最终
    quality 比值上都清晰可辨, 不靠容差。
    """
    a = min(max(int(round(float(overweight_tons) * 10)), 0), 9_999)   # 0.1 t 分辨率
    b = min(max(int(round(float(turnover_tons) * 10)), 0), 9_999)     # 0.1 t 分辨率
    return int(mixed_bays) + (a + b / 10_000.0) / 10_000.0


def evaluate(submission_dir: str, data_dir: str) -> Dict[str, Any]:
    m: Dict[str, Any] = {
        "validity_score": 0.0, "quality_score": 0.0, "overall_score": 0.0, "error_info": {},
    }
    try:
        plan_path = os.path.join(submission_dir, PLAN_FILE)
        if not os.path.exists(plan_path):
            m["error_info"] = {"fatal": [f"缺 {PLAN_FILE}"]}
            return m
        with open(plan_path, encoding="utf-8-sig") as f:
            sol = json.load(f)
        if not isinstance(sol, dict):
            m["error_info"] = {"fatal": ["solution.json 顶层不是对象"]}
            return m
        ops_raw = sol.get("operations")
        if not isinstance(ops_raw, list) or not ops_raw:
            m["error_info"] = {"fatal": ["solution.json 缺 operations 或为空"]}
            return m

        data = load_data(data_dir)
        res = data.get("resources", {})
        cap_ut = {str(x["location"]): _ut(x.get("capacity_tons", 0) or 0)
                  for x in data.get("locations", [])}
        loc_region = {loc: _region_of(loc) for loc in cap_ut}
        max_trip_ut = _ut(res.get("max_task_tons", 2))
        dur = {
            "inbound": int(res.get("inbound_transport_minutes", 3)),
            "outbound": int(res.get("outbound_transport_minutes", 10)),
            "passive_move": int(res.get("move_minutes", 10)),
            "active_move": int(res.get("move_minutes", 10)),
        }
        day_minutes = int(res.get("day_minutes", 1440))

        forklifts: Dict[str, Dict[str, Any]] = {}
        for x in res.get("forklifts", []):
            fid = str(x.get("forklift_id", ""))
            if not fid:
                continue
            forklifts[fid] = {
                "regions": [str(r) for r in (x.get("region_scope") or [])],
                "types": {str(t) for t in (x.get("allowed_types") or OP_TYPES)},
                "shift_start": int(x.get("shift_start", 0)),
                "shift_end": int(x.get("shift_end", day_minutes)),
                "cap_ut": _ut(x.get("daily_ton_cap", 10 ** 9)),
            }

        in_tasks = {str(t["task_id"]): t for t in data.get("inbound_tasks", [])}
        out_tasks = {str(t["task_id"]): t for t in data.get("outbound_tasks", [])}

        # ── 期初状态: state[(loc, sku, batch)] = 微吨 ──────────────────────────
        state: Dict[Tuple[str, str, str], int] = defaultdict(int)
        for it in data.get("inventory", []):
            state[(str(it["location"]), str(it["sku"]), str(it["batch"]))] += _ut(it.get("tons", 0) or 0)
        init_weight: Dict[str, int] = defaultdict(int)
        for (loc, _s, _b), t in state.items():
            init_weight[loc] += t
        ceiling = {loc: max(cap_ut.get(loc, 0), init_weight.get(loc, 0)) for loc in cap_ut}
        weight: Dict[str, int] = defaultdict(int, init_weight)

        errs: List[str] = []

        def add(msg: str) -> None:
            if msg not in errs:
                errs.append(msg)

        # ── 逐行做结构性校验, 再按时间顺序推演库存 ────────────────────────────
        ops: List[Dict[str, Any]] = []
        for idx, raw in enumerate(ops_raw):
            oid = str(raw.get("operation_id") or f"operations[{idx}]")
            otype = str(raw.get("type", ""))
            if otype not in OP_TYPES:
                add(f"{oid}: 作业类型非法 ({otype})")
                continue
            try:
                start = int(round(float(raw["start_minute"])))
                end = int(round(float(raw["end_minute"])))
                tons_ut = _ut(raw["tons"])
            except (KeyError, TypeError, ValueError):
                add(f"{oid}: 时刻或吨数缺失/非数值")
                continue
            ops.append({
                "id": oid, "type": otype, "start": start, "end": end, "ut": tons_ut,
                "sku": str(raw.get("sku", "")), "batch": str(raw.get("batch", "")),
                "source": str(raw.get("source", "")), "target": str(raw.get("target", "")),
                "region": str(raw.get("region", "")), "forklift": str(raw.get("forklift", "")),
                "task_id": str(raw.get("task_id", "")),
                "trip_id": str(raw.get("trip_id") or oid),
            })
        if not ops:
            m["error_info"] = {"fatal": ["operations 中没有一条可解析的作业行"]}
            return m

        ops.sort(key=lambda o: (o["start"], o["id"]))
        covered_in: Dict[str, int] = defaultdict(int)
        covered_out: Dict[str, int] = defaultdict(int)
        fork_used: Dict[str, int] = defaultdict(int)

        for op in ops:
            oid, otype = op["id"], op["type"]

            # 时长 / 计划期
            if op["end"] - op["start"] != dur[otype]:
                add(f"{oid}: 作业时长 {op['end'] - op['start']} 分钟, 应为 {dur[otype]} 分钟")
            if op["start"] < 0 or op["end"] > day_minutes:
                add(f"{oid}: 作业区间 [{op['start']},{op['end']}] 超出计划期 [0,{day_minutes}]")
            # 单趟限重 (逐行先查, trip 合计后面再查)
            if op["ut"] <= 0 or op["ut"] > max_trip_ut:
                add(f"{oid}: 单行吨数 {op['ut'] / UT:.6f} 不在 (0, {max_trip_ut / UT:.3f}] 内")

            # 源 / 目的 / 库区
            src, tgt, region = op["source"], op["target"], op["region"]
            if otype == "inbound":
                if src != VIRTUAL_IN:
                    add(f"{oid}: 入库作业的 source 必须是 {VIRTUAL_IN}, 实际 {src}")
                if tgt not in cap_ut:
                    add(f"{oid}: 目标垛位 {tgt} 不存在")
                elif loc_region[tgt] != region:
                    add(f"{oid}: 标注库区 {region} 与目标垛位 {tgt} 所在库区 {loc_region[tgt]} 不符")
            elif otype == "outbound":
                if tgt != VIRTUAL_OUT:
                    add(f"{oid}: 出库作业的 target 必须是 {VIRTUAL_OUT}, 实际 {tgt}")
                if src not in cap_ut:
                    add(f"{oid}: 源垛位 {src} 不存在")
                elif loc_region[src] != region:
                    add(f"{oid}: 标注库区 {region} 与源垛位 {src} 所在库区 {loc_region[src]} 不符")
            else:
                if src not in cap_ut or tgt not in cap_ut:
                    add(f"{oid}: 倒垛的源/目标垛位 {src}->{tgt} 必须都是真实垛位")
                elif not (loc_region[src] == loc_region[tgt] == region):
                    add(f"{oid}: 跨库区倒垛 {src}({loc_region[src]}) -> {tgt}({loc_region[tgt]}), 标注库区 {region}")
                elif src == tgt:
                    add(f"{oid}: 倒垛的源与目标是同一垛位 {src}")

            # 叉车资质 / 班次 / 日吨位
            spec = forklifts.get(op["forklift"])
            if spec is None:
                add(f"{oid}: 叉车 {op['forklift']} 不在名单内")
            else:
                if region not in spec["regions"]:
                    add(f"{oid}: 叉车 {op['forklift']} 无库区 {region} 作业权限 (可作业 {spec['regions']})")
                if otype not in spec["types"]:
                    add(f"{oid}: 叉车 {op['forklift']} 不允许执行 {otype}")
                if op["start"] < spec["shift_start"] or op["end"] > spec["shift_end"]:
                    add(f"{oid}: 作业区间 [{op['start']},{op['end']}] 越出叉车 {op['forklift']} 班次 "
                        f"[{spec['shift_start']},{spec['shift_end']}]")
                fork_used[op["forklift"]] += op["ut"]

            # 任务绑定: 锁死量
            if otype == "inbound":
                task = in_tasks.get(op["task_id"])
                if task is None:
                    add(f"{oid}: 入库任务号 {op['task_id']} 不存在")
                else:
                    covered_in[op["task_id"]] += op["ut"]
                    if op["sku"] != str(task["sku"]) or op["batch"] != str(task["batch"]):
                        add(f"{oid}: 入库 SKU/批次 ({op['sku']},{op['batch']}) 与任务 "
                            f"{op['task_id']} ({task['sku']},{task['batch']}) 不符")
                    if op["start"] < int(task["start_minute"]):
                        add(f"{oid}: 入库开始时刻 {op['start']} 早于任务时刻 {task['start_minute']}")
            elif otype == "outbound":
                task = out_tasks.get(op["task_id"])
                if task is None:
                    add(f"{oid}: 出库任务号 {op['task_id']} 不存在")
                else:
                    covered_out[op["task_id"]] += op["ut"]
                    if op["sku"] != str(task["sku"]) or op["batch"] != str(task["batch"]):
                        add(f"{oid}: 出库 SKU/批次 ({op['sku']},{op['batch']}) 与任务 "
                            f"{op['task_id']} ({task['sku']},{task['batch']}) 不符")
                    if op["start"] < int(task["start_minute"]):
                        add(f"{oid}: 出库开始时刻 {op['start']} 早于任务时刻 {task['start_minute']}")
            else:
                if op["task_id"]:
                    add(f"{oid}: 倒垛作业不应挂任务号 {op['task_id']}")

            # 库存推演: 先扣后加
            if otype in ("outbound",) + MOVE_TYPES:
                key = (src, op["sku"], op["batch"])
                have = state.get(key, 0)
                if have < op["ut"]:
                    add(f"{oid}: 源垛位 {src} 上 ({op['sku']},{op['batch']}) 实时库存 "
                        f"{have / UT:.6f} t 不足以支撑本趟 {op['ut'] / UT:.6f} t")
                state[key] = have - op["ut"]
                weight[src] -= op["ut"]
            if otype in ("inbound",) + MOVE_TYPES:
                key = (tgt, op["sku"], op["batch"])
                state[key] = state.get(key, 0) + op["ut"]
                weight[tgt] += op["ut"]
                if tgt in cap_ut and weight[tgt] > ceiling[tgt]:
                    add(f"{oid}: 落位后 {tgt} 达 {weight[tgt] / UT:.3f} t, 超过其上限 "
                        f"{ceiling[tgt] / UT:.3f} t (额定 {cap_ut[tgt] / UT:.3f} t / "
                        f"期初 {init_weight.get(tgt, 0) / UT:.3f} t)")

        # ── 每趟合计限重 + 同趟共享时间窗 + 同一叉车不得并趟 ────────────────
        trips: Dict[Tuple[str, str], List[Dict[str, Any]]] = defaultdict(list)
        for op in ops:
            trips[(op["forklift"], op["trip_id"])].append(op)
        intervals: Dict[str, List[Tuple[int, int, str]]] = defaultdict(list)
        for (fid, tid), rows in trips.items():
            if len({(r["start"], r["end"]) for r in rows}) != 1:
                add(f"trip {tid}: 同一趟的多行未共享同一时间窗")
            total = sum(r["ut"] for r in rows)
            if total > max_trip_ut:
                add(f"trip {tid}: 单趟合计 {total / UT:.6f} t 超过 {max_trip_ut / UT:.3f} t")
            intervals[fid].append((rows[0]["start"], rows[0]["end"], tid))
        for fid, spans in intervals.items():
            spans.sort()
            for prev, cur in zip(spans, spans[1:]):
                if cur[0] < prev[1]:
                    add(f"叉车 {fid}: 趟次 {prev[2]} 与 {cur[2]} 时间重叠")

        for fid, spec in forklifts.items():
            if fork_used.get(fid, 0) > spec["cap_ut"]:
                add(f"叉车 {fid} 当日经手 {fork_used[fid] / UT:.3f} t, 超过日上限 "
                    f"{spec['cap_ut'] / UT:.3f} t")

        # ── 任务覆盖 ──────────────────────────────────────────────────────────
        for tid, task in in_tasks.items():
            got, need = covered_in.get(tid, 0), _ut(task["tons"])
            if abs(got - need) > TON_TOL_UT:
                add(f"入库任务 {tid} 完成 {got / UT:.3f} t, 应为 {need / UT:.3f} t")
        for tid, task in out_tasks.items():
            got, need = covered_out.get(tid, 0), _ut(task["tons"])
            if abs(got - need) > TON_TOL_UT:
                add(f"出库任务 {tid} 完成 {got / UT:.3f} t, 应为 {need / UT:.3f} t")

        # ── 评估器自行推演的结束库存 (打分只用它) ─────────────────────────────
        recomputed = {k: v for k, v in state.items()
                      if v > 0 and k[0] in cap_ut}
        for key, v in state.items():
            if v < -TON_TOL_UT and key[0] in cap_ut:
                add(f"垛位 {key[0]} 上 ({key[1]},{key[2]}) 结存为负 {v / UT:.6f} t")

        if errs:
            m["error_info"] = {"hard_violations": errs[:MAX_VIOLATION_ROWS], "total": len(errs)}
            return m
        m["validity_score"] = 1.0

        # ── 目标值: 全部由评估器从 data/ + operations 独立重算 ────────────────
        by_loc: Dict[str, List[Tuple[str, str]]] = defaultdict(list)
        end_weight: Dict[str, int] = defaultdict(int)
        for (loc, sku, batch), v in recomputed.items():
            by_loc[loc].append((sku, batch))
            end_weight[loc] += v

        mixed_bays = 0
        for lots in by_loc.values():
            n_sku = len({s for s, _ in lots})
            n_sb = len(set(lots))
            if n_sku > 1 or n_sb > n_sku:
                mixed_bays += 1

        overweight_ut = sum(max(0, w - cap_ut.get(loc, 0)) for loc, w in end_weight.items())
        turnover_ut = sum(o["ut"] for o in ops if o["type"] in MOVE_TYPES)
        turnover_trips = len({(o["forklift"], o["trip_id"]) for o in ops if o["type"] in MOVE_TYPES})
        sku_location_count = len({(sku, loc) for (loc, sku, _b) in recomputed})

        overweight_tons = overweight_ut / UT
        turnover_tons = turnover_ut / UT
        player_obj = combine(mixed_bays, overweight_tons, turnover_tons)

        ref, direction = load_baseline()
        if direction == "lower_is_better":
            quality = ref / player_obj if player_obj > 0 else PERFECT_Q
        else:
            quality = player_obj / ref if ref > 0 else 0.0

        m["quality_score"] = round(quality, 10)
        m["overall_score"] = m["quality_score"]
        m["player_objective"] = round(player_obj, 10)
        m["reference_value"] = ref
        m["metrics"] = {
            "mixed_bays": mixed_bays,
            "overweight_remaining_tons": round(overweight_tons, 3),
            "turnover_tons": round(turnover_tons, 3),
            "turnover_operations": turnover_trips,
            "sku_location_count": sku_location_count,
            "occupied_bays": len(by_loc),
            "operation_rows": len(ops),
            "trips": len(trips),
        }
        return m

    except Exception as exc:
        m["validity_score"] = 0.0
        m["error_info"] = {"exception": str(exc), "traceback": traceback.format_exc()[-800:]}
        return m


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--submission-dir", required=True)
    ap.add_argument("--data-dir", default=str(_HERE.parent / "data"))
    a = ap.parse_args()
    print(json.dumps(evaluate(a.submission_dir, a.data_dir), ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
