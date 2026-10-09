"""
EXTRACTOR_SPEC:
  plan_file: solution.csv
  required_columns: [server_id, rack_id, pos, allocation_status, batch]
  notes: >
    数据中心服务器上架分配。每台服务器一行, 共 1112 行。
    - server_id: server.csv 里的 id, 保持原样(字符串比对, 不要补零或转科学计数法)
    - rack_id:   rack_col.csv 里的机柜列 id; 未分配时留空
    - pos:       起始 U 位号(rack.csv 的 pos); 未分配时留空
    - allocation_status: 1 已分配 / 0 未分配
    - batch:     server.csv 里的批次号
    列名可能叫 服务器/机柜/位置/状态/批次, 或 rack_col_id / start_pos / assigned,
    映射到上面五列。若选手交的是 JSON 数组(每项一个 dict), 展开成表格。
    若选手交了 solution.py 而非结果表, 说明产物不符合要求, 不要代为执行取结果。
"""
from __future__ import annotations

import csv
import json
import math
import os
from pathlib import Path
from typing import Any, Dict

_HERE = Path(__file__).resolve().parent
PLAN_FILE = "solution.csv"

W = {"w_power": 0.4, "w_operate": 0.05, "w_active": 0.3, "w_attr": 0.25}
UNASSIGNED_COST = 1.0
POWER_TOL = 1e-6



def load_baseline() -> tuple:
    with open(_HERE / "baseline" / "reference_metrics.json", encoding="utf-8") as f:
        d = json.load(f)
    return float(d["reference_value"]), d.get("direction", "lower_is_better")


def _parse_pos(raw) -> int | None:
    """U 位号解析, 与业务侧口径一致:
    纯数字直接取值; 形如 '12A' 的取数字部分(同一物理位的 A 面);
    '0B'/'0C'/'0D' 这类其余后缀不是可上架位, 跳过。
    """
    s = str(raw).strip().upper()
    if s.isdigit():
        return int(s)
    if s.endswith("A") and s[:-1].isdigit():
        return int(s[:-1])
    return None


def _load_data(data_dir: str):
    servers = {}
    with open(os.path.join(data_dir, "server.csv"), encoding="utf-8-sig") as f:
        for r in csv.DictReader(f):
            servers[str(r["id"]).strip()] = {
                "batch": str(r["batch"]).strip(),
                "power": float(r["power"]),
                "height": int(float(r["height"])),
                "nic": str(r["nic"]).strip(),
            }
    racks = {}
    with open(os.path.join(data_dir, "rack_col.csv"), encoding="utf-8-sig") as f:
        for r in csv.DictReader(f):
            racks[str(r["id"]).strip()] = {
                "rated_power": float(r["rated_power"]),
                "used_power": float(r["used_power"]),
                "state": int(float(r["state"])),
                "attr": str(r["attr"]).strip(),
                "units": {},
            }
    with open(os.path.join(data_dir, "rack.csv"), encoding="utf-8-sig") as f:
        for r in csv.DictReader(f):
            cid = str(r["col_id"]).strip()
            if cid not in racks:
                continue
            p = _parse_pos(r["pos"])
            if p is None:
                continue
            racks[cid]["units"][p] = int(float(r["state"]))
    nic_cfg: Dict[str, Dict[str, float]] = {}
    with open(os.path.join(data_dir, "nic_attr_mapping.csv"), encoding="utf-8-sig") as f:
        for r in csv.DictReader(f):
            nic_cfg.setdefault(str(r["nic"]).strip(), {})[str(r["attr"]).strip()] = float(r["weight"])
    return servers, racks, nic_cfg


def _read_plan(path: str) -> list:
    rows = []
    with open(path, encoding="utf-8-sig") as f:
        for r in csv.DictReader(f):
            rows.append(r)
    return rows


def evaluate(submission_dir: str, data_dir: str) -> Dict[str, Any]:
    m: Dict[str, Any] = {
        "validity_score": 0.0, "quality_score": 0.0, "overall_score": 0.0, "error_info": {},
    }
    try:
        ref_value, direction = load_baseline()
        m["reference_value"] = ref_value

        plan = os.path.join(submission_dir, PLAN_FILE)
        if not os.path.exists(plan):
            m["error_info"] = {"fatal": [f"缺 {PLAN_FILE}"]}
            return m
        rows = _read_plan(plan)
        if not rows:
            m["error_info"] = {"fatal": [f"{PLAN_FILE} 为空"]}
            return m

        servers, racks, nic_cfg = _load_data(data_dir)

        def fresh_state():
            """每个批次从到货前的原始存量重新起算（批次之间互不影响）。"""
            return {rid: {"used_power": r["used_power"], "state": r["state"],
                          "units": dict(r["units"])} for rid, r in racks.items()}

        # 按 server.csv 的批次归属分组（不信任提交里的 batch 列）
        rows_by_batch: Dict[str, list] = {}
        for row in rows:
            sid = str(row.get("server_id", "")).strip()
            b = servers[sid]["batch"] if sid in servers else "__unknown__"
            rows_by_batch.setdefault(b, []).append(row)

        seen = set()
        allocs = []          # (sid, rid, pos, batch) 已上架
        unassigned = []
        errors = []
        dyn_by_batch: Dict[str, dict] = {}

        for bid, brows in rows_by_batch.items():
            dyn = fresh_state()
            dyn_by_batch[bid] = dyn
            errors.extend(_place_batch(brows, servers, racks, nic_cfg, dyn, seen, allocs,
                                       unassigned, bid))
            if len(errors) >= 8:
                break

        if errors:
            m["error_info"] = {"hard_violations": errors[:8], "total": len(errors)}
            return m

        missing = set(servers) - seen
        if missing:
            m["error_info"] = {"hard_violations": [
                f"{len(missing)} 台服务器未出现在方案里, 例 {sorted(missing)[:5]}"]}
            return m

        # ── 成本(按各批次最终机柜状态) ──
        total = 0.0
        for sid, rid, pos, bid in allocs:
            srv = servers[sid]
            rk = racks[rid]
            dr = dyn_by_batch[bid][rid]
            rated = rk["rated_power"]
            c_power = (rated - dr["used_power"]) / rated if rated > 0 else 1.0
            total_slots = len(rk["units"])
            c_ops = pos / total_slots if total_slots > 0 else 1.0
            c_active = 1.0 - float(dr["state"])
            c_attr = 1.0 - nic_cfg.get(srv["nic"], {}).get(rk["attr"], 0.0)
            total += (W["w_power"] * c_power + W["w_operate"] * c_ops
                      + W["w_active"] * c_active + W["w_attr"] * c_attr)
        total += UNASSIGNED_COST * len(unassigned)

        if not math.isfinite(total):
            m["error_info"] = {"hard_violations": ["成本计算得到非有限值"]}
            return m

        m["validity_score"] = 1.0
        m["n_allocated"] = len(allocs)
        m["n_unassigned"] = len(unassigned)
        m["n_batches"] = len(rows_by_batch)
        m["player_objective"] = round(total, 6)

        if direction == "lower_is_better":
            quality = ref_value / total if total > 0 else 0.0
        else:
            quality = total / ref_value if ref_value > 0 else 0.0
        m["quality_score"] = round((quality), 6)
        m["overall_score"] = m["quality_score"]
        return m

    except Exception as e:
        import traceback
        m["validity_score"] = 0.0
        m["error_info"] = {"exception": str(e), "traceback": traceback.format_exc()[-800:]}
        return m


def _place_batch(rows, servers, racks, nic_cfg, dyn, seen, allocs, unassigned, bid) -> list:
    """在单个批次内逐台落位并校验硬约束，返回错误列表。"""
    errors: list = []
    for row in rows:
        sid = str(row.get("server_id", "")).strip()
        if sid not in servers:
            errors.append(f"批次 {bid}: 未知服务器 {sid!r}")
            break
        if sid in seen:
            errors.append(f"服务器 {sid} 出现两次")
            break
        seen.add(sid)

        st_raw = str(row.get("allocation_status", "")).strip()
        try:
            status = int(float(st_raw)) if st_raw != "" else 1
        except ValueError:
            errors.append(f"服务器 {sid}: allocation_status 非法 {st_raw!r}")
            break
        if status != 1:
            unassigned.append(sid)
            continue

        rid = str(row.get("rack_id", "")).strip()
        if rid not in racks:
            errors.append(f"服务器 {sid}: 未知机柜 {rid!r}")
            break
        pos_raw = str(row.get("pos", "")).strip()
        if pos_raw == "":
            errors.append(f"服务器 {sid}: allocation_status=1 但缺 pos")
            break
        pos = _parse_pos(pos_raw)
        if pos is None:
            errors.append(f"服务器 {sid}: pos 无法解析为可上架位号 {pos_raw!r}")
            break

        srv = servers[sid]
        rk = racks[rid]
        dr = dyn[rid]

        # 硬约束 1: 网卡与机柜属性兼容
        if rk["attr"] not in nic_cfg.get(srv["nic"], {}):
            errors.append(
                f"服务器 {sid}(网卡 {srv['nic']}) 与机柜 {rid}(属性 {rk['attr']}) 不兼容")
            break
        # 硬约束 2: 电力上限
        if dr["used_power"] + srv["power"] > rk["rated_power"] + POWER_TOL:
            errors.append(
                f"批次 {bid} 机柜 {rid} 加入 {sid} 后功率 "
                f"{dr['used_power'] + srv['power']:.1f} 超过额定 {rk['rated_power']:.1f}")
            break
        # 硬约束 3: 连续空位
        bad = None
        for k in range(srv["height"]):
            p = pos + k
            if p not in dr["units"]:
                bad = f"机柜 {rid} 不存在 U 位 {p}"
                break
            if dr["units"][p] == 1:
                bad = f"机柜 {rid} 的 U 位 {p} 已被占用"
                break
        if bad:
            errors.append(f"服务器 {sid}: {bad}")
            break

        # 落位
        dr["used_power"] += srv["power"]
        dr["state"] = 1
        for k in range(srv["height"]):
            dr["units"][pos + k] = 1
        allocs.append((sid, rid, pos, bid))
    return errors


def main() -> None:
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--submission-dir", required=True)
    ap.add_argument("--data-dir", default=str(_HERE.parent / "data"))
    a = ap.parse_args()
    print(json.dumps(evaluate(a.submission_dir, a.data_dir), ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
