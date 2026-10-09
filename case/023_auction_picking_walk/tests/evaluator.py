"""
EXTRACTOR_SPEC:
  plan_file: solution.json
  required_columns: [groups]
  notes: >
    The agent must produce a JSON file describing, for each of the 445 task
    segments (one per unique (分货员工代码 worker, 大钟 bell)), the visiting order
    of carts and, within each cart, the visiting order of picking slots.
    Records whose 分货员工代码 is empty or the literal string "NULL" are anomalous
    ownerless carts and MUST be filtered out (they are NOT expected segments;
    including them yields an "extra segment" violation).
    Expected format:
      {
        "total_distance_m": 250000.0,          // optional, evaluator recomputes independently
        "n_groups": 445,
        "groups": [
          {
            "worker_id": "WORK0179",              // 分货员工代码 as string
            "bell": 1,                          // 大钟 int 1..8
            "cart_order": ["CART05798", "CART05811"],       // visiting order of this segment's carts (strings)
            "slot_sequences": {                  // per-cart slot visiting order (slot codes like S0002)
              "CART05798": ["S0002", "S0007"],
              "CART05811": ["S0068"]
            }
          }, ...
        ]
      }

    Semantics the evaluator enforces (recomputed independently from data/):
    - A "slot code" is derived from 一次分货位 "AA"+code (e.g. AAS0002 -> S0002).
    - EACH CART IS AN INDEPENDENT TRIP. The physical model: a worker takes ONE cart
      (原始台车) at a time out of the bell, pushes it along the aisles dropping goods
      slot by slot, then returns for the next cart. So dedup is PER-CART ONLY:
        * WITHIN one cart, a slot repeated across records is visited once (drop
          everything for that slot in one stop).
        * ACROSS carts there is NO dedup — if the same slot appears on two different
          carts of the same worker×bell, the worker (pushing a different cart each
          time) goes there once per cart, and BOTH visits count for distance.
      Therefore each cart's slot_sequences[cart] must equal EXACTLY the set of
      distinct mappable slots that appear on THAT cart's records (cart-internal
      dedup), with no repeats within the cart. cart_order must list every cart of
      the segment exactly once.
    - Path = bell_node -> (cart_order[0] slots in order) -> (cart_order[1] slots) -> ...
      Each cart is a fresh leg starting from wherever the previous cart's last slot
      was (carts are done back-to-back without returning to the bell between them,
      and no return to bell at the end). Distance uses bfs_dist real-aisle shortest
      path (mm). A slot shared by two carts is traversed twice (once per cart leg).

    Common agent output patterns to handle / normalize into solution.json:
    - worker_id / cart ids as ints -> cast to str.
    - a flat per-segment slot order without cart grouping -> this is NO LONGER
      distance-neutral, because cross-cart dedup is forbidden. If the agent only
      produced a per-segment order, you must re-split it back to the per-cart slot
      sets from data/ so that each cart carries exactly its own distinct slots.
    - slot codes with the "AA" prefix still attached (AAS0002) -> strip to S0002.
    Write the normalized solution.json to the output directory.
"""

import argparse
import json
import os
import re
import traceback
from pathlib import Path

PLAN_FILE = "solution.json"
_HERE = os.path.dirname(os.path.abspath(__file__))
_DATA = os.path.join(_HERE, "..", "data")

VALID_SLOT_RE = re.compile(r'^AA(S\d{4})$')
BELL_NODE_IDS = {i: f"{i}号钟" for i in range(1, 9)}
SLOT_FALLBACK = {"S0685": "S0684", "S0720": "S0721"}
ROW_MM = 500
COL_MM = 600


def load_bfs(data_dir):
    with open(os.path.join(data_dir, "bfs_nodes.json"), encoding="utf-8") as f:
        nodes = json.load(f)
    with open(os.path.join(data_dir, "bfs_dist.json"), encoding="utf-8") as f:
        dist_matrix = json.load(f)
    slot_to_nid = {}
    nid_to_pos = {}
    for n in nodes:
        nid_to_pos[n["id"]] = (n["row"], n["col"])
        if n["type"] == "slot_entry":
            for s in n.get("slots", []):
                slot_to_nid[s] = n["id"]
    for bad, good in SLOT_FALLBACK.items():
        if good in slot_to_nid:
            slot_to_nid[bad] = slot_to_nid[good]
    return dist_matrix, slot_to_nid, nid_to_pos


def get_dist(a, b, dist_matrix, nid_to_pos):
    d = dist_matrix.get(a, {}).get(b)
    if d is not None:
        return d
    d = dist_matrix.get(b, {}).get(a)
    if d is not None:
        return d
    if a in nid_to_pos and b in nid_to_pos:
        ra, ca = nid_to_pos[a]
        rb, cb = nid_to_pos[b]
        return abs(ra - rb) * ROW_MM + abs(ca - cb) * COL_MM
    return 999_999_999


def parse_expected_groups(data_dir, slot_to_nid):
    """Independently rebuild each (worker,bell) segment as a dict of carts, where each
    cart -> set of its distinct mappable slots (CART-INTERNAL dedup only). There is NO
    cross-cart dedup: the same slot on two carts is a distinct required visit on each
    cart. Only slots mappable to a BFS node are scored.
    Returns: {(worker, bell): {cart_id_str: set(slot_codes)}}."""
    import openpyxl
    xlsx = os.path.join(data_dir, "sorting_records.xlsx")
    wb = openpyxl.load_workbook(xlsx, read_only=True)
    ws = wb.active
    expected = {}
    for row in ws.iter_rows(min_row=2, values_only=True):
        _, bell, cart, buyer, slot_raw, worker = row
        if not slot_raw or not worker or not bell:
            continue
        # 分货员工代码为空或字面 "NULL" 的记录视为异常数据，过滤删除
        # （openpyxl 读入时空单元格为 None，另有少量记录字面值为字符串 "NULL"）
        if str(worker).strip().upper() == "NULL":
            continue
        m = VALID_SLOT_RE.match(str(slot_raw).strip())
        if not m:
            continue
        code = m.group(1)
        if code not in slot_to_nid:
            continue
        try:
            bell_int = int(bell)
        except (ValueError, TypeError):
            continue
        seg = expected.setdefault((str(worker), bell_int), {})
        seg.setdefault(str(cart), set()).add(code)
    wb.close()
    return expected


def load_baseline():
    with open(os.path.join(_HERE, "baseline", "reference_metrics.json")) as f:
        return float((lambda _d:_d.get("reference_value",_d.get("baseline_cost")))(json.load(f)))


def evaluate(file_path, data_dir):
    metrics = {"validity_score": 0.0, "quality_score": 0.0, "overall_score": 0.0, "error_info": {}}
    try:
        baseline_cost = load_baseline()
        dist_matrix, slot_to_nid, nid_to_pos = load_bfs(data_dir)
        expected_groups = parse_expected_groups(data_dir, slot_to_nid)

        if not os.path.exists(file_path):
            metrics["error_info"] = {"fatal": [f"File not found: {file_path}"]}
            return metrics
        with open(file_path, "r", encoding="utf-8") as f:
            sub = json.load(f)

        groups = sub.get("groups")
        if not isinstance(groups, list) or not groups:
            metrics["error_info"] = {"fatal": ["Missing or empty 'groups'"]}
            return metrics

        # index result by (worker, bell)
        result_index = {}
        for g in groups:
            try:
                key = (str(g.get("worker_id", "")), int(g.get("bell", 0)))
            except (ValueError, TypeError):
                metrics["error_info"] = {"schema": [f"bad worker_id/bell in group: {g.get('worker_id')}/{g.get('bell')}"]}
                return metrics
            result_index[key] = g

        violations = []
        total_dist_mm = 0

        # 多余段检查：result 不得包含 expected 之外的任务段。
        # 例如未过滤员工代码为空/NULL 的无主台车（这些段本应被删除），
        # 保留即视为产物不合规 -> validity=0。
        extra_segs = [k for k in result_index if k not in expected_groups]
        if extra_segs:
            violations.append(
                f"extra segment(s) not in expected (should be filtered, e.g. NULL worker): "
                f"{sorted(str(s) for s in extra_segs)[:5]}")

        for (worker, bell_int), expected_carts in expected_groups.items():
            g = result_index.get((worker, bell_int))
            if g is None:
                violations.append(f"missing segment worker={worker} bell={bell_int}")
                continue

            cart_order = [str(c) for c in g.get("cart_order", [])]
            slot_sequences = g.get("slot_sequences", {}) or {}
            bell_node = BELL_NODE_IDS.get(bell_int, f"{bell_int}号钟")

            # cart_order must be exactly the segment's carts, each once
            if len(cart_order) != len(set(cart_order)):
                violations.append(f"worker={worker} bell={bell_int}: repeated cart in cart_order")
            expected_cart_ids = set(expected_carts.keys())
            actual_cart_ids = set(cart_order)
            missing_carts = expected_cart_ids - actual_cart_ids
            extra_carts = actual_cart_ids - expected_cart_ids
            if missing_carts:
                violations.append(f"worker={worker} bell={bell_int}: missing carts {sorted(missing_carts)[:5]}")
            if extra_carts:
                violations.append(f"worker={worker} bell={bell_int}: unknown carts {sorted(extra_carts)[:5]}")

            # per-cart slot coverage (cart-internal dedup; NO cross-cart dedup)
            for cart in cart_order:
                if cart not in expected_carts:
                    continue
                exp_slots = expected_carts[cart]
                seq = [str(s) for s in slot_sequences.get(cart, [])]
                actual = set(seq)
                miss = exp_slots - actual
                extra = actual - exp_slots
                if miss:
                    violations.append(f"worker={worker} bell={bell_int} cart={cart}: missing slots {sorted(miss)[:5]}")
                if extra:
                    violations.append(f"worker={worker} bell={bell_int} cart={cart}: extra/cross-bell slots {sorted(extra)[:5]}")
                if len(seq) != len(actual):
                    violations.append(f"worker={worker} bell={bell_int} cart={cart}: repeated slot visit within cart")

            if violations:
                # short-circuit early once we have enough
                if len(violations) >= 20:
                    break
                continue

            # independent distance recompute: bell -> cart1 slots -> cart2 slots -> ...
            # each cart is a fresh leg starting from the previous cart's last slot
            # (no return to bell between carts or at the end). NO cross-cart dedup.
            current = bell_node
            for cart in cart_order:
                for slot in slot_sequences.get(cart, []):
                    nid = slot_to_nid.get(str(slot))
                    if nid is None:
                        continue
                    total_dist_mm += get_dist(current, nid, dist_matrix, nid_to_pos)
                    current = nid

        if violations:
            metrics["error_info"] = {"constraint": violations[:10]}
            return metrics

        metrics["validity_score"] = 1.0
        total_distance_m = round(total_dist_mm / 1000.0, 2)

        quality = baseline_cost / total_distance_m if total_distance_m > 0 else 0.0
        metrics["quality_score"] = round(quality, 4)
        metrics["overall_score"] = round(quality, 4)
        metrics["total_distance_m"] = total_distance_m
        metrics["baseline_cost"] = baseline_cost
        metrics["n_groups_expected"] = len(expected_groups)
        metrics["n_groups_result"] = len(groups)

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
