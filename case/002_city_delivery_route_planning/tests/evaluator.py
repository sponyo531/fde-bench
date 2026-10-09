"""
EXTRACTOR_SPEC:
  plan_file: routes.json
  required_columns: [routes]
  notes: >
    The agent may produce output in various formats — JSON or CSV.
    Normalize to: {"routes": [{"customers": ["id1", "id2", ...]}, ...]}.
    Each list entry = one vehicle's ordered customer sequence.
    Customer IDs must be strings matching 网点编码 in customers.csv.
    Exclude the depot node (the row where 总重量 is empty/NaN).

    Common agent output patterns to handle:
    - JSON with nested route objects: extract the customer ID list from each route
    - CSV with columns like (车号/route_id, 网点编码/customer_id, 访问序号/order):
      group by vehicle, sort by visit order, collect customer IDs in order
    - CSV with one row per vehicle listing all customer IDs in a single column:
      split the column value into a list

    Use Bash with python3 to transform data if needed, then write routes.json
    to the output directory.
"""

import json
import math
import os
import traceback
from pathlib import Path

import pandas as pd

PLAN_FILE = "routes.json"
REQUIRED_COLUMNS = ["routes"]

_HERE = os.path.dirname(os.path.abspath(__file__))
_DATA = os.path.join(_HERE, "..", "data", "customers.csv")

# --- routing parameters (business constants; not read from data/) ---
HIGHWAY_SPEED_KMH = 70.0   # depot <-> first/last customer
LOCAL_SPEED_KMH   = 35.0   # between consecutive customers
SERVICE_TIME_S    = 120.0  # seconds per customer stop
CAPACITY          = 7125.0
N_VEHICLES        = 12

# --- scoring weights (evaluator-private) ---
W_TIME     = 1.0
W_BALANCE  = 90.0
W_HULL     = 2.0
W_CROSSING = 3.0


# ── geometry helpers ─────────────────────────────────────────────────────────

def haversine_km(lon1, lat1, lon2, lat2):
    R = 6371.0
    dlat = math.radians(lat2 - lat1)
    dlon = math.radians(lon2 - lon1)
    a = math.sin(dlat / 2) ** 2 + math.cos(math.radians(lat1)) * math.cos(math.radians(lat2)) * math.sin(dlon / 2) ** 2
    return 2 * R * math.asin(math.sqrt(max(0.0, a)))


def route_duration_s(depot_pos, customer_positions):
    """Total time for one route: highway legs + local legs + service times."""
    if not customer_positions:
        return 0.0
    secs = 0.0
    # depot -> first customer (highway)
    secs += haversine_km(*depot_pos, *customer_positions[0]) / HIGHWAY_SPEED_KMH * 3600
    # between customers (local)
    for i in range(len(customer_positions) - 1):
        secs += haversine_km(*customer_positions[i], *customer_positions[i + 1]) / LOCAL_SPEED_KMH * 3600
    # last customer -> depot (highway)
    secs += haversine_km(*customer_positions[-1], *depot_pos) / HIGHWAY_SPEED_KMH * 3600
    # service time
    secs += len(customer_positions) * SERVICE_TIME_S
    return secs


# ── spatial metrics (hull intrusions + segment crossings) ────────────────────

def _cross(ox, oy, ax, ay, bx, by):
    return (ax - ox) * (by - oy) - (ay - oy) * (bx - ox)


def build_convex_hull(pts):
    if len(pts) < 3:
        return pts[:]
    pts = sorted(pts)
    hull = []
    for p in pts:
        while len(hull) >= 2 and _cross(*hull[-2], *hull[-1], *p) <= 0:
            hull.pop()
        hull.append(p)
    lower = len(hull)
    for p in reversed(pts[:-1]):
        while len(hull) > lower and _cross(*hull[-2], *hull[-1], *p) <= 0:
            hull.pop()
        hull.append(p)
    hull.pop()
    return hull


def point_in_hull(px, py, hull):
    if len(hull) < 3:
        return False
    n = len(hull)
    for i in range(n):
        j = (i + 1) % n
        if _cross(*hull[i], *hull[j], px, py) < 0:
            return False
    return True


def segments_intersect(p1, p2, p3, p4):
    d1 = _cross(*p3, *p4, *p1)
    d2 = _cross(*p3, *p4, *p2)
    d3 = _cross(*p1, *p2, *p3)
    d4 = _cross(*p1, *p2, *p4)
    return ((d1 > 0 and d2 < 0) or (d1 < 0 and d2 > 0)) and \
           ((d3 > 0 and d4 < 0) or (d3 < 0 and d4 > 0))


def calc_spatial_metrics(routes_pos, depot_pos):
    hulls = [build_convex_hull(pts) if len(pts) >= 3 else [] for pts in routes_pos]
    intrusions = 0
    for i, hi in enumerate(hulls):
        if len(hi) < 3:
            continue
        for j, pts_j in enumerate(routes_pos):
            if i == j:
                continue
            for px, py in pts_j:
                if point_in_hull(px, py, hi):
                    intrusions += 1

    segments = []
    for ri, pts in enumerate(routes_pos):
        path = [depot_pos] + list(pts) + [depot_pos]
        for k in range(len(path) - 1):
            segments.append((ri, path[k], path[k + 1]))
    crossings = 0
    for i in range(len(segments)):
        for j in range(i + 1, len(segments)):
            if segments[i][0] == segments[j][0]:
                continue
            if segments_intersect(segments[i][1], segments[i][2], segments[j][1], segments[j][2]):
                crossings += 1
    return intrusions, crossings


# ── combined cost ─────────────────────────────────────────────────────────────

def combined_cost(durations, routes_pos, depot_pos):
    if not durations:
        return float("inf")
    total_dur = sum(durations)
    max_d, min_d = max(durations), min(durations)
    balance_penalty = (max_d - min_d) * W_BALANCE
    intrusions, crossings = calc_spatial_metrics(routes_pos, depot_pos)
    spatial_penalty = intrusions * 300.0 * W_HULL + crossings * 600.0 * W_CROSSING
    cost = total_dur * W_TIME + balance_penalty + spatial_penalty
    return cost, total_dur, balance_penalty, intrusions, crossings, spatial_penalty


# ── main evaluate ─────────────────────────────────────────────────────────────

def evaluate(data_dir: str, baseline_dir: str = None, submission_dir: str = "."):
    metrics = {
        "validity_score": 0.0,
        "quality_score": 0.0,
        "overall_score": 0.0,
        "error_info": {},
    }

    try:
        # --- load customer data ---
        data_path = str(Path(data_dir) / "customers.csv")
        df = pd.read_csv(data_path, encoding="utf-8")
        depot_row = df[df["总重量"].isna()].iloc[0]
        depot_id = str(depot_row["网点编码"]).strip()
        depot_pos = (depot_row["门店经度"], depot_row["门店纬度"])

        cust_df = df[df["总重量"].notna()].copy()
        cust_df["_id"] = cust_df["网点编码"].astype(str).str.strip()
        pos_map   = {r["_id"]: (r["门店经度"], r["门店纬度"]) for _, r in cust_df.iterrows()}
        pos_map[depot_id] = depot_pos
        weight_map = {r["_id"]: float(r["总重量"]) for _, r in cust_df.iterrows()}
        all_customer_ids = set(weight_map.keys())

        # --- load submission ---
        file_path = str(Path(submission_dir) / PLAN_FILE)
        if not os.path.exists(file_path):
            metrics["error_info"] = {"fatal": [f"File not found: {file_path}"]}
            return metrics

        with open(file_path, "r", encoding="utf-8") as f:
            submission = json.load(f)

        if "routes" not in submission or not isinstance(submission["routes"], list):
            metrics["error_info"] = {"fatal": ["Missing or invalid 'routes' key in JSON"]}
            return metrics

        routes = submission["routes"]

        # --- extract customer lists ---
        route_customers = []
        for i, r in enumerate(routes):
            if isinstance(r, dict):
                custs = [str(c).strip() for c in r.get("customers", [])]
            elif isinstance(r, list):
                custs = [str(c).strip() for c in r]
            else:
                metrics["error_info"] = {"fatal": [f"Route {i} has unexpected type {type(r)}"]}
                return metrics
            route_customers.append(custs)

        # --- validate coverage ---
        all_submitted = []
        for custs in route_customers:
            all_submitted.extend(custs)
        submitted_set = set(all_submitted)

        # gt 硬规则: 仓库编码不能出现在任何路线的序列里。depot_id 不是客户,
        # 一旦被写进路线就是非法ID, 不能从 unknown 里剔掉(旧实现会放过单次仓库出现在路线)。
        unknown = submitted_set - all_customer_ids
        missing = all_customer_ids - submitted_set
        duplicates = [cid for cid in all_submitted if all_submitted.count(cid) > 1]
        duplicates = list(set(duplicates))

        schema_errors = []
        if unknown:
            schema_errors.append(f"{len(unknown)} unknown customer IDs")
        if missing:
            schema_errors.append(f"{len(missing)} customers not covered")
        if duplicates:
            schema_errors.append(f"{len(duplicates)} duplicate customer IDs")
        if schema_errors:
            metrics["error_info"] = {"schema": schema_errors}
            return metrics

        # --- validate capacity ---
        value_errors = []
        for i, custs in enumerate(route_customers):
            load = sum(weight_map.get(c, 0) for c in custs)
            if load > CAPACITY:
                value_errors.append(f"Route {i}: load {load:.1f} > capacity {CAPACITY}")
        if value_errors:
            metrics["error_info"] = {"value": value_errors}
            return metrics

        # --- validate vehicle count ---
        if len(route_customers) != N_VEHICLES:
            metrics["error_info"] = {"value": [
                f"Must use exactly {N_VEHICLES} vehicles, got {len(route_customers)}"
            ]}
            return metrics

        # --- validity passed ---
        metrics["validity_score"] = 1.0

        # --- compute submission metrics ---
        sub_durations = []
        sub_routes_pos = []
        for custs in route_customers:
            cpos = [pos_map[c] for c in custs if c in pos_map]
            sub_routes_pos.append(cpos)
            sub_durations.append(route_duration_s(depot_pos, cpos))

        sub_cost, total_dur, bal_pen, intrusions, crossings, spatial_pen = \
            combined_cost(sub_durations, sub_routes_pos, depot_pos)

        # --- load reference ---
        if baseline_dir is not None:
            _baseline_dir = Path(baseline_dir)
        else:
            _baseline_dir = Path(_HERE) / "baseline"
        with open(_baseline_dir / "reference_metrics.json", "r", encoding="utf-8") as _f:
            _ref = json.load(_f)
        ref_cost = float(_ref["reference_value"])

        quality = ref_cost / sub_cost if sub_cost > 0 else 0.0
        metrics["quality_score"] = round(quality, 4)
        metrics["overall_score"] = round(metrics["validity_score"] * quality, 4)

        # diagnostics
        metrics["total_duration_s"] = round(total_dur)
        metrics["balance_penalty"]  = round(bal_pen)
        metrics["hull_intrusions"]  = intrusions
        metrics["segment_crossings"] = crossings
        metrics["spatial_penalty"]  = round(spatial_pen)
        metrics["n_routes"]         = len(route_customers)
        metrics["reference_cost"]   = round(ref_cost)
        metrics["submission_cost"]  = round(sub_cost)

    except Exception as e:
        metrics["error_info"] = {"exception": str(e), "traceback": traceback.format_exc()}

    return metrics


def main():
    import argparse
    from pathlib import Path
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-dir", type=str, required=True)
    parser.add_argument("--baseline-dir", type=str, default=None)
    parser.add_argument("--submission-dir", type=str, default=".")
    args = parser.parse_args()

    result = evaluate(
        data_dir=args.data_dir,
        baseline_dir=args.baseline_dir,
        submission_dir=args.submission_dir,
    )
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
