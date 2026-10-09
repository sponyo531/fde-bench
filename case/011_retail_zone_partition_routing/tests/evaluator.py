"""
EXTRACTOR_SPEC:
  files:
    - name: partition_result.json
      required_columns: [zones]
      column_formats:
        zones: list of {zone_id (int), retailer_ids (list[int]), vehicle_count (int)}
      notes: >
        Step 1 output — zoning plan. Schema:
          {
            "zones": [
              {
                "zone_id": <int>,             # 5 zones total (ids unique)
                "retailer_ids": [<int>, ...], # matches `idx` column in demo_data.xlsx
                "vehicle_count": <int>
              },
              ... (exactly 5 entries)
            ]
          }

    - name: routes_result.json
      required_columns: [zones]
      column_formats:
        zones: list of {zone_id (int), routes (list of {route_id, sequence, trips, scheduled_trip_count?})}
      notes: >
        Step 2 output — routing plan. Schema:
          {
            "zones": [
              {
                "zone_id": <int>,
                "routes": [
                  {
                    "route_id": <int>,              # unique within the zone
                    "sequence": [<int>, ...],       # ordered retailer idx visited by this route
                    "trips": [
                      {"trip_id": <int|str>, "num_stops": <int>, "load": <float>},
                      ...
                    ],
                    "scheduled_trip_count": <int>   # optional; fallback = len(trips)
                  },
                  ... (exactly 5 entries)
                ]
              },
              ... (5 entries, one per zone)
            ]
          }
        If the agent stored the per-stop sequence in a companion CSV (columns like
        zone/route/trip/order/retailer_id), sort by (trip, order) within each
        (zone, route) group and lift the retailer_id column back into
        route.sequence at the route level. Do not execute agent code, do not alter
        retailer ids or trip loads.
  notes: >
    Retailer ids in both files must match `idx` in `data/demo_data.xlsx`. Retailers
    with `average_quantity <= 0` are excluded and must not appear in any zone or
    route. Zone ids must be consistent between the two files. Within a zone the
    `scheduled_trip_count` must be equal across the 5 routes.
"""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
from typing import Any

import openpyxl
from shapely import concave_hull, STRtree
from shapely.geometry import LineString, MultiPoint
from shapely.geometry.base import BaseGeometry
from shapely.ops import unary_union


# ============================================================================
# 业务参数（与 instruction.md 一致；不依赖外部 yaml）
# ============================================================================

WAREHOUSE_LON = 126.009988
WAREHOUSE_LAT = 27.979015
VEHICLE_CAPACITY = 8000.0          # 单车装载条数（4 米 2 黄牌）
SPEED_KMH = 30.0                   # 平均行驶速度
TIME_OVERHEAD_RATIO = 0.2          # 20% 时间扰动
SERVICE_TIME_PER_STOP_MIN = 1.0    # 每户停车交接时间

NUM_ZONES = 5                      # 片区数 M
NUM_ROUTES_PER_ZONE = 5            # 每片区路线数 N
MAX_NODE_RATIO = 1.2               # 单片区最大网点数 = 平均值 × 1.2

EARTH_RADIUS_KM = 6371.0
CONCAVE_HULL_RATIO = 0.0           # 最紧凹包

# 归一化上限
STEP1_COMPACTNESS_MAX_KM = 20.0
STEP2_TOTAL_TIME_MAX_MIN = 50000.0

# 步骤一 5 项目标权重相当
W_STEP1 = {
    "node_balance":     0.20,
    "time_balance":     0.20,
    "compactness":      0.20,
    "non_overlap":      0.20,
    "load_match":       0.20,
}

# 步骤二 5 项目标权重（total_time 主导，其余四项等权）
W_STEP2 = {
    "total_time":       0.40,
    "node_balance":     0.15,
    "time_balance":     0.15,
    "non_crossing":     0.15,
    "load_factor":      0.15,
}

# 步骤一与步骤二在最终 quality_score 中同等重要
W_STEP1_VS_STEP2 = (0.5, 0.5)


# ============================================================================
# 几何与时间工具（从 scripts/compute_exact_metrics.py 复制，避免 import 耦合）
# ============================================================================

def haversine_km(lon1: float, lat1: float, lon2: float, lat2: float) -> float:
    dlon = math.radians(lon2 - lon1)
    dlat = math.radians(lat2 - lat1)
    lat1r = math.radians(lat1)
    lat2r = math.radians(lat2)
    a = math.sin(dlat / 2.0) ** 2 + math.cos(lat1r) * math.cos(lat2r) * math.sin(dlon / 2.0) ** 2
    return EARTH_RADIUS_KM * 2.0 * math.atan2(math.sqrt(a), math.sqrt(1.0 - a))


def travel_time_min(dist_km: float) -> float:
    return dist_km / SPEED_KMH * 60.0 * (1.0 + TIME_OVERHEAD_RATIO)


def make_local_projector(points_lon_lat: list[tuple[float, float]]):
    if not points_lon_lat:
        center_lon, center_lat = 0.0, 0.0
    else:
        center_lon = sum(lon for lon, _ in points_lon_lat) / len(points_lon_lat)
        center_lat = sum(lat for _, lat in points_lon_lat) / len(points_lon_lat)
    lat_scale = 111_320.0
    lon_scale = lat_scale * math.cos(math.radians(center_lat))

    def project(lon: float, lat: float) -> tuple[float, float]:
        return (lon - center_lon) * lon_scale, (lat - center_lat) * lat_scale

    return project


def build_zone_concave_hull(zone_points_xy: list[tuple[float, float]]) -> BaseGeometry:
    if not zone_points_xy:
        return MultiPoint([])
    if len(zone_points_xy) == 1:
        return MultiPoint(zone_points_xy)
    if len(zone_points_xy) == 2:
        return LineString(zone_points_xy)
    geom = concave_hull(MultiPoint(zone_points_xy), ratio=CONCAVE_HULL_RATIO, allow_holes=False)
    if geom.area <= 0:
        geom = MultiPoint(zone_points_xy).convex_hull
    return geom


def route_distance_and_duration(
    sequence: list[int],
    retailer_map: dict[int, dict[str, Any]],
) -> tuple[float, float]:
    """单条路线总距离/时间口径：仓库 → 全部零售户 → 仓库（一辆无限载荷的车）"""
    if not sequence:
        return 0.0, 0.0
    total_dist = 0.0
    prev_lon, prev_lat = WAREHOUSE_LON, WAREHOUSE_LAT
    for rid in sequence:
        r = retailer_map[rid]
        total_dist += haversine_km(prev_lon, prev_lat, r["lon"], r["lat"])
        prev_lon, prev_lat = r["lon"], r["lat"]
    total_dist += haversine_km(prev_lon, prev_lat, WAREHOUSE_LON, WAREHOUSE_LAT)
    total_duration = travel_time_min(total_dist) + len(sequence) * SERVICE_TIME_PER_STOP_MIN
    return total_dist, total_duration


def build_route_non_depot_lines(sequence: list[int], retailer_map: dict[int, dict[str, Any]], project) -> list[LineString]:
    """路线中不含仓库往返的线段（用于不交叉性计算）"""
    if len(sequence) < 2:
        return []
    lines = []
    for i in range(len(sequence) - 1):
        a = retailer_map[sequence[i]]
        b = retailer_map[sequence[i + 1]]
        p1 = project(a["lon"], a["lat"])
        p2 = project(b["lon"], b["lat"])
        if p1 != p2:
            lines.append(LineString([p1, p2]))
    return lines


def compute_route_last_trip_load_factor(
    sequence: list[int],
    retailer_map: dict[int, dict[str, Any]],
) -> tuple[float, int]:
    """按客户序列贪心装车，返回(最后一辆车满载率, 车次数)。同户不可拆分。"""
    if not sequence:
        return 0.0, 0
    required_trip_count = 0
    current_trip_load = 0.0
    last_trip_load = 0.0
    for rid in sequence:
        demand = retailer_map[rid]["quantity"]
        if math.isclose(current_trip_load, 0.0):
            required_trip_count += 1
            last_trip_load = 0.0
        remaining = VEHICLE_CAPACITY - current_trip_load
        if demand > remaining and not math.isclose(current_trip_load, 0.0):
            current_trip_load = 0.0
            required_trip_count += 1
            last_trip_load = 0.0
        current_trip_load += demand
        last_trip_load += demand
        if math.isclose(current_trip_load, VEHICLE_CAPACITY):
            current_trip_load = 0.0
    last_factor = 1.0 if math.isclose(last_trip_load, 0.0) else last_trip_load / VEHICLE_CAPACITY
    return last_factor, required_trip_count


def max_deviation_from_mean(values: list[float]) -> float:
    if not values:
        return 0.0
    mean = sum(values) / len(values)
    return max(abs(v - mean) for v in values)


# ============================================================================
# 数据加载
# ============================================================================

def load_retailers_from_xlsx(xlsx_path: Path) -> dict[int, dict[str, Any]]:
    """从 demo_data.xlsx 加载零售户。跳过 average_quantity <= 0 的户。"""
    wb = openpyxl.load_workbook(xlsx_path, read_only=True)
    ws = wb.active
    retailers: dict[int, dict[str, Any]] = {}
    for row in ws.iter_rows(min_row=2, values_only=True):
        idx, level, lon, lat, qty = row[:5]
        try:
            qty_f = float(qty) if qty is not None else 0.0
        except (TypeError, ValueError):
            qty_f = 0.0
        if qty_f <= 0:
            continue
        retailers[int(idx)] = {
            "id": int(idx),
            "level": str(level) if level is not None else "",
            "lon": float(lon),
            "lat": float(lat),
            "quantity": qty_f,
        }
    wb.close()
    return retailers


def load_submission(submission_dir: Path) -> tuple[dict[str, Any], dict[str, Any]]:
    part_path = submission_dir / "partition_result.json"
    routes_path = submission_dir / "routes_result.json"
    if not part_path.exists():
        raise FileNotFoundError(f"missing partition_result.json in {submission_dir}")
    if not routes_path.exists():
        raise FileNotFoundError(f"missing routes_result.json in {submission_dir}")
    with part_path.open() as f:
        part = json.load(f)
    with routes_path.open() as f:
        routes = json.load(f)
    return part, routes


# ============================================================================
# Validity 校验
# ============================================================================

def _validate_solution(
    part: dict[str, Any],
    routes_data: dict[str, Any],
    retailer_map: dict[int, dict[str, Any]],
) -> list[str]:
    """
    硬约束统一校验。

    与原 compute_exact_metrics.py 配套的校验函数同源；适配点：
      - 入参从 (inp, part, routes_data) 改为 (part, routes_data, retailer_map)，
        因为评估器不再读 input.json，retailer_map 由 demo_data.xlsx 直读得到。
      - num_zones / num_routes / vehicle_capacity 改用顶部常量。
    每条违规生成一条错误信息，全部收集后一次返回。
    """
    violations: list[str] = []

    # 从输入中提取校验所需的基准约束
    positive_ids = set(retailer_map.keys())
    num_zones = NUM_ZONES
    num_routes = NUM_ROUTES_PER_ZONE
    vehicle_capacity = float(VEHICLE_CAPACITY)
    max_nodes_per_zone = int(math.ceil(len(positive_ids) / num_zones * MAX_NODE_RATIO))

    # 校验 partition_result 顶层结构与分区数量
    zones = part.get("zones")
    if not isinstance(zones, list):
        return ["partition_result.json missing top-level 'zones' list"]
    if len(zones) != num_zones:
        violations.append(f"partition zone count mismatch: expected {num_zones}, got {len(zones)}")

    zone_id_to_partition_ids: dict[int, set[int]] = {}
    partition_seen: list[int] = []
    # 逐区校验分区方案：
    # 1) zone_id 不能重复
    # 2) 每个分区不能为空
    # 3) 每个分区零售商数量不能超过 Q
    # 4) 记录每分区零售商，供后续路由结果做跨区一致性检查
    for zone in zones:
        zone_id = zone.get("zone_id")
        retailer_ids = zone.get("retailer_ids", [])
        if zone_id in zone_id_to_partition_ids:
            violations.append(f"duplicate partition zone_id: {zone_id}")
            continue
        if not retailer_ids:
            violations.append(f"partition zone {zone_id} is empty")
        if len(retailer_ids) > max_nodes_per_zone:
            violations.append(
                f"partition zone {zone_id} exceeds Q limit: {len(retailer_ids)} > {max_nodes_per_zone}"
            )
        zone_id_to_partition_ids[zone_id] = set(retailer_ids)
        partition_seen.extend(retailer_ids)

    partition_set = set(partition_seen)
    # 检查分区结果对正需求零售商的覆盖是否完整且唯一
    if partition_set != positive_ids:
        missing = sorted(positive_ids - partition_set)
        extra = sorted(partition_set - positive_ids)
        if missing:
            violations.append(f"partition missing positive-demand retailers: {missing[:10]}")
        if extra:
            violations.append(f"partition contains invalid retailers: {extra[:10]}")
    if len(partition_seen) != len(partition_set):
        violations.append("partition assigns some retailers more than once")

    # 校验 routes_result 顶层结构
    route_zones = routes_data.get("zones")
    if not isinstance(route_zones, list):
        return violations + ["routes_result.json missing top-level 'zones' list"]
    if len(route_zones) != num_zones:
        violations.append(f"routing zone count mismatch: expected {num_zones}, got {len(route_zones)}")

    routing_seen: list[int] = []
    route_zone_ids: set[int] = set()
    # 逐区校验路由方案是否建立在合法分区之上，并满足每区固定路线数要求
    for zone in route_zones:
        zone_id = zone.get("zone_id")
        route_zone_ids.add(zone_id)
        routes = zone.get("routes", [])
        if zone_id not in zone_id_to_partition_ids:
            violations.append(f"routing zone {zone_id} not found in partition")
            partition_ids: set[int] = set()
        else:
            partition_ids = zone_id_to_partition_ids[zone_id]

        if len(routes) != num_routes:
            violations.append(
                f"routing zone {zone_id} route count mismatch: expected {num_routes}, got {len(routes)}"
            )

        scheduled_trip_counts: list[int] = []
        route_ids_seen: set[int] = set()
        # 逐条路线校验
        for route in routes:
            route_id = route.get("route_id")
            if route_id in route_ids_seen:
                violations.append(f"routing zone {zone_id} has duplicate route_id {route_id}")
            route_ids_seen.add(route_id)

            sequence = route.get("sequence", [])
            routing_seen.extend(sequence)

            invalid_ids = [rid for rid in sequence if rid not in positive_ids]
            if invalid_ids:
                violations.append(
                    f"routing zone {zone_id} route {route_id} contains invalid retailers: {invalid_ids[:10]}"
                )
            cross_zone = [rid for rid in sequence if rid not in partition_ids]
            if cross_zone:
                violations.append(
                    f"routing zone {zone_id} route {route_id} contains retailers outside partition: {cross_zone[:10]}"
                )

            trips = route.get("trips", [])
            scheduled_trip_count = route.get("scheduled_trip_count")
            if scheduled_trip_count is None:
                scheduled_trip_count = len(trips)
            scheduled_trip_counts.append(int(scheduled_trip_count))

            # 逐 trip 校验：有停靠点的 trip 不得超载
            for trip in trips:
                num_stops = int(trip.get("num_stops", 0))
                load = float(trip.get("load", 0.0))
                if num_stops > 0 and load > vehicle_capacity + 1e-6:
                    violations.append(
                        f"routing zone {zone_id} route {route_id} trip {trip.get('trip_id')} overload: "
                        f"{load:.4f} > {vehicle_capacity:.4f}"
                    )

        # 同区内所有路线的 scheduled_trip_count 应保持一致
        if scheduled_trip_counts and len(set(scheduled_trip_counts)) != 1:
            violations.append(
                f"routing zone {zone_id} scheduled_trip_count not equal across routes: {scheduled_trip_counts}"
            )

    routing_set = set(routing_seen)
    # 检查路由对正需求零售商的覆盖
    if routing_set != positive_ids:
        missing = sorted(positive_ids - routing_set)
        extra = sorted(routing_set - positive_ids)
        if missing:
            violations.append(f"routing missing positive-demand retailers: {missing[:10]}")
        if extra:
            violations.append(f"routing contains invalid retailers: {extra[:10]}")
    if len(routing_seen) != len(routing_set):
        violations.append("routing assigns some retailers more than once")

    # routing 与 partition 的 zone_id 集合一致性
    if route_zone_ids != set(zone_id_to_partition_ids.keys()):
        missing_zones = sorted(set(zone_id_to_partition_ids.keys()) - route_zone_ids)
        extra_zones = sorted(route_zone_ids - set(zone_id_to_partition_ids.keys()))
        if missing_zones:
            violations.append(f"routing missing zones from partition: {missing_zones}")
        if extra_zones:
            violations.append(f"routing has extra zones not in partition: {extra_zones}")

    return violations


# ============================================================================
# 步骤一指标
# ============================================================================

def compute_step1_metrics(
    part: dict[str, Any],
    retailer_map: dict[int, dict[str, Any]],
) -> dict[str, Any]:
    zones = sorted(part["zones"], key=lambda z: z["zone_id"])
    all_points = [(r["lon"], r["lat"]) for r in retailer_map.values()]
    project = make_local_projector(all_points + [(WAREHOUSE_LON, WAREHOUSE_LAT)])

    zone_details: list[dict[str, Any]] = []
    hulls: dict[int, BaseGeometry] = {}

    for zone in zones:
        zid = zone["zone_id"]
        rids = [r for r in zone.get("retailer_ids", []) if r in retailer_map]
        members = [retailer_map[r] for r in rids]
        n = len(members)
        total_demand = sum(m["quantity"] for m in members)

        dists = [haversine_km(m["lon"], m["lat"], WAREHOUSE_LON, WAREHOUSE_LAT) for m in members]
        total_dist = sum(dists)
        total_time = sum(travel_time_min(d) for d in dists)

        if members:
            cx = sum(m["lon"] for m in members) / n
            cy = sum(m["lat"] for m in members) / n
            compact = sum(haversine_km(m["lon"], m["lat"], cx, cy) for m in members) / n
        else:
            compact = 0.0

        # 步骤一满载率：(总订货量 % 8000) / 8000，越接近 1.0 或刚好 0 越好
        remainder = total_demand % VEHICLE_CAPACITY if VEHICLE_CAPACITY > 0 else 0.0
        load_match = 1.0 if math.isclose(remainder, 0.0) else remainder / VEHICLE_CAPACITY

        # 凹包
        zone_xy = [project(m["lon"], m["lat"]) for m in members]
        hull = build_zone_concave_hull(zone_xy)
        hulls[zid] = hull

        zone_details.append({
            "zone_id": zid,
            "vehicle_count": zone.get("vehicle_count"),
            "num_retailers": n,
            "total_demand": total_demand,
            "total_dist_to_warehouse_km": total_dist,
            "total_time_to_warehouse_min": total_time,
            "compactness_avg_distance_km": compact,
            "load_match_score": load_match,
            "concave_hull_area_m2": hull.area if hasattr(hull, "area") else 0.0,
        })

    # 不重叠性：每个片区与其他所有片区凹包的相交几何，取并集求面积
    zone_overlap_geoms: dict[int, list] = {z["zone_id"]: [] for z in zone_details}
    for i, za in enumerate(zone_details):
        for zb in zone_details[i + 1:]:
            ha, hb = hulls[za["zone_id"]], hulls[zb["zone_id"]]
            if ha.is_empty or hb.is_empty:
                continue
            inter = ha.intersection(hb)
            if not inter.is_empty:
                zone_overlap_geoms[za["zone_id"]].append(inter)
                zone_overlap_geoms[zb["zone_id"]].append(inter)

    overlap_ratios = []
    for zd in zone_details:
        zid = zd["zone_id"]
        zone_area = zd["concave_hull_area_m2"]
        geoms = zone_overlap_geoms[zid]
        overlap_area = unary_union(geoms).area if geoms else 0.0
        ratio = overlap_area / zone_area if zone_area > 0 else 0.0
        zd["overlap_ratio"] = ratio
        overlap_ratios.append(ratio)

    counts = [z["num_retailers"] for z in zone_details]
    times = [z["total_time_to_warehouse_min"] for z in zone_details]
    compactness_values = [z["compactness_avg_distance_km"] for z in zone_details]
    load_match_values = [z["load_match_score"] for z in zone_details]

    return {
        "zones": zone_details,
        "node_balance_max_deviation": max_deviation_from_mean(counts),
        "node_avg": sum(counts) / len(counts) if counts else 0,
        "time_balance_max_deviation_min": max_deviation_from_mean(times),
        "time_avg_min": sum(times) / len(times) if times else 0,
        "compactness_overall_avg_km": sum(compactness_values) / len(compactness_values) if compactness_values else 0,
        "non_overlap_avg_ratio": sum(overlap_ratios) / len(overlap_ratios) if overlap_ratios else 0,
        "load_match_score_avg": sum(load_match_values) / len(load_match_values) if load_match_values else 0,
    }


# ============================================================================
# 步骤二指标
# ============================================================================

def compute_step2_metrics(
    routes_data: dict[str, Any],
    retailer_map: dict[int, dict[str, Any]],
) -> dict[str, Any]:
    all_points = [(r["lon"], r["lat"]) for r in retailer_map.values()]
    project = make_local_projector(all_points + [(WAREHOUSE_LON, WAREHOUSE_LAT)])

    zone_results: list[dict[str, Any]] = []

    for zone in sorted(routes_data["zones"], key=lambda z: z["zone_id"]):
        route_metrics: list[dict[str, Any]] = []
        route_lines: dict[int, list[LineString]] = {}

        for route in sorted(zone.get("routes", []), key=lambda r: r["route_id"]):
            seq = [r for r in route.get("sequence", []) if r in retailer_map]
            dist_km, dur_min = route_distance_and_duration(seq, retailer_map)
            load_factor, _ = compute_route_last_trip_load_factor(seq, retailer_map)
            segs = build_route_non_depot_lines(seq, retailer_map, project)
            route_lines[route["route_id"]] = segs
            route_metrics.append({
                "route_id": route["route_id"],
                "num_retailers": len(seq),
                "tour_distance_km": dist_km,
                "tour_duration_min": dur_min,
                "route_load_factor": load_factor,
            })

        # 不交叉性（线段级，STRtree 加速）
        flat = [(rm["route_id"], seg) for rm in route_metrics for seg in route_lines[rm["route_id"]]]
        total_segs = len(flat)
        crossing_seg = 0
        if total_segs > 0:
            geoms = [s for _, s in flat]
            rids = [rid for rid, _ in flat]
            tree = STRtree(geoms)
            for i, (rid_a, seg_a) in enumerate(flat):
                for j in tree.query(seg_a, predicate="intersects"):
                    if j != i and rids[j] != rid_a:
                        crossing_seg += 1
                        break
        non_crossing_ratio = crossing_seg / total_segs if total_segs > 0 else 0.0

        counts = [r["num_retailers"] for r in route_metrics]
        durations = [r["tour_duration_min"] for r in route_metrics]
        load_factors = [r["route_load_factor"] for r in route_metrics]

        zone_results.append({
            "zone_id": zone["zone_id"],
            "routes": route_metrics,
            "total_time_min": sum(durations),
            "node_balance_max_deviation": max_deviation_from_mean(counts),
            "node_avg": sum(counts) / len(counts) if counts else 0,
            "time_balance_max_deviation_min": max_deviation_from_mean(durations),
            "time_avg_min": sum(durations) / len(durations) if durations else 0,
            "non_crossing_segment_ratio": non_crossing_ratio,
            "route_load_factor_avg": sum(load_factors) / len(load_factors) if load_factors else 0,
        })

    return {"zones": zone_results}


# ============================================================================
# Quality Score 合成
# ============================================================================

def normalize_smaller_better(value: float, scale: float) -> float:
    """value 越小越好 → 映射到 [0, 1]，1 表示最优。线性截断在 scale 处归零。"""
    if scale <= 0:
        return 1.0
    return max(0.0, 1.0 - value / scale)


def compose_quality_score(
    step1: dict[str, Any],
    step2: dict[str, Any],
) -> dict[str, Any]:
    """
    把步骤一/步骤二指标合成 quality_score ∈ [0, 1]。

    归一化基准（内嵌于评估器）：
      - 步骤一网点数偏差上限 = 均值的 30%
      - 步骤一时间偏差上限 = 均值的 30%
      - 步骤一聚集均距上限 = STEP1_COMPACTNESS_MAX_KM
      - 步骤一不重叠率上限 = 0.5
      - 步骤一满载率：直接用 load_match_score（[0, 1]）
      - 步骤二总时间上限 = STEP2_TOTAL_TIME_MAX_MIN
      - 步骤二网点数偏差上限 = 均值的 30%
      - 步骤二时间偏差上限 = 均值的 30%
      - 步骤二不交叉率上限 = 1.0
      - 步骤二满载率：直接用
    """
    # ---- 步骤一各项分（每项 ∈ [0, 1]）----
    s1_node_avg = step1["node_avg"] or 1
    s1_time_avg = step1["time_avg_min"] or 1
    s1 = {
        "node_balance":  normalize_smaller_better(step1["node_balance_max_deviation"], s1_node_avg * 0.30),
        "time_balance":  normalize_smaller_better(step1["time_balance_max_deviation_min"], s1_time_avg * 0.30),
        "compactness":   normalize_smaller_better(step1["compactness_overall_avg_km"], STEP1_COMPACTNESS_MAX_KM),
        "non_overlap":   normalize_smaller_better(step1["non_overlap_avg_ratio"], 0.5),
        "load_match":    step1["load_match_score_avg"],   # 已在 [0, 1]
    }
    step1_score = sum(W_STEP1[k] * s1[k] for k in W_STEP1)

    # ---- 步骤二（按 zone 计 5 项，再对 zone 取均值）----
    zone_scores = []
    for z in step2["zones"]:
        node_avg = z["node_avg"] or 1
        time_avg = z["time_avg_min"] or 1
        s2 = {
            "total_time":   normalize_smaller_better(z["total_time_min"], STEP2_TOTAL_TIME_MAX_MIN),
            "node_balance": normalize_smaller_better(z["node_balance_max_deviation"], node_avg * 0.30),
            "time_balance": normalize_smaller_better(z["time_balance_max_deviation_min"], time_avg * 0.30),
            "non_crossing": normalize_smaller_better(z["non_crossing_segment_ratio"], 1.0),
            "load_factor":  z["route_load_factor_avg"],
        }
        zone_scores.append(sum(W_STEP2[k] * s2[k] for k in W_STEP2))
    step2_score = sum(zone_scores) / len(zone_scores) if zone_scores else 0.0

    quality = W_STEP1_VS_STEP2[0] * step1_score + W_STEP1_VS_STEP2[1] * step2_score
    return {
        "quality_score": quality,
        "step1_score": step1_score,
        "step2_score": step2_score,
        "step2_zone_scores": zone_scores,
    }


# ============================================================================
# Bench 标准入口
# ============================================================================

def evaluate(data_dir: str, baseline_dir: str | None = None, submission_dir: str = ".") -> dict[str, Any]:
    """
    Bench 标准评估接口。

    Args:
        data_dir: 含 demo_data.xlsx 的原始数据目录
        baseline_dir: 保留参数（当前评估器不再读 baseline，所有归一化上限内嵌为常量）
        submission_dir: 含 partition_result.json + routes_result.json 的提交目录

    Returns:
        {
            "validity_score": 0.0 or 1.0,
            "quality_score": float in [0, 1],
            "overall_score": validity * quality,
            "errors": list[str],
            "step1_metrics": {...},
            "step2_metrics": {...},
        }
    """
    data_dir_p = Path(data_dir)
    sub_dir_p = Path(submission_dir)

    result: dict[str, Any] = {
        "validity_score": 0.0,
        "quality_score": 0.0,
        "overall_score": 0.0,
        "errors": [],
    }

    # 1. 加载原始数据
    try:
        retailer_map = load_retailers_from_xlsx(data_dir_p / "demo_data.xlsx")
    except Exception as e:
        result["errors"].append(f"load_retailers failed: {e}")
        return result

    # 2. 加载提交
    try:
        part, routes = load_submission(sub_dir_p)
    except Exception as e:
        result["errors"].append(str(e))
        return result

    # 3. Validity 校验（partition + routes 一次过）
    errs = _validate_solution(part, routes, retailer_map)
    if errs:
        result["errors"] = errs
        return result

    # 4. 计算指标
    try:
        step1 = compute_step1_metrics(part, retailer_map)
        step2 = compute_step2_metrics(routes, retailer_map)
    except Exception as e:
        result["errors"].append(f"metric computation failed: {e}")
        return result

    # 5. 合成 quality_score
    score_info = compose_quality_score(step1, step2)

    # higher_is_better: quality = 选手绝对分 / baseline，不加 min(,1) 截断
    with open(Path(__file__).resolve().parent / "baseline" / "reference_metrics.json",
              encoding="utf-8") as _bf:
        _baseline = float(json.load(_bf)["reference_value"])
    _abs_score = score_info["quality_score"]
    _quality_norm = _abs_score / _baseline if _baseline > 0 else 0.0

    result["validity_score"] = 1.0
    result["quality_score"] = round(_quality_norm, 6)
    result["overall_score"] = round(result["validity_score"] * _quality_norm, 6)
    result["player_absolute_score"] = _abs_score
    result["reference_value"] = _baseline
    result["step1_score"] = score_info["step1_score"]
    result["step2_score"] = score_info["step2_score"]
    result["step1_metrics"] = step1
    result["step2_metrics"] = step2
    return result


# ============================================================================
# CLI
# ============================================================================

def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-dir", required=True)
    parser.add_argument("--baseline-dir", default=None)
    parser.add_argument("--submission-dir", required=True)
    args = parser.parse_args()

    res = evaluate(args.data_dir, args.baseline_dir, args.submission_dir)
    print(json.dumps(res, ensure_ascii=False, indent=2, default=str))


if __name__ == "__main__":
    main()
