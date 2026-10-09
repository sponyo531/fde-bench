#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""整车配载（3D bin packing + 派车门槛 + 三维摆放合法性）离线评估器。

被评 agent 产出 ``solution.json``（结构化装载方案，不是可视化 HTML 也不是分析报告），
评估器从 ``data/`` 独立读取原始订单表与车型表，**独立重算**每 SKU 总箱数/总重量、
每辆车装载重量、区块几何合法性、装卸序合法性——绝不信任 agent 自报任何数值。

评分维度（合成为单一 quality）：
  - 装载覆盖率 coverage = 已装总箱数 / 订单总箱数
  - 长度利用率 length_util = 各车 max(x+dx) / 车长 的加权平均
  - 重量平衡 balance = 1 - std(各车装载重量) / mean(各车装载重量)
  quality = 0.6*coverage + 0.25*length_util + 0.15*balance  → 再 / baseline 归一。
硬约束违反 → validity=0 & quality=0；只输出前 8 条违反用于诊断。

EXTRACTOR_SPEC:
  plan_file: solution.json
  required_columns: []
  notes: >
    这是"交产物"任务：agent 产出一份整车配载方案 JSON，顶层字段 `trucks`（列表）
    和可选的 `unassigned`（列表）。标准结构：
      trucks[i] = {
        "truck_id": "V01",
        "cab_model": "13米高栏",           # 从车型表中选出的车型名
        "blocks": [
          {
            "block_id": "V01-B01",
            "sku": "<订单表中的物料代码>",
            "qty": <整数箱数>,
            "weight_kg": <该区块的毛重 kg>,
            "load_form": "CARTON" | "PALLET",
            "x": <米>, "y": <米>, "z": <米>,
            "dx": <米>, "dy": <米>, "dz": <米>,
            "seq_load": <该车内 1..N 的装车序号>,
            "seq_unload": <对应卸车序号>
          }, ...
        ]
      }
      unassigned[i] = {"sku": "...", "qty": <整数>, "reason": "..."}

    抽取器在 agent workspace 里找到最终装载方案（可能叫任意文件名，如
    loading_plan.json / solution.json / plan.json；也可能是 CSV / Excel），
    在**不改动装载决策**（不改哪个 SKU 装在哪辆车、每个区块放在哪、装卸序）的前提下，
    规整成上述标准 JSON 存为 solution.json：
      - 顶层字段若叫 loading_plan / plan / result / vehicles / plans[XXX] 等，改名为 trucks；
      - 若 agent 交了 PLAN-A + PLAN-B 两版方案，任选**装载覆盖率更高**的一版作为
        submission（PLAN-A/B 是同一批货的两种码放方式，最终只交一版评分）；
      - SKU 编码保留原样，不去点号/连字符/前导零；
      - x/y/z/dx/dy/dz 若给的是 cm 或 mm，统一换算成米；
      - seq_load 若缺，按 x/z 由小到大排序补齐；seq_unload 若缺，按单站规则
        seq_unload = N + 1 - seq_load 补齐（多站规则由 agent 自定）。
    禁止改动 agent 对每块货的装车判断本身。评估器不读 agent 自报的评分字段，
    只认 truck_id / cab_model / blocks（sku/qty/weight_kg/坐标/尺寸/装卸序）/ unassigned。
"""

from __future__ import annotations

import argparse
import json
import os
import re
import traceback
from collections import defaultdict
from pathlib import Path
from typing import Optional, Any, Dict, List, Tuple

import numpy as np
import pandas as pd

_HERE = os.path.dirname(os.path.abspath(__file__))
PLAN_FILE = "solution.json"

# ---- 业务常量（抽自整车配载业务规范，锁死） ----
MIN_TRUCK_WEIGHT_KG = 30000.0        # 30 吨派车门槛
CENTROID_X_LO_RATIO = 0.40           # 重心 X 下限（车长比例）
CENTROID_X_HI_RATIO = 0.60           # 重心 X 上限
CENTROID_Y_TOL_RATIO = 0.10          # 重心 Y 偏离中线容差（车宽比例）
LR_BALANCE_TOL_RATIO = 0.10          # 左右重量平衡容差
COORD_EPS = 1e-3                     # 米级坐标误差容差（1mm）
WEIGHT_EPS = 1.0                     # 重量守恒容差（kg），订单本身分摊到区块可能微秒误差
BLOCK_FILL_TOL = 0.05            # 区块填充体积容差(5%)：区块体积 vs 箱数×单箱体积, 堵 length_util 刷分


# ======================================================================
# 数据加载（独立重算真值）
# ======================================================================
def _norm_sku(s: Any) -> str:
    if s is None or (isinstance(s, float) and pd.isna(s)):
        return ""
    return str(s).strip()


def load_orders(data_dir: str) -> pd.DataFrame:
    path = os.path.join(data_dir, "订单数据_匿名仓库.xlsx")
    df = pd.read_excel(path, header=1)
    df = df[df["物料代码"].notna()].copy()
    df["sku"] = df["物料代码"].map(_norm_sku)
    df["boxes"] = pd.to_numeric(df["提单数量"], errors="coerce").fillna(0).astype(int)
    df["gross_kg"] = pd.to_numeric(df["订单毛重(MT)"], errors="coerce").fillna(0.0) * 1000.0
    df = df[df["boxes"] > 0].copy()
    return df[["sku", "boxes", "gross_kg", "售达方客户代码", "提单收货地址"]]


def _parse_dims(s: Any) -> Optional[Tuple[float, float, float]]:
    """解析物料表 "长*宽*高（cm）" 字符串 → (米,米,米)。空/NaN/「暂无」→ None。"""
    if s is None: return None
    t = str(s).strip()
    if not t or t in ("暂无","nan","NaN","None",""): return None
    try:
        parts = [float(x) for x in t.replace("（","(").replace("）",")").split("*")]
    except (ValueError, TypeError):
        return None
    if len(parts) != 3: return None
    return tuple(round(x/100.0, 6) for x in parts)  # cm -> m


def load_materials(data_dir: str) -> Dict[str, Tuple[float, float, float]]:
    """订单里每个 SKU 的单箱尺寸（米）。自产+外采两张物料表按物料编码关联。

    035 真正缺的校验：length_util 只看自报 dx，agent 能把区块拉成跨满车长的一块
    巨板来刷分，而 gt 要求"区块必须是完全填满的规则长方体，件数 = 沿长×宽×高箱数"。
    这里给出单箱体积，供 check_truck_hard_constraints 校验"区块体积 ≈ 箱数×单箱体积"。
    """
    out = {}
    for fname in ("工厂自产物料数据1.xlsx", "外采物料数据2.xlsx"):
        path = os.path.join(data_dir, fname)
        if not os.path.isfile(path):
            continue
        try:
            df = pd.read_excel(path)
        except Exception:
            continue
        # clean 数据已将 raw 的“SAP物料编码”匿名为“物料编码”；保留
        # raw 字段作为兼容回退，避免字段去敏后静默跳过全部尺寸数据。
        code_col = "物料编码" if "物料编码" in df.columns else "SAP物料编码"
        dim_col = [c for c in df.columns if "尺寸" in c and "长" in c][0]
        for _, r in df.iterrows():
            code = str(r.get(code_col, "")).strip()
            if not code:
                continue
            dims = _parse_dims(r.get(dim_col))
            if dims is None:
                continue
            # 同一 SKU 若两表都有，取先读到且非空的那个
            out.setdefault(code, dims)
    return out

def load_trucks(data_dir: str) -> Dict[str, Dict[str, float]]:
    """车型库 → {车型名: {L, W, H, rated_kg}}，长宽高单位米、rated_kg 单位 kg。"""
    path = os.path.join(data_dir, "日常发运车型.xlsx")
    df = pd.read_excel(path, header=None, skiprows=3)
    df.columns = ["_", "name", "L", "W", "H", "rated_ton"][: df.shape[1]]
    out: Dict[str, Dict[str, float]] = {}
    for _, r in df.iterrows():
        nm = str(r["name"]).strip() if pd.notna(r["name"]) else ""
        if not nm:
            continue
        try:
            out[nm] = {
                "L": float(r["L"]),
                "W": float(r["W"]),
                "H": float(r["H"]),
                "rated_kg": float(r["rated_ton"]) * 1000.0,
            }
        except (TypeError, ValueError):
            continue
    return out


def order_sku_totals(order_df: pd.DataFrame) -> Tuple[Dict[str, int], Dict[str, float]]:
    boxes = order_df.groupby("sku")["boxes"].sum().to_dict()
    weight = order_df.groupby("sku")["gross_kg"].sum().to_dict()
    boxes = {k: int(v) for k, v in boxes.items()}
    weight = {k: float(v) for k, v in weight.items()}
    return boxes, weight


def order_sku_kg_per_box(boxes: Dict[str, int], weight: Dict[str, float]) -> Dict[str, float]:
    return {k: (weight[k] / boxes[k]) if boxes[k] > 0 else 0.0 for k in boxes}


# ======================================================================
# 硬约束检查
# ======================================================================
def _overlap_1d(a0: float, a1: float, b0: float, b1: float, eps: float = COORD_EPS) -> bool:
    """两个区间 [a0, a1] [b0, b1] 是否严格相交（有正重叠体积）。"""
    return (a1 - b0 > eps) and (b1 - a0 > eps)


def _blocks_overlap(b1: Dict[str, Any], b2: Dict[str, Any]) -> bool:
    return (
        _overlap_1d(b1["x"], b1["x"] + b1["dx"], b2["x"], b2["x"] + b2["dx"])
        and _overlap_1d(b1["y"], b1["y"] + b1["dy"], b2["y"], b2["y"] + b2["dy"])
        and _overlap_1d(b1["z"], b1["z"] + b1["dz"], b2["z"], b2["z"] + b2["dz"])
    )


def _check_seq_load_x_z_monotone(blocks: List[Dict[str, Any]]) -> bool:
    """seq_load 顺序满足沿 X 由小到大，同 X 由 z 小到大。
       实现宽松：只要 seq_load 严格递增时 x 单调不降；同 x 时 z 单调不降。"""
    ordered = sorted(blocks, key=lambda b: int(b["seq_load"]))
    for i in range(1, len(ordered)):
        prev, curr = ordered[i - 1], ordered[i]
        if curr["x"] + COORD_EPS < prev["x"]:
            return False
        if abs(curr["x"] - prev["x"]) <= COORD_EPS and curr["z"] + COORD_EPS < prev["z"]:
            return False
    return True


def _check_same_sku_contiguous(blocks: List[Dict[str, Any]]) -> List[str]:
    """同一 SKU 的多个区块必须相邻并形成连续区域（图联通判定）。
       两个区块相邻 = 三维投影至少有一面接触（其中一维 |差| <= eps，其余两维投影相交）。"""
    violations: List[str] = []
    by_sku: Dict[str, List[Dict[str, Any]]] = defaultdict(list)
    for b in blocks:
        by_sku[_norm_sku(b.get("sku"))].append(b)

    for sku, blks in by_sku.items():
        if len(blks) <= 1:
            continue
        # 建立邻接图
        n = len(blks)
        adj = [[] for _ in range(n)]
        for i in range(n):
            for j in range(i + 1, n):
                if _blocks_adjacent(blks[i], blks[j]):
                    adj[i].append(j)
                    adj[j].append(i)
        # BFS 判连通
        visited = [False] * n
        stack = [0]
        visited[0] = True
        while stack:
            u = stack.pop()
            for v in adj[u]:
                if not visited[v]:
                    visited[v] = True
                    stack.append(v)
        if not all(visited):
            violations.append(f"SKU {sku}: 同车 {n} 个区块不连续（存在孤立子块）")
    return violations


def _blocks_adjacent(b1: Dict[str, Any], b2: Dict[str, Any]) -> bool:
    """两区块相邻：某一维 |界面差| ≤ eps（贴面）且其余两维投影严格相交或紧贴。"""
    def _touch_or_overlap(a0, a1, b0, b1):
        return (min(a1, b1) - max(a0, b0)) >= -COORD_EPS

    def _face_touch(a0, a1, b0, b1):
        return (abs(a1 - b0) <= COORD_EPS) or (abs(b1 - a0) <= COORD_EPS)

    x1, X1, x2, X2 = b1["x"], b1["x"] + b1["dx"], b2["x"], b2["x"] + b2["dx"]
    y1, Y1, y2, Y2 = b1["y"], b1["y"] + b1["dy"], b2["y"], b2["y"] + b2["dy"]
    z1, Z1, z2, Z2 = b1["z"], b1["z"] + b1["dz"], b2["z"], b2["z"] + b2["dz"]

    # X 面贴合
    if _face_touch(x1, X1, x2, X2) and _touch_or_overlap(y1, Y1, y2, Y2) and _touch_or_overlap(z1, Z1, z2, Z2):
        return True
    if _face_touch(y1, Y1, y2, Y2) and _touch_or_overlap(x1, X1, x2, X2) and _touch_or_overlap(z1, Z1, z2, Z2):
        return True
    if _face_touch(z1, Z1, z2, Z2) and _touch_or_overlap(x1, X1, x2, X2) and _touch_or_overlap(y1, Y1, y2, Y2):
        return True
    return False


def check_truck_hard_constraints(
    truck: Dict[str, Any],
    truck_spec: Dict[str, float],
    box_vol: Optional[Dict[str, float]] = None,
) -> List[str]:
    """逐辆车硬约束检查，返回违反列表（空=通过）。

    box_vol: SKU -> 单箱体积(米³)，来自物料表。用于校验"区块=完全填满的规则
    长方体"：区块体积 dx*dy*dz 必须 ≈ 箱数 × 单箱体积，否则就是一块中间带大量
    空位的巨板（length_util 刷分手法）。
    """
    v: List[str] = []
    tid = truck.get("truck_id", "?")
    blocks = truck.get("blocks", []) or []
    L, W, H, rated = truck_spec["L"], truck_spec["W"], truck_spec["H"], truck_spec["rated_kg"]

    if not blocks:
        v.append(f"{tid}: 无区块")
        return v

    # 区块字段基础校验
    for i, b in enumerate(blocks):
        for k in ("sku", "qty", "weight_kg", "x", "y", "z", "dx", "dy", "dz", "seq_load"):
            if k not in b:
                v.append(f"{tid}: block[{i}] 缺字段 {k}")
                return v
        if b["dx"] <= 0 or b["dy"] <= 0 or b["dz"] <= 0:
            v.append(f"{tid}: block[{i}] dx/dy/dz 必须为正")
            return v
        if b["qty"] <= 0:
            v.append(f"{tid}: block[{i}] qty 必须为正")
            return v

    # HC_GEOM 区块=完全填满的规则长方体：区块体积 ≈ 箱数 × 单箱体积
    # 堵 length_util 刷分：把区块拉成跨满车长的一块巨板，几箱货占巨大体积
    if box_vol:
        for b in blocks:
            box = box_vol.get(_norm_sku(b.get("sku")))
            if box is None:
                continue  # SKU 查不到尺寸就不强校验
            vol = float(b["dx"]) * float(b["dy"]) * float(b["dz"])
            expect = float(b["qty"]) * float(box)
            if expect <= 0:
                continue
            ratio = abs(vol - expect) / expect
            if ratio > BLOCK_FILL_TOL:
                v.append(
                    f"{tid}: block sku={b['sku']} 体积 {vol:.3f}m³ 与 {int(b['qty'])}箱×"
                    f"单箱{float(box):.4f}m³={expect:.3f}m³ 偏差 {ratio*100:.1f}%"
                    f"（区块未填满, 中间有空位）"
                )

    # HC3 30吨门槛
    total_kg = sum(float(b["weight_kg"]) for b in blocks)
    if total_kg + WEIGHT_EPS < MIN_TRUCK_WEIGHT_KG:
        v.append(f"{tid}: 装载重量 {total_kg:.1f} kg < 30000 kg 门槛")
    # HC4 车型标载
    if total_kg > rated + WEIGHT_EPS:
        v.append(f"{tid}: 装载重量 {total_kg:.1f} kg > 车型标载 {rated:.0f} kg")

    # HC5 车厢边界
    for b in blocks:
        if b["x"] < -COORD_EPS or b["x"] + b["dx"] > L + COORD_EPS:
            v.append(f"{tid}: block sku={b['sku']} X 越界 [{b['x']:.3f},{b['x']+b['dx']:.3f}] vs 车长 {L}")
            break
        if b["y"] < -COORD_EPS or b["y"] + b["dy"] > W + COORD_EPS:
            v.append(f"{tid}: block sku={b['sku']} Y 越界 [{b['y']:.3f},{b['y']+b['dy']:.3f}] vs 车宽 {W}")
            break
        if b["z"] < -COORD_EPS or b["z"] + b["dz"] > H + COORD_EPS:
            v.append(f"{tid}: block sku={b['sku']} Z 越界 [{b['z']:.3f},{b['z']+b['dz']:.3f}] vs 车高 {H}")
            break

    # HC6 区块不重叠（O(n^2) 检查，货车区块数 ~几十，OK）
    for i in range(len(blocks)):
        for j in range(i + 1, len(blocks)):
            if _blocks_overlap(blocks[i], blocks[j]):
                v.append(
                    f"{tid}: 区块重叠 sku={blocks[i]['sku']}(seq={blocks[i]['seq_load']}) "
                    f"↔ sku={blocks[j]['sku']}(seq={blocks[j]['seq_load']})"
                )
                break
        if any(x.startswith(f"{tid}: 区块重叠") for x in v):
            break

    # 重心与左右平衡
    if total_kg > 0:
        cx = sum(float(b["weight_kg"]) * (b["x"] + b["dx"] / 2) for b in blocks) / total_kg
        cy = sum(float(b["weight_kg"]) * (b["y"] + b["dy"] / 2) for b in blocks) / total_kg
        if not (CENTROID_X_LO_RATIO * L - COORD_EPS <= cx <= CENTROID_X_HI_RATIO * L + COORD_EPS):
            v.append(f"{tid}: 重心 X={cx:.3f} 超出 [{CENTROID_X_LO_RATIO*L:.2f},{CENTROID_X_HI_RATIO*L:.2f}]")
        if abs(cy - W / 2) > CENTROID_Y_TOL_RATIO * W + COORD_EPS:
            v.append(f"{tid}: 重心 Y={cy:.3f} 偏离中线 {abs(cy - W/2):.3f} > {CENTROID_Y_TOL_RATIO*W:.3f}")

        # HC8 左右重量平衡：以 y_center = W/2 为界，跨界区块按体积分摊
        left_kg = right_kg = 0.0
        mid = W / 2
        for b in blocks:
            y0, y1 = b["y"], b["y"] + b["dy"]
            wkg = float(b["weight_kg"])
            if y1 <= mid + COORD_EPS:
                left_kg += wkg
            elif y0 >= mid - COORD_EPS:
                right_kg += wkg
            else:
                # 按体积比例分摊
                left_share = (mid - y0) / (y1 - y0)
                left_kg += wkg * left_share
                right_kg += wkg * (1 - left_share)
        if total_kg > 0 and abs(left_kg - right_kg) / total_kg > LR_BALANCE_TOL_RATIO + 1e-6:
            v.append(
                f"{tid}: 左右重量偏差 |{left_kg:.1f}-{right_kg:.1f}|/{total_kg:.1f}"
                f" = {abs(left_kg-right_kg)/total_kg*100:.1f}% > 10%"
            )

    # HC9 seq_load 是 1..N 排列，seq_unload = N+1-seq_load（单站规则）
    n = len(blocks)
    seq_loads = [int(b["seq_load"]) for b in blocks]
    if sorted(seq_loads) != list(range(1, n + 1)):
        v.append(f"{tid}: seq_load 不是 1..{n} 的排列，实际={sorted(seq_loads)[:10]}...")
    else:
        # 检查单站装卸序对应关系（若给了 seq_unload）
        for b in blocks:
            if "seq_unload" in b:
                expected = n + 1 - int(b["seq_load"])
                if int(b["seq_unload"]) != expected:
                    v.append(
                        f"{tid}: 单站装卸序不符 block seq_load={b['seq_load']} 期望 seq_unload={expected} "
                        f"实际={b['seq_unload']}"
                    )
                    break
        # X/Z 单调性
        if not _check_seq_load_x_z_monotone(blocks):
            v.append(f"{tid}: seq_load 顺序未满足 X 非降 + 同 X 时 Z 非降")

    # HC10 同 SKU 连续成区
    v.extend(_check_same_sku_contiguous(blocks))

    return v


def check_global_hard_constraints(
    trucks: List[Dict[str, Any]],
    unassigned: List[Dict[str, Any]],
    order_boxes: Dict[str, int],
    order_weight: Dict[str, float],
) -> List[str]:
    v: List[str] = []

    # 全局数量守恒
    dispatched_boxes: Dict[str, int] = defaultdict(int)
    dispatched_weight: Dict[str, float] = defaultdict(float)
    for tr in trucks:
        for b in tr.get("blocks", []):
            sku = _norm_sku(b.get("sku"))
            dispatched_boxes[sku] += int(b.get("qty", 0))
            dispatched_weight[sku] += float(b.get("weight_kg", 0))
    unassigned_boxes: Dict[str, int] = defaultdict(int)
    for u in unassigned:
        sku = _norm_sku(u.get("sku"))
        _q = int(u.get("qty", 0))
        # 修复：unassigned 填负数可在数量守恒下把已装箱数抬到订单量以上，coverage > 1。
        if _q < 0:
            v.append(f"SKU {sku}: unassigned.qty 为负 ({_q})")
            continue
        unassigned_boxes[sku] += _q
    for tr in trucks:
        for b in tr.get("blocks", []):
            if int(b.get("qty", 0)) < 0:
                v.append(f"{tr.get('truck_id')}: block sku={_norm_sku(b.get('sku'))} qty 为负")

    all_skus = set(order_boxes) | set(dispatched_boxes) | set(unassigned_boxes)
    for sku in all_skus:
        expect = order_boxes.get(sku, 0)
        actual = dispatched_boxes.get(sku, 0) + unassigned_boxes.get(sku, 0)
        if actual != expect:
            v.append(f"SKU {sku}: 数量不守恒 (已装 {dispatched_boxes.get(sku,0)} + 未装 {unassigned_boxes.get(sku,0)}) ≠ 订单 {expect}")

    # 全局重量守恒: 自报重量必须等于"已装箱数 x 该 SKU 单箱毛重", 双向校验。
    # 只卡上界的话, 少报 weight_kg 就能绕开 30t 门槛与车型标载。
    kg_per_box = order_sku_kg_per_box(order_boxes, order_weight)
    for sku in all_skus:
        actual_kg = dispatched_weight.get(sku, 0.0)
        expect_kg = dispatched_boxes.get(sku, 0) * kg_per_box.get(sku, 0.0)
        tol = max(10.0, abs(expect_kg) * 1e-3)
        if abs(actual_kg - expect_kg) > tol:
            v.append(f"SKU {sku}: 自报已装重量 {actual_kg:.1f} kg 与 "
                     f"{dispatched_boxes.get(sku,0)} 箱应有的 {expect_kg:.1f} kg 不符")

    return v


# ======================================================================
# 质量评分：装载覆盖率 + 长度利用率 + 重量平衡
# ======================================================================
def compute_quality(
    trucks: List[Dict[str, Any]],
    truck_specs: Dict[str, Dict[str, float]],
    order_boxes: Dict[str, int],
) -> Tuple[float, Dict[str, float]]:
    total_boxes = sum(order_boxes.values())
    dispatched_boxes = sum(
        int(b.get("qty", 0)) for tr in trucks for b in tr.get("blocks", [])
    )
    coverage = dispatched_boxes / total_boxes if total_boxes > 0 else 0.0
    coverage = min(coverage, 1.0)   # 修复：覆盖率按定义不超过 1

    # 长度利用率：sum(每车 max(x+dx)) / sum(每车车长)
    used_len, avail_len = 0.0, 0.0
    for tr in trucks:
        cab = str(tr.get("cab_model", "")).strip()
        spec = truck_specs.get(cab)
        if not spec:
            continue
        blks = tr.get("blocks", []) or []
        if not blks:
            continue
        max_x_end = max(b["x"] + b["dx"] for b in blks)
        used_len += max_x_end
        avail_len += spec["L"]
    length_util = used_len / avail_len if avail_len > 0 else 0.0

    # 重量平衡：1 - CV = 1 - std/mean
    truck_weights = [
        sum(float(b.get("weight_kg", 0)) for b in tr.get("blocks", []))
        for tr in trucks
    ]
    truck_weights = [w for w in truck_weights if w > 0]
    if len(truck_weights) >= 2:
        arr = np.array(truck_weights, dtype=float)
        cv = float(arr.std() / arr.mean()) if arr.mean() > 0 else 1.0
        balance = max(0.0, 1.0 - cv)
    elif len(truck_weights) == 1:
        balance = 1.0  # 单车无所谓平衡
    else:
        balance = 0.0

    combined = 0.60 * coverage + 0.25 * length_util + 0.15 * balance
    return combined, {
        "coverage": round(coverage, 6),
        "length_util": round(length_util, 6),
        "balance": round(balance, 6),
        "combined": round(combined, 6),
        "dispatched_boxes": dispatched_boxes,
        "total_boxes": total_boxes,
        "n_trucks": len([w for w in truck_weights if w > 0]),
    }


def load_baseline() -> Tuple[float, str]:
    p = os.path.join(_HERE, "baseline", "reference_metrics.json")
    with open(p, "r", encoding="utf-8") as f:
        d = json.load(f)
    return float(d["reference_value"]), str(d.get("direction", "higher_is_better"))


# ======================================================================
# 主评估
# ======================================================================
def evaluate(submission_dir: str, data_dir: str) -> Dict[str, Any]:
    m = {"validity_score": 0.0, "quality_score": 0.0, "overall_score": 0.0, "error_info": {}}
    try:
        plan_path = os.path.join(submission_dir, PLAN_FILE)
        if not os.path.isfile(plan_path):
            m["error_info"] = {"fatal": [f"未找到 {PLAN_FILE}"]}
            return m
        with open(plan_path, "r", encoding="utf-8") as f:
            sol = json.load(f)
        if not isinstance(sol, dict):
            m["error_info"] = {"fatal": ["solution.json 顶层必须是 dict"]}
            return m
        trucks = sol.get("trucks")
        if not isinstance(trucks, list) or not trucks:
            m["error_info"] = {"fatal": ["solution.json 缺 trucks 列表或为空"]}
            return m
        unassigned = sol.get("unassigned", []) or []
        if not isinstance(unassigned, list):
            unassigned = []

        # 独立重算订单真值
        order_df = load_orders(data_dir)
        order_boxes, order_weight = order_sku_totals(order_df)
        truck_specs = load_trucks(data_dir)
        box_vol = {k: float(v[0]*v[1]*v[2]) for k, v in load_materials(data_dir).items()}

        # ── 逐车硬约束 ──
        violations: List[str] = []
        truck_ids = set()
        for tr in trucks:
            tid = str(tr.get("truck_id", "")).strip()
            if not tid:
                violations.append("存在无 truck_id 的车辆")
                continue
            if tid in truck_ids:
                violations.append(f"truck_id 重复: {tid}")
                continue
            truck_ids.add(tid)
            cab = str(tr.get("cab_model", "")).strip()
            spec = truck_specs.get(cab)
            if not spec:
                violations.append(f"{tid}: 未知车型 '{cab}'，不在车型库 {sorted(truck_specs.keys())}")
                continue
            violations.extend(check_truck_hard_constraints(tr, spec, box_vol))

        # ── 全局数量/重量守恒 ──
        violations.extend(check_global_hard_constraints(trucks, unassigned, order_boxes, order_weight))

        if violations:
            m["validity_score"] = 0.0
            m["quality_score"] = 0.0
            m["overall_score"] = 0.0
            m["error_info"] = {"constraint": violations[:8], "n_violations": len(violations)}
            return m

        m["validity_score"] = 1.0
        # ── 质量分 ──
        player_combined, detail = compute_quality(trucks, truck_specs, order_boxes)
        baseline, direction = load_baseline()
        if direction == "higher_is_better":
            quality = player_combined / baseline if baseline > 0 else 0.0
        else:
            quality = baseline / player_combined if player_combined > 0 else 0.0
        m["quality_score"] = round(float(quality), 6)
        m["overall_score"] = m["quality_score"]
        m["error_info"] = {
            "player_combined": player_combined,
            "reference_value": baseline,
            **detail,
        }
        return m

    except Exception as e:
        m["error_info"] = {"exception": str(e), "traceback": traceback.format_exc()[-1200:]}
        return m


def main():
    ap = argparse.ArgumentParser(description="整车配载评估器")
    ap.add_argument("--submission-dir", required=True)
    ap.add_argument("--data-dir", default=os.path.join(_HERE, "..", "data"))
    a = ap.parse_args()
    print(json.dumps(evaluate(a.submission_dir, a.data_dir), ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
