"""
EXTRACTOR_SPEC:
  plan_file: solution.json
  schema: |
    {
      "pillars": [
        {"shape": "circle",   "cx": <m>, "cy": <m>, "r": <m>},
        {"shape": "ellipse",  "cx": <m>, "cy": <m>, "a": <m>, "b": <m>, "theta": <rad>},
        {"shape": "teardrop", "cx": <m>, "cy": <m>, "r_front": <m>, "k": <float>}
      ]
    }
  notes: >
    微通道内微柱结构设计。选手产物是一组微柱的形状与参数，长度单位为米。

    若选手用了别的键名（type/kind 代替 shape、x/y 代替 cx/cy、radius 代替 r、
    semi_major/semi_minor 代替 a/b、angle 代替 theta、ratio 代替 k），映射到
    上面的 schema。形状名统一成小写 circle / ellipse / teardrop。

    数值一律照搬选手给出的原始值 —— 即使看起来像微米（如 120 而非 1.2e-4），
    也不要替选手做单位换算，评估器会按米解释。
    不要自行增删微柱、不要重新优化。
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import traceback
from pathlib import Path
from typing import Any, Dict

import numpy as np

_HERE = Path(__file__).resolve().parent


# ===================== 内联自 tests/pillar_geometry.py（评估器自包含，原文件已删除） =====================
"""
pillar_geometry.py
微柱形状参数化：圆形、椭圆形、水滴形
提供：点-in-shape 检测、固体掩码生成、周长计算、碰撞检测
"""

import numpy as np


# -------------------------------------------------------------------
# 形状类（通用接口）
# -------------------------------------------------------------------

class CirclePillar:
    shape_id = 0

    def __init__(self, cx, cy, r):
        self.cx = cx
        self.cy = cy
        self.r = r

    def contains(self, x, y):
        return (x - self.cx)**2 + (y - self.cy)**2 <= self.r**2

    def perimeter(self):
        return 2 * np.pi * self.r

    def bounding_radius(self):
        """外接圆半径（用于间距检测）"""
        return self.r

    def min_thickness(self):
        return 2 * self.r

    def min_dist_to(self, other):
        """形心距离 - 两外接圆半径之和，用于间距近似"""
        d = np.hypot(self.cx - other.cx, self.cy - other.cy)
        return d - self.bounding_radius() - other.bounding_radius()

    def to_dict(self):
        return {'type': 'circle', 'cx': self.cx, 'cy': self.cy, 'r': self.r}


class EllipsePillar:
    shape_id = 1

    def __init__(self, cx, cy, a, b, theta):
        """theta: 长轴与 x 轴夹角（rad）"""
        self.cx = cx
        self.cy = cy
        self.a = a
        self.b = b
        self.theta = theta

    def contains(self, x, y):
        dx = x - self.cx
        dy = y - self.cy
        cos_t = np.cos(self.theta)
        sin_t = np.sin(self.theta)
        xr =  dx * cos_t + dy * sin_t
        yr = -dx * sin_t + dy * cos_t
        return (xr / self.a)**2 + (yr / self.b)**2 <= 1.0

    def perimeter(self):
        # Ramanujan 近似
        a, b = self.a, self.b
        h = ((a - b) / (a + b))**2
        return np.pi * (a + b) * (1 + 3*h / (10 + np.sqrt(4 - 3*h)))

    def bounding_radius(self):
        return max(self.a, self.b)

    def min_thickness(self):
        return 2 * min(self.a, self.b)

    def min_dist_to(self, other):
        d = np.hypot(self.cx - other.cx, self.cy - other.cy)
        return d - self.bounding_radius() - other.bounding_radius()

    def to_dict(self):
        return {
            'type': 'ellipse', 'cx': self.cx, 'cy': self.cy,
            'a': self.a, 'b': self.b, 'theta': float(self.theta)
        }


class TeardropPillar:
    shape_id = 2
    """
    前缘圆弧（半径 r_front），后缘线性收缩到尖端（尖端朝 +x）
    形心 (cx, cy)，总长 = k * 2 * r_front
    """

    def __init__(self, cx, cy, r_front, k):
        self.cx = cx
        self.cy = cy
        self.r_front = r_front
        self.k = k
        L = 2 * k * r_front
        # 前缘圆心在形心左侧
        self._xfc = cx - L / 2 + r_front   # 前缘圆心 x
        self._xtip = cx + L / 2             # 尖端 x
        self._L = L

    def contains(self, x, y):
        # 前缘圆（向量化）
        in_front = (x - self._xfc)**2 + (y - self.cy)**2 <= self.r_front**2
        # 线性锥形后缘
        in_back_x = (x >= self._xfc) & (x <= self._xtip)
        span = self._xtip - self._xfc
        if span < 1e-12:
            in_back = np.zeros_like(in_front, dtype=bool) if hasattr(in_front, '__len__') else False
        else:
            half_w = self.r_front * (1.0 - (x - self._xfc) / span)
            in_back = in_back_x & (np.abs(y - self.cy) <= half_w)
        return in_front | in_back

    def perimeter(self):
        # 前缘半圆 + 两条斜边
        r = self.r_front
        L_half = self._xtip - self._xfc       # 后缘长度
        slant = np.sqrt(L_half**2 + r**2)      # 单侧斜边长度
        return np.pi * r + 2 * slant

    def bounding_radius(self):
        # k < 1 时总长小于前缘直径，前缘圆会伸出 L/2 之外——外接圆必须真的外接，
        # 否则约束按 L/2 算而 contains() 按 r_front 判固体，两者脱节。
        return max(self._L / 2, self.r_front)

    def min_thickness(self):
        return 2 * self.r_front

    def min_dist_to(self, other):
        d = np.hypot(self.cx - other.cx, self.cy - other.cy)
        return d - self.bounding_radius() - other.bounding_radius()

    def to_dict(self):
        return {
            'type': 'teardrop', 'cx': self.cx, 'cy': self.cy,
            'r_front': self.r_front, 'k': self.k
        }


# -------------------------------------------------------------------
# 工具函数
# -------------------------------------------------------------------

def build_solid_mask(pillars, nx, ny, dx, dy):
    """
    在 nx×ny 网格上标记固体单元（单元中心点 in 任一微柱）
    返回 (nx, ny) bool array
    """
    mask = np.zeros((nx, ny), dtype=bool)
    if not pillars:
        return mask

    # 单元中心坐标
    xc = (np.arange(nx) + 0.5) * dx    # (nx,)
    yc = (np.arange(ny) + 0.5) * dy    # (ny,)
    XX, YY = np.meshgrid(xc, yc, indexing='ij')  # (nx, ny)

    for p in pillars:
        mask |= p.contains(XX, YY)

    return mask


def total_perimeter(pillars):
    """所有微柱周长之和（m）"""
    return sum(p.perimeter() for p in pillars)


# -------------------------------------------------------------------
# 约束检测
# -------------------------------------------------------------------

WALL_GAP = 10e-6     # 微柱与壁面最小间距
PILLAR_GAP = 10e-6   # 相邻微柱最小间距
MIN_THICKNESS = 10e-6  # 微柱任何方向的最小厚度（微加工可实现下限，与特征尺寸下限一致）
L_CHAN = 500e-6      # 通道长度
W_CHAN = 200e-6      # 通道宽度


def check_constraints(pillars):
    """
    检查所有约束，返回 True 表示合法
    """
    for p in pillars:
        # 水滴形总长必须不小于前缘直径（k ≥ 1），否则不构成水滴形轮廓
        if isinstance(p, TeardropPillar) and not (p.k >= 1.0 and p.r_front > 0):
            return False
        # 任何方向的厚度不得小于 MIN_THICKNESS：否则"柱子"退化成一格厚的刀片/导流板
        if p.min_thickness() < MIN_THICKNESS - 1e-10:
            return False
        br = p.bounding_radius()
        # 微柱与所有壁面最小间距 WALL_GAP
        if p.cx - br < WALL_GAP:
            return False
        if p.cx + br > L_CHAN - WALL_GAP:
            return False
        if p.cy - br < WALL_GAP:
            return False
        if p.cy + br > W_CHAN - WALL_GAP:
            return False
        # 微柱特征尺寸约束 [10μm, 80μm]（含浮点容差 0.1nm）
        char_dim = 2 * br
        tol = 1e-10
        if char_dim < 10e-6 - tol or char_dim > 80e-6 + tol:
            return False

    # 微柱间距
    for i in range(len(pillars)):
        for j in range(i + 1, len(pillars)):
            if pillars[i].min_dist_to(pillars[j]) < PILLAR_GAP:
                return False
    return True


# -------------------------------------------------------------------
# 编解码（基因 → 微柱列表）
# -------------------------------------------------------------------

MAX_PILLARS = 8
GENE_PER_PILLAR = 7   # [active, x, y, shape(0/1/2), p1, p2, p3]


def decode_individual(ind):
    """
    ind: numpy array (MAX_PILLARS * GENE_PER_PILLAR,)
    返回 (pillars_list, valid)
    """
    pillars = []
    for k in range(MAX_PILLARS):
        g = ind[k * GENE_PER_PILLAR: (k + 1) * GENE_PER_PILLAR]
        active = g[0] > 0.5
        if not active:
            continue

        cx = float(np.clip(g[1], 0, 1)) * L_CHAN
        cy = float(np.clip(g[2], 0, 1)) * W_CHAN
        shape_type = int(np.clip(round(g[3] * 2), 0, 2))
        p1 = float(g[4])
        p2 = float(g[5])
        p3 = float(g[6])

        try:
            if shape_type == 0:   # 圆形 r ∈ [5,40] μm
                r = 5e-6 + (p1 % 1.0) * 35e-6
                pillar = CirclePillar(cx, cy, r)

            elif shape_type == 1:  # 椭圆 a∈[5,40] b∈[5,a] θ∈[0,π]
                a = 5e-6 + (p1 % 1.0) * 35e-6
                b = 5e-6 + (p2 % 1.0) * (a - 5e-6)
                theta = (p3 % 1.0) * np.pi
                pillar = EllipsePillar(cx, cy, a, b, theta)

            else:                  # 水滴形 r_front∈[5,30] k∈[1.5,4]
                r_front = 5e-6 + (p1 % 1.0) * 25e-6
                k = 1.5 + (p2 % 1.0) * 2.5
                pillar = TeardropPillar(cx, cy, r_front, k)

            pillars.append(pillar)
        except Exception:
            continue

    valid = check_constraints(pillars)
    return pillars, valid


def encode_random(rng=None):
    """生成随机个体基因（约束感知初始化）"""
    if rng is None:
        rng = np.random.default_rng()

    ind = np.zeros(MAX_PILLARS * GENE_PER_PILLAR)

    # 每个 slot
    for k in range(MAX_PILLARS):
        base = k * GENE_PER_PILLAR

        # 60% 激活概率
        if rng.random() > 0.6:
            ind[base] = 0.0
            ind[base+1:base+GENE_PER_PILLAR] = rng.random(GENE_PER_PILLAR - 1)
            continue
        ind[base] = 1.0

        # 形状类型
        shape = rng.integers(0, 3)
        ind[base + 3] = shape / 2.0  # 0→0, 1→0.5, 2→1.0

        # 根据形状确定参数（确保尺寸小，bounding_radius 可控）
        if shape == 0:  # circle r∈[5,20]μm → bounding_r≤20μm
            r = rng.uniform(5e-6, 20e-6)
            ind[base + 4] = (r - 5e-6) / 35e-6  # p1

        elif shape == 1:  # ellipse a∈[5,20]μm
            a = rng.uniform(5e-6, 20e-6)
            b = rng.uniform(5e-6, a)
            theta = rng.uniform(0, np.pi)
            ind[base + 4] = (a - 5e-6) / 35e-6
            ind[base + 5] = (b - 5e-6) / max(a - 5e-6, 1e-9)
            ind[base + 6] = theta / np.pi

        else:  # teardrop r_front∈[5,15]μm k∈[1.5,2.5] → total_len≤75μm, br≤37.5μm
            r_front = rng.uniform(5e-6, 15e-6)
            k = rng.uniform(1.5, 2.5)
            ind[base + 4] = (r_front - 5e-6) / 25e-6
            ind[base + 5] = (k - 1.5) / 2.5

        # 位置：留足 bounding_radius + WALL_GAP 的余量
        # 保守估计 bounding_r ≤ 40μm
        margin = 50e-6   # = 40μm (max br) + 10μm (wall_gap)
        x_lo = margin / L_CHAN
        x_hi = 1.0 - margin / L_CHAN
        y_lo = margin / W_CHAN
        y_hi = 1.0 - margin / W_CHAN

        ind[base + 1] = rng.uniform(max(0.05, x_lo), min(0.95, x_hi))
        ind[base + 2] = rng.uniform(max(0.05, y_lo), min(0.95, y_hi))

    return ind

# ===================== 内联自 tests/cfd_solver.py（评估器自包含，原文件已删除） =====================
"""
cfd_solver.py
2D 稳态 Stokes 流动求解器（Brinkman 惩罚法 + MAC 交错网格）
域: [0,L] x [0,W]，入口左侧均匀速度 U_in，出口右侧 p=0，上下壁面无滑移
"""

import numpy as np
from scipy.sparse import csr_matrix
from scipy.sparse.linalg import spsolve


def solve_stokes_brinkman(
    solid_mask,
    nx=200, ny=80,
    L=500e-6, W=200e-6,
    mu=1.789e-5, U_in=1.0,
    K_solid=1e-14,
):
    """
    Parameters
    ----------
    solid_mask : (nx, ny) bool array  True = 固体单元
    Returns
    -------
    u_field : (nx+1, ny)   x 方向速度（x 面心）
    v_field : (nx, ny+1)   y 方向速度（y 面心）
    p_field : (nx, ny)     压力（单元中心）
    """
    dx = L / nx
    dy = W / ny
    alpha = mu / K_solid        # Brinkman 固体阻力系数

    n_u = (nx + 1) * ny         # u DOF
    n_v = nx * (ny + 1)         # v DOF
    n_p = nx * ny               # p DOF
    N = n_u + n_v + n_p

    # 索引辅助函数
    def ui(i, j): return i * ny + j
    def vi(i, j): return n_u + i * (ny + 1) + j
    def pi_(i, j): return n_u + n_v + i * ny + j

    rows, cols, data_vals = [], [], []
    b = np.zeros(N)

    def add(r, c, v):
        rows.append(int(r))
        cols.append(int(c))
        data_vals.append(float(v))

    def set_dirichlet(eq, val):
        add(eq, eq, 1.0)
        b[eq] = val

    # ----------------------------------------------------------------
    # U-动量方程
    # ----------------------------------------------------------------
    for i in range(nx + 1):
        for j in range(ny):
            eq = ui(i, j)

            # 入口 Dirichlet: u = U_in
            if i == 0:
                set_dirichlet(eq, U_in)
                continue

            # 出口 Neumann: u[nx,j] = u[nx-1,j]
            if i == nx:
                add(eq, eq, 1.0)
                add(eq, ui(nx - 1, j), -1.0)
                continue

            # 内部 u 节点 ── Brinkman chi（相邻两单元平均）
            chi = 0.5 * (float(solid_mask[i - 1, j]) + float(solid_mask[i, j]))
            br = alpha * chi

            diag = -(2.0 * mu / dx**2 + 2.0 * mu / dy**2 + br)

            # d²u/dx²（u[i-1]、u[i+1] 均为合法 DOF）
            add(eq, ui(i + 1, j), mu / dx**2)
            add(eq, ui(i - 1, j), mu / dx**2)

            # d²u/dy²（j=0: 下壁 ghost u[i,-1]=-u[i,0]; j=ny-1: 上壁 ghost）
            if j == 0:
                diag -= mu / dy**2          # ghost 使中心系数再减 mu/dy²
            else:
                add(eq, ui(i, j - 1), mu / dy**2)

            if j == ny - 1:
                diag -= mu / dy**2
            else:
                add(eq, ui(i, j + 1), mu / dy**2)

            add(eq, eq, diag)

            # 压力梯度 -dp/dx = -(p[i,j]-p[i-1,j])/dx
            add(eq, pi_(i,     j), -1.0 / dx)
            add(eq, pi_(i - 1, j),  1.0 / dx)

    # ----------------------------------------------------------------
    # V-动量方程
    # ----------------------------------------------------------------
    for i in range(nx):
        for j in range(ny + 1):
            eq = vi(i, j)

            # 上下壁 Dirichlet: v = 0
            if j == 0 or j == ny:
                set_dirichlet(eq, 0.0)
                continue

            # 内部 v 节点
            chi = 0.5 * (float(solid_mask[i, j - 1]) + float(solid_mask[i, j]))
            br = alpha * chi

            # d²v/dy²（j=1..ny-1，上下均有合法 DOF）
            diag = -(2.0 * mu / dy**2 + br)
            add(eq, vi(i, j + 1), mu / dy**2)
            add(eq, vi(i, j - 1), mu / dy**2)

            # d²v/dx²（按边界情况分类）
            if i == 0 and nx == 1:
                diag -= 2.0 * mu / dx**2
            elif i == 0:
                # 入口 ghost: v[-1,j] = -v[0,j]  → 系数 -3mu/dx²
                diag -= 3.0 * mu / dx**2
                add(eq, vi(1, j), mu / dx**2)
            elif i == nx - 1:
                # 出口 Neumann: v[nx,j]=v[nx-1,j] → 系数 -mu/dx²
                diag -= mu / dx**2
                add(eq, vi(nx - 2, j), mu / dx**2)
            else:
                diag -= 2.0 * mu / dx**2
                add(eq, vi(i - 1, j), mu / dx**2)
                add(eq, vi(i + 1, j), mu / dx**2)

            add(eq, eq, diag)

            # 压力梯度 -dp/dy
            add(eq, pi_(i, j    ), -1.0 / dy)
            add(eq, pi_(i, j - 1),  1.0 / dy)

    # ----------------------------------------------------------------
    # 连续性方程（散度 = 0）
    # 出口列 (i=nx-1) 替换为压力 Dirichlet p=0
    # ----------------------------------------------------------------
    for i in range(nx):
        for j in range(ny):
            eq = pi_(i, j)

            if i == nx - 1:
                set_dirichlet(eq, 0.0)
                continue

            # (u[i+1,j]-u[i,j])/dx + (v[i,j+1]-v[i,j])/dy = 0
            add(eq, ui(i + 1, j), 1.0 / dx)
            add(eq, ui(i,     j), -1.0 / dx)
            add(eq, vi(i, j + 1), 1.0 / dy)
            add(eq, vi(i, j    ), -1.0 / dy)

    # ----------------------------------------------------------------
    # 组装稀疏矩阵并求解
    # ----------------------------------------------------------------
    A = csr_matrix((data_vals, (rows, cols)), shape=(N, N))
    x = spsolve(A, b)

    u_field = x[:n_u].reshape(nx + 1, ny)
    v_field = x[n_u: n_u + n_v].reshape(nx, ny + 1)
    p_field = x[n_u + n_v:].reshape(nx, ny)

    return u_field, v_field, p_field


def get_velocity_at_cross_section(u_field, x_frac, nx, ny, dy):
    """
    在通道长度 x_frac 处（0~1）提取 u 速度剖面（沿 y 方向）
    u_field: (nx+1, ny), u 节点在 x = i*dx
    返回 (y_coords, u_vals) 各 ny 点
    """
    # x_frac 处最近的 u 节点索引
    i = int(round(x_frac * nx))
    i = max(1, min(nx - 1, i))
    u_profile = u_field[i, :]                       # (ny,)
    y_coords = (np.arange(ny) + 0.5) * dy           # 单元中心 y
    return y_coords, u_profile


# 采样截面最低流体占比。按题设约束（壁面间距、柱间间距各 ≥10μm）任何合法布局在任一
# 截面上至少留 (10+10+10)/200 = 15% 的流体，所以这条不会误伤合法设计；它防的是把通道
# 堵到只剩个位数单元——一个样本的标准差恒为 0，"均匀性"会被白拿满分。
MIN_FLUID_FRAC = 0.10


def sample_columns(nx):
    """均匀性采样的截面列号：流向 20%~80% 每 2% 一个，共 31 个。"""
    fracs = [0.20 + 0.02 * k for k in range(31)]
    cols = []
    for frac in fracs:
        i = int(round(frac * nx))
        i = max(1, min(nx - 1, i))
        cols.append(i)
    return sorted(set(cols))


def blocked_sections(solid_mask, nx, min_frac=MIN_FLUID_FRAC):
    """返回流体占比低于 min_frac 的采样截面列号列表（空列表 = 通道未堵塞）。"""
    fluid = ~np.asarray(solid_mask).astype(bool)
    return [i for i in sample_columns(nx) if fluid[i, :].mean() < min_frac]


def compute_metrics(u_field, p_field, solid_mask, nx, ny, dy):
    """
    返回 (nmrse, delta_p)
    nmrse : 三截面速度分布的归一化均方根误差
    delta_p: 入口 - 出口 压降 (Pa)
    """
    # 2026-08-23 修，两处：
    # ① 原来只在 x = 25% / 50% / 75% 三个截面取样。柱子稀疏时，只要把柱心挪开这三个面，
    #    速度剖面在采样处看起来就是平的 —— 实测同一根 r=40μm 的柱子放 x=250μm 时
    #    nmrse 0.6472 / 得分 0.702，移到 x=300μm 变成 0.4557 / 0.911，白拿三成分。
    #    现改为沿流向密集取样（20%~80% 每 2% 一个截面，共 31 个面），躲不开。
    # ② 原注释写「排除固体」，但代码把 u_field[i, :] 整列拿走、并未排除固体单元。
    #    柱内单元 u≈0，混进来测的就变成「通道被堵了多少」而不是「流动多不多匀」，
    #    而 solid_mask 这个入参本来就是为此准备的、却一直没用。现按注释的原意排除固体。
    cols = sample_columns(nx)

    fluid = ~solid_mask.astype(bool) if solid_mask is not None else None
    parts = []
    for i in cols:
        col = u_field[i, :]
        if fluid is not None:
            col = col[fluid[i, :]]
        if col.size:
            parts.append(col)
    if not parts or (solid_mask is not None and blocked_sections(solid_mask, nx)):
        return 1e6, float(np.mean(p_field[0, :]))

    all_u = np.concatenate(parts)
    u_mean = np.mean(all_u)
    if u_mean < 1e-12:
        nmrse = 1e6
    else:
        nmrse = np.sqrt(np.mean((all_u - u_mean)**2)) / u_mean

    # 压降：入口平均压力（左侧单元 i=0）vs 出口（右侧 p=0 已设置）
    delta_p = float(np.mean(p_field[0, :]))       # 入口处压力 - 0

    return nmrse, delta_p

# ===================== 以下为原 evaluator.py =====================
pass  # 已内联: from cfd_solver import ...
pass  # 已内联: from pillar_geometry import ...

PLAN_FILE = "solution.json"
NX, NY = 200, 80
L_CHAN, W_CHAN = 500e-6, 200e-6
MAX_PILLARS = 8
NMRSE_REF = 1.0
DP_REF = 50.0
PERIM_REF = 2000e-6


def load_baseline() -> float:
    with open(_HERE / "baseline" / "reference_metrics.json", encoding="utf-8") as f:
        return float(json.load(f)["reference_value"])


def _build(p: Dict[str, Any]):
    shape = str(p.get("shape", "")).strip().lower()
    if shape == "circle":
        return CirclePillar(float(p["cx"]), float(p["cy"]), float(p["r"]))
    if shape == "ellipse":
        return EllipsePillar(float(p["cx"]), float(p["cy"]), float(p["a"]),
                             float(p["b"]), float(p.get("theta", 0.0)))
    if shape == "teardrop":
        return TeardropPillar(float(p["cx"]), float(p["cy"]),
                              float(p["r_front"]), float(p["k"]))
    raise ValueError(f"未知形状: {shape}")


def evaluate(submission_dir: str, data_dir: str) -> Dict[str, Any]:
    m: Dict[str, Any] = {
        "validity_score": 0.0, "quality_score": 0.0, "overall_score": 0.0, "error_info": {},
    }
    try:
        plan = os.path.join(submission_dir, PLAN_FILE)
        if not os.path.exists(plan):
            m["error_info"] = {"fatal": [f"缺 {PLAN_FILE}"]}
            return m
        with open(plan, encoding="utf-8-sig") as f:
            sol = json.load(f)
        raw = sol.get("pillars")
        if not isinstance(raw, list) or not raw:
            m["error_info"] = {"fatal": ["solution.json 缺 pillars 或为空"]}
            return m
        if len(raw) > MAX_PILLARS:
            m["error_info"] = {"fatal": [f"微柱数 {len(raw)} 超上限 {MAX_PILLARS}"]}
            return m

        pillars = []
        for i, p in enumerate(raw, 1):
            try:
                pillars.append(_build(p))
            except Exception as e:
                m["error_info"] = {"fatal": [f"第{i}根微柱参数非法: {e}"]}
                return m

        for i, p in enumerate(pillars, 1):
            for v in (p.cx, p.cy, p.bounding_radius()):
                if not np.isfinite(v):
                    m["error_info"] = {"fatal": [f"第{i}根微柱含非有限数值"]}
                    return m

        if not check_constraints(pillars):
            m["error_info"] = {"hard_violations": [
                "微柱布局违反约束: 特征尺寸须在 10-80μm、任何方向厚度≥10μm、离壁面≥10μm、"
                "柱间外沿间距≥10μm、须完全落在 500μm×200μm 通道内"]}
            return m

        m["validity_score"] = 1.0

        dx, dy = L_CHAN / NX, W_CHAN / NY
        mask = build_solid_mask(pillars, NX, NY, dx, dy)
        blocked = blocked_sections(mask, NX)
        if blocked:
            m["validity_score"] = 0.0
            m["error_info"] = {"hard_violations": [
                f"分离通道被堵塞: {len(blocked)} 个采样截面的流体占比低于 {MIN_FLUID_FRAC:.0%}"]}
            return m
        u, _v, pf = solve_stokes_brinkman(mask, nx=NX, ny=NY, L=L_CHAN, W=W_CHAN)
        nmrse, dp = compute_metrics(u, pf, mask, NX, NY, dy)
        perim = total_perimeter(pillars)

        s_nmrse = max(0.0, 1.0 - nmrse / NMRSE_REF)
        s_press = max(0.0, 1.0 - abs(dp) / DP_REF)
        s_perim = min(1.0, perim / PERIM_REF)
        score = 0.6 * s_nmrse + 0.2 * s_press + 0.2 * s_perim

        base = load_baseline()
        m["quality_score"] = round(score / base if base > 0 else 0.0, 6)
        m["overall_score"] = m["quality_score"]
        m["player_objective"] = round(float(score), 6)
        m["reference_value"] = base
        m["nmrse"] = round(float(nmrse), 6)
        m["delta_p_pa"] = round(float(dp), 4)
        m["total_perimeter_um"] = round(float(perim) * 1e6, 2)
        m["n_pillars"] = len(pillars)
        return m

    except Exception as e:
        m["validity_score"] = 0.0
        m["error_info"] = {"exception": str(e), "traceback": traceback.format_exc()[-800:]}
        return m


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--submission-dir", required=True)
    ap.add_argument("--data-dir", default=str(_HERE.parent / "data"))
    a = ap.parse_args()
    print(json.dumps(evaluate(a.submission_dir, a.data_dir), ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
