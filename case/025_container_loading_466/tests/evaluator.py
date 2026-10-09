"""
EXTRACTOR_SPEC:
  plan_file: solution.json
  required_columns: [placements, fillers]
  notes: >
    The agent must produce a JSON file describing the 3D placement of all 466 cargo
    items inside a 40ft container (inner 1203 x 235 x 239 cm). Expected format:
      {
        "placements": [
          {"cargo_id": 122, "x": 0, "y": 0, "z": 0, "lx": 120, "ly": 100, "lz": 92},
          ...  // one object per physical item, 466 total
        ],
        "fillers": [
          {"type": "防滑垫", "location": "托盘底部与柜底之间", "purpose": "防止托盘滑移"},
          ...
        ],
        "summary": { ... optional, ignored by the evaluator ... }
      }
    Field meaning (all in cm, integers or floats):
      cargo_id  -> one of {121,122,123,124,125,126,128,129}
      x,y,z     -> the min corner of the box. The container has ONE door at the +x end
                   (x = 1203 = the door / open face); x = 0 is the closed inner end.
                   y along width 0..235, z is height, 0 = floor.
      lx,ly,lz  -> the box side lengths along x,y,z (must be a rotation of the cargo's
                   original L/W/H)
      fillers   -> a non-empty list. Each item must state the filler type, where it is
                   placed, and what it is used for. Normalize Chinese field names such
                   as 填充物/位置/作用 to type/location/purpose.

    Common agent output patterns to handle:
    - placements nested under a top-level key other than "placements" (e.g. "items",
      "boxes", "loading_plan") -> rename to "placements".
    - per-item center coordinates instead of min corner -> convert cx-lx/2, etc.
    - a CSV of placements with columns cargo_id,x,y,z,lx,ly,lz -> build the list.
    - filler notes written as prose or under keys such as padding/dunnage/filler_notes
      -> extract each described filler into the canonical fillers list.
    - the solver's self-reported score/utilization -> irrelevant; the evaluator
      recomputes everything from geometry.

    Use Bash with python3 to transform data if needed, then write solution.json
    to the output directory.
"""

import argparse
import json
import os
import traceback
from collections import Counter
from pathlib import Path

PLAN_FILE = "solution.json"
_HERE = os.path.dirname(os.path.abspath(__file__))
_DATA = os.path.join(_HERE, "..", "data")

# ── 集装箱内尺寸（40ft 标准柜规格常量，非 xlsx 业务数据；独立基准，不信任 solver 自报）
CL, CW, CH = 1203, 235, 239

# 货物清单原始文件（data/ 下的中文原名，单 sheet「Sheet1」）
MANIFEST_FILE = "cargo_manifest.xlsx"
MANIFEST_SHEET = "Sheet1"


def _to_num(v):
    """把单元格值转 float；某些尺寸存成字符串（如 '41.0'）需 strip 后转。"""
    if isinstance(v, str):
        v = v.strip()
    return float(v)


def load_cargo_manifest(data_dir):
    """从 data/cargo_manifest.xlsx 解析 8 品类的尺寸/单重/件数。

    该表格式坑：
      - 表头两行合并（第2行才给「长/宽/高」子表头）；
      - 序号跳号（缺7号，品名27缺失），行间夹全空行；
      - cargo 129 宽度存为字符串 '41.0'。
    因此按「有效行」取数：cargo_id 与长宽高三列均非空且可转数值才算一行，
    不依赖行号；表头行/空行因无法转数值被自动跳过。

    列布局（1-based）：1序号 2品名(cargo_id) 3颜色 4包装 5件数 6单重 7长 8宽 9高 10备注。
    """
    import openpyxl
    path = os.path.join(data_dir, MANIFEST_FILE)
    wb = openpyxl.load_workbook(path, data_only=True)
    ws = wb[MANIFEST_SHEET]

    dims, weight, counts = {}, {}, {}
    for row in ws.iter_rows(min_row=1, values_only=True):
        cid_raw = row[1]                    # 品名 = cargo_id
        cnt_raw, wt_raw = row[4], row[5]    # 件数 / 单重
        L, W, H = row[6], row[7], row[8]    # 长/宽/高(cm)
        if cid_raw is None or L is None or W is None or H is None:
            continue                         # 表头行 / 空行 / 缺尺寸行
        try:
            cid = int(_to_num(cid_raw))
            dims[cid] = sorted([_to_num(L), _to_num(W), _to_num(H)])
            weight[cid] = _to_num(wt_raw)
            counts[cid] = int(_to_num(cnt_raw))
        except (ValueError, TypeError):
            continue                         # 表头文本等无法转数值的行
    return dims, weight, counts


def load_baseline():
    with open(os.path.join(_HERE, "baseline", "reference_metrics.json")) as f:
        return float((lambda _d:_d.get("reference_value",_d.get("baseline_cost")))(json.load(f)))


def check_overlap(pl):
    """全量两两重叠检测（n<=466，O(n^2) 可接受）。返回违反件数。"""
    n = len(pl)
    count = 0
    for i in range(n):
        a = pl[i]
        for j in range(i + 1, n):
            b = pl[j]
            if not (a['x'] >= b['x']+b['lx'] or a['x']+a['lx'] <= b['x'] or
                    a['y'] >= b['y']+b['ly'] or a['y']+a['ly'] <= b['y'] or
                    a['z'] >= b['z']+b['lz'] or a['z']+a['lz'] <= b['z']):
                count += 1
    return count


def _xy_overlap(a, b):
    """两件货在 x-y 底面(与 z 无关)上的投影是否相交。"""
    return not (a['x'] >= b['x'] + b['lx'] or a['x'] + a['lx'] <= b['x'] or
                a['y'] >= b['y'] + b['ly'] or a['y'] + a['ly'] <= b['y'])


def check_floating(pl, tol=1.0):
    """支撑约束(后门单向装载的物理前提之一): 每件货物要么落地(z≈0),要么其底面
    必须"坐"在下方货物的顶面上——即存在正下方(顶面 z 与本件底面 z 相齐)且 x-y 底面
    投影相交的货物为其提供支撑。悬空(下方无任何接触面)的货物无法从后门沿 -x 推入到位,
    只能从上方硬塞,违反"只有后门进出"的限制。返回悬空(无任何支撑接触)的件数。"""
    bad = 0
    for a in pl:
        if a['z'] <= tol:            # 落地
            continue
        supported = False
        for b in pl:
            if b is a:
                continue
            if abs((b['z'] + b['lz']) - a['z']) <= tol and _xy_overlap(a, b):
                supported = True
                break
        if not supported:
            bad += 1
    return bad


def compute_quality(pl, cargo_weight):
    """独立重算质量分(不依赖 solver 自报值)。

    用户诉求:"空间利用率优先(硬性达标)+ 满足基本安全稳固"。利用率因 466 件必须全装而固定,
    已由"全覆盖"等硬约束保证;质量分只体现"基本安全稳固"——越稳越好。稳固性用重量加权重心
    高度 cog_z 衡量:重货在下、整体重心越低越不易倾覆,故 stability = 100×(1 - cog_z/CH)。
    不引入用户从未提及的重心 X/Y 平衡、分段紧凑度等指标。"""
    total_w = sum_wz = 0.0
    for p in pl:
        w = cargo_weight.get(p['cargo_id'], 0)
        total_w += w
        sum_wz += w * (p['z'] + p['lz'] / 2)
    cog_z = sum_wz / total_w if total_w > 0 else CH / 2

    stability = max(0.0, 100.0 * (1 - cog_z / CH))
    score = stability
    util = 100.0 * sum(p['lx'] * p['ly'] * p['lz'] for p in pl) / (CL * CW * CH)
    return {
        "score": round(score, 4),
        "stability": round(stability, 2),
        "cog_z": round(cog_z, 2),
        "weighted_avg_z": round(cog_z, 2),
        "utilization_pct": round(util, 2),
    }


def _valid_placement_fields(p):
    return all(k in p for k in ("cargo_id", "x", "y", "z", "lx", "ly", "lz"))


def _valid_fillers(fillers):
    """The instruction explicitly requires filler type, location and purpose."""
    if not isinstance(fillers, list) or not fillers:
        return False
    for item in fillers:
        if not isinstance(item, dict):
            return False
        for key in ("type", "location", "purpose"):
            if not isinstance(item.get(key), str) or not item[key].strip():
                return False
    return True


def evaluate(file_path, data_dir):
    metrics = {"validity_score": 0.0, "quality_score": 0.0, "overall_score": 0.0,
               "error_info": {}}
    try:
        baseline_cost = load_baseline()

        # ── 从 data/ 读货物清单（尺寸/单重/件数），不硬编码
        cargo_dims, cargo_weight, expected_counts = load_cargo_manifest(data_dir)

        if not os.path.exists(file_path):
            metrics["error_info"] = {"fatal": [f"File not found: {file_path}"]}
            return metrics
        with open(file_path, "r", encoding="utf-8") as f:
            sub = json.load(f)

        pl = sub.get("placements")
        if not pl or not isinstance(pl, list):
            metrics["error_info"] = {"fatal": ["Missing or empty 'placements'"]}
            return metrics
        for p in pl:
            if not _valid_placement_fields(p):
                metrics["error_info"] = {"schema": ["placement missing required field "
                                                    "(cargo_id/x/y/z/lx/ly/lz)"]}
                return metrics

        fillers = sub.get("fillers")
        if not _valid_fillers(fillers):
            metrics["error_info"] = {
                "schema": ["'fillers' must be a non-empty list; every item must contain "
                           "non-empty type/location/purpose"]
            }
            return metrics

        violations = []

        # ── 硬约束1: 货物件数完整性（独立计算）
        cnt = Counter(p['cargo_id'] for p in pl)
        for cid, exp in expected_counts.items():
            act = cnt.get(cid, 0)
            if act != exp:
                violations.append(f"货物{cid}件数错误: 期望{exp}, 实际{act}")
        for cid in cnt:
            if cid not in expected_counts:
                violations.append(f"未知 cargo_id: {cid}")

        # ── 硬约束2: 尺寸合法性（旋转排列）
        dim_bad = 0
        for p in pl:
            cid = p['cargo_id']
            if cid not in cargo_dims:
                continue
            if sorted([p['lx'], p['ly'], p['lz']]) != cargo_dims[cid]:
                dim_bad += 1
                if dim_bad <= 3:
                    violations.append(
                        f"cargo{cid} 尺寸非法: {p['lx']}x{p['ly']}x{p['lz']}, "
                        f"期望排列自{cargo_dims[cid]}")
        if dim_bad > 3:
            violations.append(f"... 共{dim_bad}件尺寸非法")

        # ── 硬约束3: 边界约束
        bnd = 0
        for p in pl:
            if p['x'] < 0 or p['y'] < 0 or p['z'] < 0:
                bnd += 1
            if p['x'] + p['lx'] > CL or p['y'] + p['ly'] > CW or p['z'] + p['lz'] > CH:
                bnd += 1
        if bnd > 0:
            violations.append(f"{bnd}件货物越界")

        # ── 硬约束4: 托盘落地（cargo 122 -> z==0）
        tray = sum(1 for p in pl if p['cargo_id'] == 122 and p['z'] != 0)
        if tray > 0:
            violations.append(f"{tray}个托盘未落地(z!=0)")

        # ── 硬约束5: 无重叠
        ov = check_overlap(pl)
        if ov > 0:
            violations.append(f"发现{ov}处货物重叠")

        # ── 硬约束6: 后门可达性——支撑约束(不悬空)
        #    只有后门(+x)进出:非落地货物必须坐在下方货物顶面上,不能悬空(否则只能从上方硬塞)
        fl = check_floating(pl)
        if fl > 0:
            violations.append(f"{fl}件货物悬空(无下方支撑),违反后门单向装载(不可从上方硬塞)")

        if violations:
            metrics["error_info"] = {"constraint": violations[:10]}
            metrics["violation_count"] = len(violations)
            metrics["total_items"] = len(pl)
            return metrics

        metrics["validity_score"] = 1.0

        # ── 独立重算质量分
        q = compute_quality(pl, cargo_weight)
        player_score = q["score"]
        # 最大化目标：quality = player / baseline（不加 min(,1) 截断）
        quality = player_score / baseline_cost if baseline_cost > 0 else 0.0
        metrics["quality_score"] = round(quality, 4)
        metrics["overall_score"] = round(quality, 4)
        metrics["player_score"] = player_score
        metrics["baseline_cost"] = baseline_cost
        metrics["total_items"] = len(pl)
        metrics["filler_items"] = len(fillers)
        metrics.update({k: q[k] for k in q if k != "score"})

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
