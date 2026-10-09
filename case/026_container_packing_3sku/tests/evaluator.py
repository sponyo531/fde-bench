"""
EXTRACTOR_SPEC:
  plan_file: solution.json
  required_columns: [placements, sku1_count, sku2_count, sku3_count]
  notes: >
    The agent must produce a JSON file describing where every box is placed inside
    the single container. Expected format:
      {
        "placements": [
          {"sku_id": "ITEM_A", "item_index": 1,
           "x": 0.0, "y": 0.0, "z": 0.0,        // left-front-bottom corner (m)
           "l": 1.09, "w": 0.5, "h": 0.885,      // occupied size after rotation (m)
           "rotated": false},                    // whether l/w were swapped
          ...
        ],
        "sku1_count": 66, "sku2_count": 8, "sku3_count": 132,
        "total_weight_kg": 10160.0, "space_utilization": 0.874
      }
    The list MUST contain exactly 66 ITEM_A + 8 ITEM_B + as many ITEM_C as packed.

    Common agent output patterns to handle:
    - a CSV "packing_detail.csv"/装箱明细 with columns
      SKUID/sku_id, x/y/z (or 坐标), l/w/h (or 放置长/宽/高), rotated/是否旋转
      -> build the placements list. Chinese "是否旋转": 是->true, 否->false.
    - boxes keyed lx/ly/lz instead of l/w/h -> rename (lz is the height h).
    - counts absent -> the evaluator derives them from placements; only placements
      are strictly required. Keep coordinates in meters, do NOT round away geometry.
    - upright rule: h must equal the SKU's original height (ITEM_A 0.885, ITEM_B 1.48,
      ITEM_C 0.615); (l,w) must be the SKU's (length,width) or the swapped pair.

    Use Bash with python3 to transform data if needed, then write solution.json
    to the output directory.

## Constraint validation (all recomputed independently from item specs)

| name | source | check |
|------|--------|-------|
| C1 ITEM_A==req     | data 数量 | count placements |
| C2 ITEM_B==req     | data 数量 | count placements |
| C3 weight<=max   | data 重量/最大载重 | sum piece weights |
| C4 in bounds     | data 集装箱长宽高 | x+l/y+w/z+h within, coords>=0 |
| C5 no overlap    | placements      | O(n^2) AABB test |
| C6 upright       | data SKU高度    | h == original height |
| C7 legal rotate  | data SKU长/宽    | (l,w) in {(L,W),(W,L)} |

## Scoring

Maximization objective: player_value = validated ITEM_C count.
quality = player_value / baseline_cost   (NO min(,1) clamp; >1 allowed).
Any hard-constraint violation -> validity=0, quality=0.
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

EPS = 1e-6


def _num(v):
    """xlsx cell -> float, treating blank/NaN as None."""
    if v is None:
        return None
    try:
        f = float(v)
    except (TypeError, ValueError):
        return None
    if f != f:  # NaN
        return None
    return f


def load_specs(data_dir):
    """Read container spec and per-SKU specs directly from load_data.xlsx.

    Single sheet '多货推荐1', header on row 1 (SKUID/GROUP/SKU数量/SKU单件体积/
    SKU长度/SKU宽度/SKU高度/放置状态/重量). SKU rows start on row 2; ITEM_C's 数量 is
    blank (-> required None). After some blank rows the same sheet holds the container
    block: a '集装箱信息' marker row, then a header row (名称/长/宽/高/最大载重), then
    the container data row. We locate that block by the '集装箱信息' marker."""
    import pandas as pd
    xlsx = os.path.join(data_dir, "load_data.xlsx")
    raw = pd.read_excel(xlsx, "多货推荐1", header=None)

    # ---- SKU table: header on row 0, map by column title ----
    header = [str(c).strip() if c is not None else "" for c in raw.iloc[0].tolist()]

    def col(*cands):
        for i, name in enumerate(header):
            for c in cands:
                if c in name:
                    return i
        raise KeyError(f"none of {cands} in {header}")

    ci_id = col('SKUID')
    ci_qty = col('SKU数量', '数量')
    ci_len = col('SKU长度', '长度')
    ci_wid = col('SKU宽度', '宽度')
    ci_hgt = col('SKU高度', '高度')
    ci_wt = col('重量')

    items = {}
    container_marker_row = None
    for r in range(1, len(raw)):
        sid = raw.iloc[r, ci_id]
        sid = str(sid).strip() if sid is not None else ""
        if sid == '集装箱信息':
            container_marker_row = r
            break
        if not sid or sid.lower() == 'nan':
            continue
        qty = _num(raw.iloc[r, ci_qty])
        items[sid] = {
            "length": _num(raw.iloc[r, ci_len]),
            "width": _num(raw.iloc[r, ci_wid]),
            "height": _num(raw.iloc[r, ci_hgt]),
            "weight": _num(raw.iloc[r, ci_wt]),
            "required": int(qty) if qty is not None else None,
        }

    # ---- container block: marker row, then header row, then data row ----
    if container_marker_row is None:
        raise KeyError("'集装箱信息' block not found in sheet")
    chdr = [str(c).strip() if c is not None else "" for c in
            raw.iloc[container_marker_row + 1].tolist()]

    def ccol(*cands):
        for i, name in enumerate(chdr):
            for c in cands:
                if c in name:
                    return i
        raise KeyError(f"none of {cands} in container header {chdr}")

    di_len = ccol('长')
    di_wid = ccol('宽')
    di_hgt = ccol('高')
    di_wt = ccol('最大载重', '载重')
    crow = raw.iloc[container_marker_row + 2]
    container = {
        "L": _num(crow[di_len]),
        "W": _num(crow[di_wid]),
        "H": _num(crow[di_hgt]),
        "max_weight": _num(crow[di_wt]),
    }
    return container, items


def load_baseline():
    with open(os.path.join(_HERE, "baseline", "reference_metrics.json")) as f:
        return float((lambda _d:_d.get("reference_value",_d.get("baseline_cost")))(json.load(f)))


def evaluate(file_path, data_dir):
    metrics = {"validity_score": 0.0, "quality_score": 0.0, "overall_score": 0.0, "error_info": {}}
    try:
        CONTAINER, ITEMS = load_specs(data_dir)
        baseline_cost = load_baseline()

        if not os.path.exists(file_path):
            metrics["error_info"] = {"fatal": [f"File not found: {file_path}"]}
            return metrics
        with open(file_path, "r", encoding="utf-8") as f:
            sub = json.load(f)

        placements = sub.get("placements")
        if not placements or not isinstance(placements, list):
            metrics["error_info"] = {"fatal": ["Missing or empty 'placements'"]}
            return metrics

        violations = []

        # ---- count SKUs ----
        counts = Counter()
        for p in placements:
            sid = p.get("sku_id")
            if sid in ITEMS:
                counts[sid] += 1
            else:
                violations.append(f"unknown sku_id={sid}")

        # C1 / C2 required counts (required counts read from item specs)
        for sid, spec in ITEMS.items():
            req = spec.get("required")
            if req is not None and counts[sid] != req:
                violations.append(f"required count: {sid}={counts[sid]} != {req}")

        # C3 total weight (independent recompute from item specs)
        total_weight = sum(ITEMS[p["sku_id"]]["weight"]
                           for p in placements if p.get("sku_id") in ITEMS)
        if total_weight > CONTAINER["max_weight"] + EPS:
            violations.append(f"C3: total_weight={total_weight:.1f} > {CONTAINER['max_weight']}")

        # C4 / C6 / C7 per-item geometry
        for i, p in enumerate(placements):
            sid = p.get("sku_id")
            if sid not in ITEMS:
                continue
            spec = ITEMS[sid]
            try:
                x, y, z = float(p["x"]), float(p["y"]), float(p["z"])
                l, w, h = float(p["l"]), float(p["w"]), float(p["h"])
            except (KeyError, TypeError, ValueError):
                violations.append(f"item[{i}] {sid} missing/invalid geometry fields")
                continue
            if x < -EPS or y < -EPS or z < -EPS:
                violations.append(f"C4: item[{i}] {sid} negative coord ({x},{y},{z})")
            if x + l > CONTAINER["L"] + EPS:
                violations.append(f"C4: item[{i}] {sid} x+l={x + l:.4f} > {CONTAINER['L']}")
            if y + w > CONTAINER["W"] + EPS:
                violations.append(f"C4: item[{i}] {sid} y+w={y + w:.4f} > {CONTAINER['W']}")
            if z + h > CONTAINER["H"] + EPS:
                violations.append(f"C4: item[{i}] {sid} z+h={z + h:.4f} > {CONTAINER['H']}")
            if abs(h - spec["height"]) > EPS:
                violations.append(f"C6: item[{i}] {sid} h={h} != {spec['height']} (upright)")
            ok_lw = ((abs(l - spec["length"]) < EPS and abs(w - spec["width"]) < EPS) or
                     (abs(l - spec["width"]) < EPS and abs(w - spec["length"]) < EPS))
            if not ok_lw:
                violations.append(
                    f"C7: item[{i}] {sid} (l={l},w={w}) not a rotation of "
                    f"({spec['length']},{spec['width']})")

        # C5 no overlap (O(n^2) AABB, independent recompute)
        n = len(placements)
        overlap_count = 0
        for i in range(n):
            pi = placements[i]
            try:
                xi1, yi1, zi1 = float(pi["x"]), float(pi["y"]), float(pi["z"])
                xi2, yi2, zi2 = xi1 + float(pi["l"]), yi1 + float(pi["w"]), zi1 + float(pi["h"])
            except (KeyError, TypeError, ValueError):
                continue
            for j in range(i + 1, n):
                pj = placements[j]
                try:
                    xj1, yj1, zj1 = float(pj["x"]), float(pj["y"]), float(pj["z"])
                    xj2, yj2, zj2 = xj1 + float(pj["l"]), yj1 + float(pj["w"]), zj1 + float(pj["h"])
                except (KeyError, TypeError, ValueError):
                    continue
                ox = min(xi2, xj2) - max(xi1, xj1)
                oy = min(yi2, yj2) - max(yi1, yj1)
                oz = min(zi2, zj2) - max(zi1, zj1)
                if ox > EPS and oy > EPS and oz > EPS:
                    overlap_count += 1
                    if overlap_count <= 3:
                        violations.append(
                            f"C5: item[{i}]({pi.get('sku_id')}) overlaps item[{j}]"
                            f"({pj.get('sku_id')}) by ({ox:.4f},{oy:.4f},{oz:.4f})")
        if overlap_count > 3:
            violations.append(f"C5: ... and {overlap_count - 3} more overlaps (total={overlap_count})")

        if violations:
            metrics["error_info"] = {"constraint": violations[:8]}
            return metrics

        metrics["validity_score"] = 1.0

        # ---- quality: maximization of ITEM_C count (recomputed) ----
        sku3_count = counts["ITEM_C"]
        container_vol = CONTAINER["L"] * CONTAINER["W"] * CONTAINER["H"]
        used_vol = sum(float(p["l"]) * float(p["w"]) * float(p["h"])
                       for p in placements if p.get("sku_id") in ITEMS)
        util = used_vol / container_vol if container_vol > 0 else 0.0

        quality = sku3_count / baseline_cost if baseline_cost > 0 else 0.0
        metrics["quality_score"] = round(quality, 4)
        metrics["overall_score"] = round(quality, 4)
        metrics["sku3_count"] = sku3_count
        metrics["sku1_count"] = counts["ITEM_A"]
        metrics["sku2_count"] = counts["ITEM_B"]
        metrics["total_weight_kg"] = round(total_weight, 2)
        metrics["space_utilization"] = round(util, 4)
        metrics["baseline_cost"] = baseline_cost

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
