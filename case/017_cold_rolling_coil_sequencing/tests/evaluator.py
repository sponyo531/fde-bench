"""
EXTRACTOR_SPEC:
  plan_file: solution.json
  required_columns: [sequence]
  notes: >
    The agent must produce a JSON file giving the optimized production order of
    all 81 coils. Expected format:
      {
        "sequence": ["COIL_001", "COIL_002", ...]   // 81 出口材料号, in optimized order
      }
    Optionally the agent may include a "metrics" object with its self-reported
    counts, but the evaluator IGNORES it and recomputes every metric independently
    from data/work_plan.csv.

    Common agent output patterns to normalize into solution.json:
    - a CSV (e.g. 优化排序结果.csv / result.csv) with an order column
      (新优化顺序 / 优化后顺序 / order) and a material-number column
      (出口材料号 / IN_MAT_NO / index / mat_no): sort by the order column and take
      the material-number column as the sequence.
    - a result.json with key "sequence" already present -> copy as-is.
    - a bare JSON list of 81 material numbers -> wrap as {"sequence": [...]}.

    Material numbers are strings (e.g. "COIL_001"); keep them as strings,
    strip whitespace. Use Bash with python3 to transform if needed, then write
    solution.json into the output directory.
"""

import argparse
import csv
import json
import os
import traceback
from pathlib import Path

PLAN_FILE = "solution.json"
_HERE = os.path.dirname(os.path.abspath(__file__))
_DATA = os.path.join(_HERE, "..", "data")

# Clean data uses an explicit anonymous grade catalog.  The catalog is the
# sole source of category/roll/front-width semantics; grade IDs themselves
# are opaque and must never be interpreted by their spelling.
def load_grade_catalog(data_dir):
    path = os.path.join(data_dir, "grade_catalog.csv")
    out = {}
    with open(path, encoding="utf-8-sig", newline="") as f:
        for row in csv.DictReader(f):
            gid = str(row["grade_id"]).strip()
            fw = str(row.get("front_width_mm", "")).strip()
            out[gid] = {
                "category": str(row.get("category_id", "")).strip(),
                "is_small": str(row.get("roll_size", "")).strip().lower() == "small",
                "front_width": float(fw) if fw else None,
            }
    return out


# Categories marked ``is_transition=true`` in the protected catalog.
TRANSITION_CATS = {'CATEGORY_002', 'CATEGORY_004'}

# Preserve the special exception without exposing source grade names:
# CATEGORY_001↔CATEGORY_004 is explicitly forbidden, while
# CATEGORY_001↔CATEGORY_002 remains allowed.  These anonymous categories are
# the stable tokens in the protected clean catalog.
BLOCKED_LINKAGE_PAIRS = {
    frozenset({'CATEGORY_001', 'CATEGORY_004'}),
}


def is_linkage_allowed(cat_a, cat_b):
    """相邻两卷的钢种大类能否直接衔接。

    口径取自 data 里真实系统判过的 `优化前/后违规类型`: 本数据 80 组相邻对覆盖
    12 种大类切换, 下面的规则与其 12/12 吻合。
      · 同大类连排一律可以;
      · 过渡类别与其它大类**双向**可切，但 CATEGORY_001↔CATEGORY_004 是保留的特殊禁配；
        CATEGORY_001 仅可与 CATEGORY_002 跨类衔接；
      · 非过渡类别之间直接切换计为违规。
    """
    if cat_a == cat_b:
        return True
    if frozenset({cat_a, cat_b}) in BLOCKED_LINKAGE_PAIRS:
        return False
    if cat_a in TRANSITION_CATS or cat_b in TRANSITION_CATS:
        return True
    return False


def check_thickness_jump(t1, t2, cat1, cat2):
    jump = abs(t1 - t2)
    t_ref = max(t1, t2)
    if 0.60 <= t_ref < 1.20:
        return jump > 0.250
    elif 1.20 <= t_ref < 1.60:
        return jump > 0.350
    elif t_ref >= 1.60:
        return jump > (0.500 if cat1 == cat2 else 0.300)
    return False


WIDTH_RULES = [
    (0.30, 0.60, 'N', 180),
    (0.60, 0.80, 'W', 165),
    (0.60, 0.80, 'N', 250),
    (0.80, 1.00, 'W', 200),
    (0.80, 1.00, 'N', 260),
    (1.00, 2.50, 'W', 250),
    (1.00, 2.50, 'N', 260),
]


def check_width_jump(w1, w2, t1, trim1):
    jump = abs(w1 - w2)
    trim_code = 'W' if trim1 == '切边' else 'N'
    for t_min, t_max, t_code, max_w in WIDTH_RULES:
        if t_min <= t1 < t_max and t_code == trim_code:
            return jump > max_w
    return False


def load_coils(data_dir):
    """从 CSV 加载所有钢卷信息，以出口材料号为键（独立数据源，不信任 submission）"""
    path = os.path.join(data_dir, "work_plan.csv")
    coils = {}
    catalog = load_grade_catalog(data_dir)
    with open(path, encoding="utf-8-sig", newline="") as f:
        for row in csv.DictReader(f):
            mat_no = str(row['出口材料号']).strip()
            grade = row['钢种'].strip()
            if grade not in catalog:
                raise ValueError(f"钢种 ID {grade!r} 未在 grade_catalog.csv 中定义")
            meta = catalog[grade]
            thickness = float(row['出口材料厚度'])
            width = float(row['出口材料宽度'])
            trim = row['优化前切边标记'].strip() if row['优化前切边标记'] else '不切边'
            coils[mat_no] = {
                'mat_no': mat_no,
                'steel_grade': grade,
                'category': meta['category'],
                'thickness': thickness,
                'width': width,
                'trim': trim,
                'is_small': meta['is_small'],
                'front_width': meta['front_width'],
            }
    return coils


def evaluate_sequence(sequence, coils):
    """独立重算所有指标（不读取、不信任 submission 自报的 metrics）"""
    cl = [coils[m] for m in sequence]
    sg = th = wd = winc = 0
    for i in range(1, len(cl)):
        p, c = cl[i - 1], cl[i]
        if p['category'] != c['category'] and not is_linkage_allowed(p['category'], c['category']):
            sg += 1
        if check_thickness_jump(p['thickness'], c['thickness'], p['category'], c['category']):
            th += 1
        if check_width_jump(p['width'], c['width'], p['thickness'], p['trim']):
            wd += 1
        if c['width'] > p['width']:
            winc += 1
    gg = 1
    for i in range(1, len(cl)):
        if cl[i]['steel_grade'] != cl[i - 1]['steel_grade']:
            gg += 1
    tv = sg + th + wd
    return {
        'steel_grade_violations': sg,
        'thickness_violations': th,
        'width_violations': wd,
        'total_violations': tv,
        'width_increases': winc,
        'grade_groups': gg,
        # 惩罚值（越小越好，用于 quality 计算）。
        # 权重按各层取值上限倒推，严格保证字典序优先级 违规 >> 宽度回升 >> 钢种分组：
        #   分组 grade_groups ≤ 81 → 宽度权重 100 (>81)
        #   宽度 width_increases ≤ 80，×100=8000，+分组81 < 10000 → 违规权重 10000
        'penalty': tv * 10000 + winc * 100 + gg,
    }


def check_hard_constraints(sequence, coils, all_mat_nos):
    errors = []
    # 1. 完整性
    if len(sequence) != len(all_mat_nos):
        errors.append(f"完整性违反: 输出{len(sequence)}卷，应为{len(all_mat_nos)}卷")
    seq_set = set(sequence)
    missing = set(all_mat_nos) - seq_set
    extra = seq_set - set(all_mat_nos)
    if missing:
        errors.append(f"缺失材料号: {list(missing)[:5]}")
    if extra:
        errors.append(f"多余材料号: {list(extra)[:5]}")
    if len(sequence) != len(seq_set):
        errors.append("存在重复材料号")
    if errors:
        return errors

    cl = [coils[m] for m in sequence]
    # 2. 大小辊分离（切换 <= 1 次）
    sm = [c['is_small'] for c in cl]
    trans = sum(1 for i in range(1, len(sm)) if sm[i] != sm[i - 1])
    if trans > 1:
        errors.append(f"大小辊分离违反: {trans}次切换（应≤1次）")
    # 3. Catalog-defined front-width coils must occupy the prefix.
    pos = [i for i, c in enumerate(cl)
           if c['front_width'] is not None and c['width'] == c['front_width']]
    if pos:
        k = len(pos)
        if set(pos) != set(range(k)):
            errors.append(f"前置宽度卷违反: 该{k}卷须占据序列最前面(位置0..{k-1}),实际最大位置{max(pos)}")
    return errors


def load_baseline():
    with open(os.path.join(_HERE, "baseline", "reference_metrics.json")) as f:
        d = json.load(f)
        return float(d.get("reference_value", d.get("baseline_cost")))


def evaluate(file_path, data_dir):
    metrics = {"validity_score": 0.0, "quality_score": 0.0, "overall_score": 0.0, "error_info": {}}
    try:
        coils = load_coils(data_dir)
        all_mat_nos = list(coils.keys())
        baseline_cost = load_baseline()

        if not os.path.exists(file_path):
            metrics["error_info"] = {"fatal": [f"File not found: {file_path}"]}
            return metrics
        with open(file_path, "r", encoding="utf-8") as f:
            sub = json.load(f)

        sequence = sub.get("sequence")
        if not sequence or not isinstance(sequence, list):
            metrics["error_info"] = {"fatal": ["Missing or empty 'sequence'"]}
            return metrics
        sequence = [str(m).strip() for m in sequence]

        # 未知材料号（schema 级错误）
        bad = [m for m in set(sequence) if m not in coils]
        if bad:
            metrics["error_info"] = {"schema": [f"unknown material numbers: {bad[:5]}"]}
            return metrics

        # ── 硬约束（独立校验）──
        errors = check_hard_constraints(sequence, coils, all_mat_nos)
        if errors:
            metrics["error_info"] = {"constraint": errors[:5]}
            return metrics

        metrics["validity_score"] = 1.0

        # ── 独立重算目标值（penalty，越小越好）──
        m = evaluate_sequence(sequence, coils)
        player_penalty = m['penalty']

        # 最小化目标：quality = baseline / 选手；选手优于 baseline 时 >1（不截断）
        quality = baseline_cost / player_penalty if player_penalty > 0 else 0.0
        metrics["quality_score"] = round(quality, 4)
        metrics["overall_score"] = round(quality, 4)
        metrics["penalty"] = player_penalty
        metrics["baseline_cost"] = baseline_cost
        metrics["total_violations"] = m['total_violations']
        metrics["steel_grade_violations"] = m['steel_grade_violations']
        metrics["thickness_violations"] = m['thickness_violations']
        metrics["width_violations"] = m['width_violations']
        metrics["width_increases"] = m['width_increases']
        metrics["grade_groups"] = m['grade_groups']

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
