"""
EXTRACTOR_SPEC:
  plan_file: recommendation.json
  required_columns: null
  notes: >
    Agent must output recommendation.json (a JSON file, not CSV) containing
    EXACTLY 5 recipe candidates. Each candidate is an object with:
      - "rank": integer 1..5
      - "半成品名称": one of "BLEND_01", "BLEND_02", "BLEND_03"
      - "原材料用量": object mapping material name (string) to KG (number)

    Example minimal valid output:
      [
        {"rank": 1, "半成品名称": "BLEND_01",
         "原材料用量": {"M020": 320, "M001": 50, "M034": 50}},
        {"rank": 2, ...}, ..., {"rank": 5, ...}
      ]

    If the agent uses different field names ("formula" / "ingredients" /
    "materials" / "name"), map them to 半成品名称/原材料用量. Material
    names that are not in the original dataset's material list are dropped
    with a warning (not a hard fail).

    The task is to recommend 5 recipes meeting a fixed target spec written
    in instruction.md. The evaluator uses an internal oracle model to predict
    each recipe's 17 detection targets and check spec compliance.
"""

import argparse
import json
import math
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
import lightgbm as lgb

warnings.filterwarnings("ignore")

# ══════════════════════════════════════════════════════════════════════════════
# 内联 oracle（评估器私有正向性能模型；原 tests/oracle/oracle_predict.py）
#   现场从客户原始表训练、不落 pkl（benchmark 铁律2 禁 pkl/中间产物）：
#     - data/formula_bom.csv              (配方→原料 KG)     [agent 可见]
#     - tests/_private/batch_targets.csv (配方→7 性能目标)  [私有]
#   特征 = per-原料 KG 用量向量(抗泡剂 ML→0.0485 KG-equiv)；每目标一个 LightGBM。
# ══════════════════════════════════════════════════════════════════════════════
_ORACLE_HERE = Path(__file__).resolve().parent
_ORACLE_DATA = _ORACLE_HERE.parent / "data"
_ORACLE_PRIVATE = _ORACLE_HERE / "_private"

_CATALOG = None
def _material_catalog():
    global _CATALOG
    if _CATALOG is None:
        p = _ORACLE_DATA / "material_catalog.csv"
        if p.exists():
            c = pd.read_csv(p, dtype=str).fillna("")
            _CATALOG = {str(r.material_id): {"role": str(r.role), "grade": str(r.grade),
                        "treat_rate_pct": (float(r.treat_rate_pct) if str(r.treat_rate_pct) else None)}
                        for r in c.itertuples(index=False)}
        else:
            _CATALOG = {}
    return _CATALOG


def _load_baseline() -> float:
    with open(_ORACLE_HERE / "baseline" / "reference_metrics.json", encoding="utf-8") as f:
        return float(json.load(f)["reference_value"])

_ORACLE_TARGET_MAP = {
    "KV100": "运动粘度__100℃__",
    "CCS":   "低温动力粘度__-30℃__",
    "MRV":   "低温泵送粘度__在无屈服应力时、-35℃__",
    "倾点":  "倾点__倾点__",
    "HTHS":  "高温高剪切粘度__150℃、10⁶S⁻¹__",
    "TBN":   "碱值__以KOH计__",
}


def _oracle_bom_kg(row):
    qty = float(row["数量"])
    unit = str(row["单位"]).strip().upper()
    return 0.0485 if unit == "ML" else qty


def _train_oracle():
    """现场从客户原始表训练 oracle（每次评估调用；不落 pkl）。"""
    bom = pd.read_csv(_ORACLE_DATA / "formula_bom.csv")
    bt = pd.read_csv(_ORACLE_PRIVATE / "batch_targets.csv")

    bom = bom[bom["formula"].notna()].copy()
    bom["kg"] = bom.apply(_oracle_bom_kg, axis=1)
    wide = bom.pivot_table(index="formula", columns="原料", values="kg",
                           aggfunc="sum", fill_value=0.0)
    wide.columns.name = None
    material_cols = list(wide.columns)

    bt = bt[bt["formula"].notna() & (bt["formula"] != "F015")].copy()
    for col in _ORACLE_TARGET_MAP:
        bt[col] = pd.to_numeric(bt[col], errors="coerce")
    tgt = bt.groupby("formula")[list(_ORACLE_TARGET_MAP)].mean()

    df = wide.join(tgt, how="inner")
    X = df[material_cols].fillna(0.0).values

    models, target_cols = {}, []
    for col, tkey in _ORACLE_TARGET_MAP.items():
        y = df[col].values.astype(float)
        mask = ~np.isnan(y)
        if mask.sum() < 5:
            models[tkey] = {"_constant": float(np.nanmean(y)) if mask.any() else 0.0}
        else:
            m = lgb.LGBMRegressor(n_estimators=300, num_leaves=15, learning_rate=0.05,
                                  min_child_samples=3, random_state=42, n_jobs=1, verbose=-1)
            m.fit(X[mask], y[mask])
            models[tkey] = m
        target_cols.append(tkey)

    return {"models": models, "material_cols": material_cols,
            "onehot_cols": [], "target_cols": target_cols}


_ORACLE = None


def _oracle_get():
    global _ORACLE
    if _ORACLE is None:
        _ORACLE = _train_oracle()
    return _ORACLE


def _oracle_features(recipe, oracle=None):
    if oracle is None:
        oracle = _oracle_get()
    用量 = recipe.get("原材料用量", {}) or {}
    feat = [float(用量.get(m, 0.0)) for m in oracle["material_cols"]]
    return np.array(feat, dtype=float).reshape(1, -1)


def _oracle_predict(recipe, oracle=None):
    if oracle is None:
        oracle = _oracle_get()
    X = _oracle_features(recipe, oracle)
    out = {}
    for tcol, model in oracle["models"].items():
        if isinstance(model, dict) and "_constant" in model:
            out[tcol] = model["_constant"]
        else:
            out[tcol] = float(model.predict(X)[0])
    return out


def _oracle_materials(oracle=None):
    if oracle is None:
        oracle = _oracle_get()
    return list(oracle["material_cols"])


def _oracle_products(oracle=None):
    return []

# ── 目标规格（agent 不可见；instruction.md 里描述给 agent 看）─────────────────
# 产品固定 SP 5W-30；agent 按原材料 KG 用量设计配方（无 半成品 one-hot）。
HALF_PRODUCT_TARGET = None  # 不约束 半成品名称（本 case 按原料用量设计）
N_RECOMMENDATIONS = 5

# Spec format: target_key -> (kind, lo, hi, comfort)
#   kind="range": value in [lo, hi]; quality = 1 at center, 0 at boundary.
#   kind="max":   value <= hi; comfort (< hi) is where quality = 1.0.
#   kind="min":   value >= lo; comfort (> lo) is where quality = 1.0.
#
# Spec 取自客户需求表(目标需求.png)的 SP 5W-30 规格。温度口径按内部修正
# (CCS -30℃ / MRV -35℃; 需求表 -25/-30 是 10W 档笔误)。comfort 按 5W-30 真实
# 性能分布(P10/中位)校准，使"深度达标"得满分。
# 碱值(TBN≥7.5) 未入硬命中: TBN 训练样本极稀(全5W-30仅19个)、oracle 不可靠，
# 且近半数真实合格配方 oracle 预测 <7.5(客户规格与产线有系统性出入)—作方向性参考。
TARGET_SPEC = {
    "运动粘度__100℃__":            ("range", 10.8, 11.8,   None),   # 客户 10.8~11.8
    "低温动力粘度__-30℃__":        ("max",   None, 6300.0, 4862.0),  # 客户≤6300(CCS)
    "低温泵送粘度__在无屈服应力时、-35℃__": ("max", None, 60000.0, 17800.0),  # 客户≤60000(MRV)
    "倾点__倾点__":                ("max",   None, -39.0,  -48.0),   # 客户≤-39
    "高温高剪切粘度__150℃、10⁶S⁻¹__": ("min",  2.9,  None,  3.4),     # 客户≥2.9(HTHS)
}

# ── 硬结构约束（POC3 配方逻辑；违反 → validity=0）──────────────────────────────
# 基础油按 KV100 分档（源自 clean 数据中的原材料检测字段）。
LIGHT_OILS = {"M020", "M022", "M041", "M049", "M053"}      # KV100<4.8
HEAVY_OILS = {"M016", "M035", "M038", "M039", "M048", "M051"}  # KV100>6.5
# 复合剂(DI包)按厂商 treat rate 定值投加，非可调变量；用量必须落在锁定值附近。
COMPLEX_DOSE = {"M054": 6.88, "M058": 7.58, "M059": 8.30,
                "M060": 7.89, "M061": 9.40, "M062": 7.68}  # % of total
COMPLEX_DOSE_TOL = 1.5  # 复合剂占比允许偏离锁定值 ±1.5 个百分点
# 结构比例约束(% of total mass)，边界已用 26 个真实 5W-30 配方校准(全部通过)。
LIGHT_FRAC_RANGE = (16.0, 29.0)   # 轻档基础油必含(低温5W & 压Noack)
HEAVY_FRAC_RANGE = (2.0, 14.0)    # 重档基础油(保HTHS/助VII减负)
MAX_SINGLE_OIL_FRAC = 63.0        # 单一基础油占比上限(防越界外推)

# Validity bounds based on training data percentiles (KG total per recipe).
TOTAL_USAGE_MIN = 200.0
TOTAL_USAGE_MAX = 2000.0

# Score weighting: how much each dimension contributes to overall quality_score (0-100).
WEIGHT_HIT_RATE = 60
WEIGHT_QUALITY = 20
WEIGHT_DIVERSITY = 20


def load_recommendations(plan_path: Path):
    """Load and lightly normalize agent's recommendation.json.

    Accepts either a JSON list or a JSON object with a top-level array under
    common keys ('recommendations', 'recipes', 'candidates', 'top5').
    """
    with plan_path.open("r", encoding="utf-8-sig") as f:
        data = json.load(f)
    if isinstance(data, dict):
        for k in ("recommendations", "recipes", "candidates", "top5", "results"):
            if k in data and isinstance(data[k], list):
                data = data[k]
                break
        else:
            raise ValueError(
                "recommendation.json must be a list or contain one under "
                "'recommendations' / 'recipes' / 'candidates' / 'top5' / 'results'."
            )
    if not isinstance(data, list):
        raise ValueError("recommendation.json top-level must be a list.")
    return data


# 抗泡剂缺失时的补齐值。M063 是主流品种（91 个配方里 86 个用它，ML 计量），
# 0.0485 是 _oracle_bom_kg() 对所有 ML 计量返回的固定 KG-equiv。
_FOAM_DEFAULT_NAME = "M063"
_FOAM_DEFAULT_KG = 0.0485


def normalize_recipe(raw: dict, known_materials: set, known_products: set):
    """Light fuzzy mapping of agent field names; return (recipe, warnings)."""
    warnings = []

    # Half-product (optional in this case — product is fixed to SP 5W-30, the
    # recipe is defined by material usage; we only record it if provided).
    product = None
    for k in ("半成品名称", "半成品", "product", "half_product", "name", "type"):
        if k in raw:
            v = raw[k]
            if isinstance(v, str):
                product = v
                break
    if known_products and product is not None and product not in known_products:
        return None, [f"Unknown 半成品名称={product!r}"]

    # Materials
    materials_raw = None
    for k in ("原材料用量", "原材料", "materials", "ingredients", "formula", "bom", "composition"):
        if k in raw and isinstance(raw[k], dict):
            materials_raw = raw[k]
            break
    if not materials_raw:
        return None, ["Missing or empty 原材料用量 / materials object"]

    materials = {}
    for name, kg in materials_raw.items():
        try:
            kg_f = float(kg)
        except (TypeError, ValueError):
            warnings.append(f"Skipping non-numeric usage for {name!r}: {kg!r}")
            continue
        if math.isnan(kg_f) or math.isinf(kg_f):
            warnings.append(f"Skipping NaN/Inf usage for {name!r}")
            continue
        if kg_f < 0:
            warnings.append(f"Skipping negative usage for {name!r}: {kg_f}")
            continue
        if name not in known_materials:
            warnings.append(f"Unknown material {name!r} (dropped)")
            continue
        materials[name] = kg_f

    # 抗泡剂缺失则按标准痕量值补齐，不判无效。
    #
    # 为什么：抗泡剂在这份数据里是**恒定痕量项**——91/91 配方全含它，而
    # _oracle_bom_kg() 对 ML 计量的一律返回 0.0485（50 ML 和 80 ML 映射到同一个
    # 值），所以它在 oracle 的特征向量里是一列常数，对性能预测零贡献。
    # 更要紧的是 information.md:17 明确告诉 agent「抗泡剂单位是 ML 不是 KG，
    # 不要当成 KG 参与质量核算」——agent 照做，把它排除在质量平衡之外，
    # 于是它的配方对象里就没有这一项。而提交格式要的是「原料→KG」全量清单，
    # 两者对不上，纯属填表口径问题，不是配方设计缺陷。
    #
    # 实测：某 run 的三个抽取器对同一份 agent 解，两个补了抗泡剂、一个没补，
    # 得分 0.318116 vs 0.0。给没补的那份**只加一个 0.0485**、其余数值一字不动，
    # 分数就变成 0.318116——完全相同。可见这 0.318116 全部来自另外 11 种原料，
    # 抗泡剂唯一的作用是翻过 validity 这道闸门。用它把一个配比合格的方案判成
    # 0 分，惩罚的是漏填而非配方本身。
    if not any(_material_catalog().get(m, {}).get("role") == "antifoam" for m in materials):
        if _FOAM_DEFAULT_NAME in known_materials:
            materials[_FOAM_DEFAULT_NAME] = _FOAM_DEFAULT_KG
            warnings.append(
                f"未提供抗泡剂，按标准痕量值补齐 {_FOAM_DEFAULT_NAME}="
                f"{_FOAM_DEFAULT_KG} KG（ML 计量的固定 KG-equiv，对性能预测无影响）")

    return {"半成品名称": product, "原材料用量": materials}, warnings


REQUIRED_ADDITIVES = ("viscosity_modifier", "pour_point_depressant", "antifoam")


def check_structural_constraints(recipe, idx):
    """POC3 hard formulation constraints on a single recipe (fractions of total mass).
    Returns a list of violation strings (empty = pass)."""
    errs = []
    用量 = recipe["原材料用量"]
    total = sum(用量.values())
    if total <= 0:
        return [f"Recipe #{idx+1}: total usage is zero."]

    roles = _material_catalog()
    base_oils = {m: kg for m, kg in 用量.items() if roles.get(m, {}).get("role") == "base_oil"}
    light = sum(kg for m, kg in base_oils.items() if m in LIGHT_OILS) / total * 100
    heavy = sum(kg for m, kg in base_oils.items() if m in HEAVY_OILS) / total * 100
    max_oil = (max(base_oils.values()) / total * 100) if base_oils else 0.0

    if not (LIGHT_FRAC_RANGE[0] <= light <= LIGHT_FRAC_RANGE[1]):
        errs.append(f"Recipe #{idx+1}: 轻档基础油占比 {light:.1f}% 不在 "
                    f"[{LIGHT_FRAC_RANGE[0]}, {LIGHT_FRAC_RANGE[1]}]% (低温5W必含轻档)。")
    if not (HEAVY_FRAC_RANGE[0] <= heavy <= HEAVY_FRAC_RANGE[1]):
        errs.append(f"Recipe #{idx+1}: 重档基础油占比 {heavy:.1f}% 不在 "
                    f"[{HEAVY_FRAC_RANGE[0]}, {HEAVY_FRAC_RANGE[1]}]%。")
    if max_oil > MAX_SINGLE_OIL_FRAC:
        errs.append(f"Recipe #{idx+1}: 单一基础油占比 {max_oil:.1f}% > "
                    f"{MAX_SINGLE_OIL_FRAC}% (防越界外推)。")

    # 复合剂: 恰好含一种，且用量锁定在厂商 treat rate 附近。
    complexes = {m: kg for m, kg in 用量.items() if roles.get(m, {}).get("role") == "di_package"}
    if len(complexes) != 1:
        errs.append(f"Recipe #{idx+1}: 需恰好含 1 种复合剂(DI包)，实含 {len(complexes)} 种。")
    else:
        cname, ckg = next(iter(complexes.items()))
        cfrac = ckg / total * 100
        locked = COMPLEX_DOSE.get(cname)
        if locked is None:
            errs.append(f"Recipe #{idx+1}: 未知复合剂 {cname!r}。")
        elif abs(cfrac - locked) > COMPLEX_DOSE_TOL:
            errs.append(f"Recipe #{idx+1}: 复合剂 {cname} 占比 {cfrac:.2f}% 偏离锁定 "
                        f"treat rate {locked}% 超过 ±{COMPLEX_DOSE_TOL}pp (DI包按定值投加)。")

    # 添加剂齐套: 粘指剂撑高温粘度、降凝剂压倾点、抗泡剂消泡, 缺一不成配方。
    # 不查的话, 只有基础油 + 复合剂的"配方"也能判合格。
    # 注: 抗泡剂一项已在 normalize_recipe 里缺失自动补齐（恒定痕量、对预测无贡献，
    # 见那里的说明），正常路径下不会在这里触发；保留是为了补齐失败时仍有兜底。
    for kind in REQUIRED_ADDITIVES:
        if not any(roles.get(m, {}).get("role") == kind and kg > 0 for m, kg in 用量.items()):
            errs.append(f"Recipe #{idx+1}: 缺少{kind}。")
    return errs


def check_validity(recipes):
    """Hard validity checks. Returns errors_list (empty = valid)."""
    errors = []
    if len(recipes) != N_RECOMMENDATIONS:
        errors.append(
            f"Expected exactly {N_RECOMMENDATIONS} recipes, got {len(recipes)}."
        )
        return errors

    for i, r in enumerate(recipes):
        if r is None:
            errors.append(f"Recipe #{i+1} is None (failed normalization).")
            continue
        if HALF_PRODUCT_TARGET is not None and r["半成品名称"] != HALF_PRODUCT_TARGET:
            errors.append(
                f"Recipe #{i+1}: 半成品名称={r['半成品名称']!r}, expected "
                f"{HALF_PRODUCT_TARGET!r}."
            )
        if not r["原材料用量"]:
            errors.append(f"Recipe #{i+1}: empty material list.")
            continue
        total = sum(r["原材料用量"].values())
        if total < TOTAL_USAGE_MIN or total > TOTAL_USAGE_MAX:
            errors.append(
                f"Recipe #{i+1}: total usage {total:.1f} KG outside "
                f"[{TOTAL_USAGE_MIN}, {TOTAL_USAGE_MAX}]."
            )
        # POC3 hard structural constraints.
        errors.extend(check_structural_constraints(r, i))
    return errors


def check_spec_compliance(predicted: dict):
    """Return (n_targets_passed, n_total_targets, per_target_pass: dict)."""
    per_target = {}
    n_pass = 0
    for tkey, spec in TARGET_SPEC.items():
        kind, lo, hi = spec[0], spec[1], spec[2]
        pred = predicted.get(tkey)
        if pred is None:
            per_target[tkey] = False
            continue
        if kind == "range":
            ok = lo <= pred <= hi
        elif kind == "max":
            ok = pred <= hi
        elif kind == "min":
            ok = pred >= lo
        else:
            ok = False
        per_target[tkey] = ok
        if ok:
            n_pass += 1
    return n_pass, len(TARGET_SPEC), per_target


def compute_quality(predicted: dict):
    """For each spec target, compute a 'depth into spec' in [0, 1].

    Anchors:
      range: 1.0 at center, 0.0 at boundary, 0 outside (linear in distance/half-range)
      max:   1.0 at comfort (or beyond), 0.0 at threshold (hi), 0 above hi
      min:   1.0 at comfort (or beyond), 0.0 at threshold (lo), 0 below lo

    Comfort values are tuned against the training-set distribution so that
    "deeply in-spec" predictions get full credit (e.g. 倾点 quality=1 at <= -50).
    """
    scores = []
    for tkey, spec in TARGET_SPEC.items():
        kind, lo, hi, comfort = spec[0], spec[1], spec[2], spec[3]
        pred = predicted.get(tkey)
        if pred is None:
            scores.append(0.0); continue

        if kind == "range":
            center = (lo + hi) / 2.0
            half = (hi - lo) / 2.0
            if half <= 0:
                scores.append(1.0 if pred == center else 0.0); continue
            dist = abs(pred - center) / half
            scores.append(max(0.0, 1.0 - dist))

        elif kind == "max":
            if pred > hi:
                scores.append(0.0); continue
            if comfort is None or comfort >= hi:
                # Degenerate: no comfort defined, fall back to threshold-only
                scores.append(0.5 if pred <= hi else 0.0); continue
            margin = hi - comfort  # > 0 for "lower is better"
            slack = (hi - pred) / margin
            scores.append(min(1.0, max(0.0, slack)))

        elif kind == "min":
            if pred < lo:
                scores.append(0.0); continue
            if comfort is None or comfort <= lo:
                scores.append(0.5 if pred >= lo else 0.0); continue
            margin = comfort - lo  # > 0 for "higher is better"
            slack = (pred - lo) / margin
            scores.append(min(1.0, max(0.0, slack)))
        else:
            scores.append(0.0)
    return float(np.mean(scores)) if scores else 0.0


# Threshold above which two recipes are considered "near-duplicates" for hit_rate
# deduplication. ±5% noise on the same recipe gives cosine sim ≈ 0.99, so 0.95
# catches "复刻+轻微扰动" while letting genuinely-different formulas through.
SIM_THRESHOLD_DUPLICATE = 0.95


def _normalize_to_unit(recipes, all_materials):
    """Encode each recipe as a unit-sum proportion vector over all_materials."""
    vecs = []
    for r in recipes:
        v = np.array([r["原材料用量"].get(m, 0.0) for m in all_materials], dtype=float)
        s = v.sum()
        if s > 1e-9:
            v = v / s
        vecs.append(v)
    return vecs


def _cosine_sim(u, v):
    denom = float(np.linalg.norm(u) * np.linalg.norm(v))
    if denom <= 1e-9:
        return 0.0
    return float(np.dot(u, v) / denom)


def find_unique_recipes(recipes, all_materials, threshold=SIM_THRESHOLD_DUPLICATE):
    """Greedy dedup: keep first recipe of each near-duplicate cluster.

    Returns (kept_indices, duplicate_map) where duplicate_map[i] is the index
    of the cluster representative that recipe i was merged into.
    """
    vecs = _normalize_to_unit(recipes, all_materials)
    kept = []
    dup_map = {}
    for i in range(len(vecs)):
        rep = None
        for j in kept:
            if _cosine_sim(vecs[i], vecs[j]) > threshold:
                rep = j
                break
        if rep is None:
            kept.append(i)
            dup_map[i] = i
        else:
            dup_map[i] = rep
    return kept, dup_map


def compute_diversity(recipes, all_materials):
    """1 - mean(pairwise cosine similarity) of normalized recipes."""
    if len(recipes) < 2:
        return 0.0
    vecs = _normalize_to_unit(recipes, all_materials)
    sims = []
    for i in range(len(vecs)):
        for j in range(i + 1, len(vecs)):
            sims.append(_cosine_sim(vecs[i], vecs[j]))
    mean_sim = float(np.mean(sims)) if sims else 0.0
    diversity = 1.0 - mean_sim
    return max(0.0, min(1.0, diversity))


def evaluate(data_dir: Path, baseline_dir: Path, submission_dir: Path):
    oracle = _oracle_get()
    known_materials = set(_oracle_materials(oracle))
    known_products = set(_oracle_products(oracle))

    plan_path = submission_dir / "recommendation.json"
    if not plan_path.is_file():
        # Try nested paths
        cands = list(submission_dir.glob("**/recommendation.json"))
        if not cands:
            return {
                "validity_score": 0.0, "quality_score": 0.0, "overall_score": 0.0,
                "errors": [f"recommendation.json not found under {submission_dir}"],
            }
        plan_path = cands[0]

    try:
        raw_list = load_recommendations(plan_path)
    except (json.JSONDecodeError, ValueError) as exc:
        return {
            "validity_score": 0.0, "quality_score": 0.0, "overall_score": 0.0,
            "errors": [f"Failed to parse recommendation.json: {exc}"],
            "resolved_artifact": str(plan_path),
        }

    # Normalize each recipe
    normalized = []
    parse_warnings = []
    for i, raw in enumerate(raw_list):
        if not isinstance(raw, dict):
            normalized.append(None)
            parse_warnings.append(f"Recipe #{i+1} is not a JSON object.")
            continue
        recipe, ws = normalize_recipe(raw, known_materials, known_products)
        normalized.append(recipe)
        for w in ws:
            parse_warnings.append(f"Recipe #{i+1}: {w}")

    # Validity
    errors = check_validity(normalized)
    if errors:
        return {
            "validity_score": 0.0, "quality_score": 0.0, "overall_score": 0.0,
            "errors": errors, "warnings": parse_warnings,
            "resolved_artifact": str(plan_path),
        }

    # Oracle predict + per-recipe spec check
    per_recipe = []
    for i, r in enumerate(normalized):
        pred = _oracle_predict(r, oracle=oracle)
        n_pass, n_total, per_target = check_spec_compliance(pred)
        full_hit = (n_pass == n_total)
        quality = compute_quality(pred)
        # EXTRACTOR_SPEC 要求每个候选带 rank 1..5。保留 agent 提交的 rank(校验范围),
        # 缺省或非法时退回到位置序号, 不再无条件用 i+1 覆盖掉 agent 的排名。
        raw_rank = raw_list[i] if isinstance(raw_list[i], dict) else None
        rank = None
        if isinstance(raw_rank, dict):
            rv = raw_rank.get("rank")
            if rv is not None:
                try:
                    rv = int(rv)
                except (TypeError, ValueError):
                    rv = None
                if isinstance(rv, int) and 1 <= rv <= 5:
                    rank = rv
                else:
                    parse_warnings.append(f"Recipe #{i+1}: rank={rv!r} 非法, 回退到位置序号")
        per_recipe.append({
            "rank": rank or (i + 1),
            "半成品名称": r["半成品名称"],
            "predicted_targets": {k: round(v, 4) for k, v in pred.items()},
            "spec_pass_count": n_pass,
            "spec_total": n_total,
            "spec_per_target": per_target,
            "full_hit": full_hit,
            "in_spec_quality": round(quality, 4),
        })

    # Dedup: cluster near-duplicate recipes (cos_sim > SIM_THRESHOLD_DUPLICATE).
    # hit_rate uses dedup'd cluster representatives, so 5-份雷同 collapses to 1.
    sorted_materials = sorted(known_materials)
    kept_idx, dup_map = find_unique_recipes(normalized, sorted_materials)
    n_unique = len(kept_idx)
    n_duplicate = N_RECOMMENDATIONS - n_unique

    # Aggregate using dedup'd hits
    raw_full_hits = sum(1 for r in per_recipe if r["full_hit"])
    unique_full_hits = sum(1 for i in kept_idx if per_recipe[i]["full_hit"])
    hit_rate = unique_full_hits / N_RECOMMENDATIONS  # 0..1, denom stays 5

    # Quality: average over UNIQUE candidates that fully hit
    if unique_full_hits > 0:
        quality = float(np.mean(
            [per_recipe[i]["in_spec_quality"] for i in kept_idx if per_recipe[i]["full_hit"]]
        ))
    else:
        # If no unique full hits, give partial credit using all candidates' quality
        quality = float(np.mean([r["in_spec_quality"] for r in per_recipe])) * 0.3

    diversity = compute_diversity(normalized, sorted_materials)

    overall = (
        WEIGHT_HIT_RATE * hit_rate
        + WEIGHT_QUALITY * quality
        + WEIGHT_DIVERSITY * diversity
    )

    # higher_is_better: quality = 选手绝对分 / baseline，不加 min(,1) 截断
    _baseline = _load_baseline()
    quality_norm = overall / _baseline if _baseline > 0 else 0.0

    return {
        "validity_score": 1.0,
        "quality_score": round(quality_norm, 6),
        "overall_score": round(quality_norm, 6),
        "player_absolute_score": round(overall, 2),
        "reference_value": _baseline,
        "errors": [],
        "warnings": parse_warnings,
        "resolved_artifact": str(plan_path),
        "n_unique": n_unique,
        "n_duplicate": n_duplicate,
        "duplicate_map": {str(k): str(v) for k, v in dup_map.items()},
        "raw_full_hits": raw_full_hits,
        "unique_full_hits": unique_full_hits,
        "hit_rate": round(hit_rate, 4),
        "in_spec_quality": round(quality, 4),
        "diversity": round(diversity, 4),
        "weights": {
            "hit_rate": WEIGHT_HIT_RATE,
            "quality": WEIGHT_QUALITY,
            "diversity": WEIGHT_DIVERSITY,
        },
        "target_spec": {
            k: {"kind": v[0], "lo": v[1], "hi": v[2], "comfort": v[3]}
            for k, v in TARGET_SPEC.items()
        },
        "half_product_target": HALF_PRODUCT_TARGET,
        "per_recipe": per_recipe,
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-dir", type=Path, required=True)
    parser.add_argument(
        "--baseline-dir",
        default=Path(__file__).resolve().parent / "baseline",
        type=Path,
    )
    parser.add_argument("--submission-dir", type=Path, required=True)
    args = parser.parse_args()

    result = evaluate(
        args.data_dir.resolve(),
        args.baseline_dir.resolve(),
        args.submission_dir.resolve(),
    )
    print(json.dumps(result, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
