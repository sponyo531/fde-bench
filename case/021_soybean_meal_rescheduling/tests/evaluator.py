"""
EXTRACTOR_SPEC:
  plan_file: solution.json
  required_fields: [shifts]
  notes: >
    选手需产出「豆粕43%/45%续排放号」方案 JSON，文件名 solution.json，放在提交目录根。
    这是一份从 26白15:00 起、覆盖 26夜/27白/27夜/28白/28夜/29白/29夜/30白/30夜 各班次、
    两个工厂（一厂/二厂）的续排计划。评估器只信任 shifts 里的**决策变量**，
    所有吨数/库存/得分由评估器从 data/ 的 xlsx 独立重算，忽略方案自报的任何汇总值。

    结构：
      {
        "shifts": [
          {
            "shift_name": "26夜",          // 班次名，取值 {26白,26夜,27白,27夜,28白,28夜,29白,29夜,30白,30夜}
            "factory": "一厂",             // {一厂, 二厂}
            "protein": "43%",             // {43%, 45%}（27白班次此字段可空/任意，blocked=true）
            "blocked": false,             // 27白两厂必须 true（整班封存），其余班次必须 false
            "hours": [                     // 长度=n_hours 的逐小时车数列表；每元素给 3 类包装车数
              {"xiaobao": 2, "tunbao": 1, "sanliao": 1},   // 小包/吨包/散料 车数
              ...
            ],
            "train_nodes": 3,             // 该班次火车节数（仅一厂对应蛋白班次可>0）
            "guluomei_tons": 200          // 该班次过瘤胃预留吨数（不出仓，仅记账）
          },
          ...
        ]
      }
    - hours 长度需等于该班次有效小时数（26白续排=5小时，其余非封存班次=12小时，27白=12但 blocked）。
    - 每小时车数 = xiaobao+tunbao+sanliao；均为非负整数。
    - 选手若自报了 total_tons / inventory / score 等汇总字段，评估器一律忽略。
    常见需规整的情况：
    - 车数写成宽表 CSV（每班一行 小包/吨包/散料 合计）-> 需按班次有效小时均摊回 hours，
      或直接以 per-shift 合计给出（评估器兼容：若某班给了 shift 级 "xiaobao_total" 等聚合车数，
      会当作单一"合并小时"处理；但推荐给 hours 明细）。
    - protein 写成 "43"/"0.43"/"43%蛋白" -> 归一到 "43%"；factory 写成 "厂一"/"F1" -> "一厂"。
    - blocked 写成 "封存"/"是"/1 -> true。
    请用 Bash + python3 处理，最终写出 solution.json 到 output 目录。
"""
import argparse
import json
import os
import re
import math
import traceback
from pathlib import Path

PLAN_FILE = "solution.json"
_HERE = os.path.dirname(os.path.abspath(__file__))
_DATA = os.path.join(_HERE, "..", "data")
XLSX_NAME = "soybean_meal_input.xlsx"

# ── 建模前提（用户澄清确认的班次结构，非选手可决策）────────────────────────────
# 续排范围：26白(15:00起,5h) 起，到 30夜。27白两厂整班封存。
# 蛋白分配（用户澄清：26白/26夜 一厂43%二厂45%；27夜切换 一厂45%二厂43%，此后每24h轮换）。
# 每个班次的 (factory, protein, n_hours, blocked)。
SHIFT_STRUCT = [
    ("26白", "一厂", "43%", 5,  False),   # 15:00-19:00 续排 5 小时
    ("26白", "二厂", "45%", 5,  False),
    ("26夜", "一厂", "43%", 12, False),
    ("26夜", "二厂", "45%", 12, False),
    ("27白", "一厂", "43%", 12, True),    # 整班封存
    ("27白", "二厂", "43%", 12, True),    # 整班封存
    ("27夜", "一厂", "45%", 12, False),   # 切换
    ("27夜", "二厂", "43%", 12, False),
    ("28白", "一厂", "43%", 12, False),
    ("28白", "二厂", "45%", 12, False),
    ("28夜", "一厂", "45%", 12, False),
    ("28夜", "二厂", "43%", 12, False),
    ("29白", "一厂", "43%", 12, False),
    ("29白", "二厂", "45%", 12, False),
    ("29夜", "一厂", "45%", 12, False),
    ("29夜", "二厂", "43%", 12, False),
    ("30白", "一厂", "43%", 12, False),
    ("30白", "二厂", "45%", 12, False),
    ("30夜", "一厂", "45%", 12, False),
    ("30夜", "二厂", "43%", 12, False),
]

# ── 核心参数（可信硬值；产能取自得率表绿色底色行，评估器从 xlsx 需求校验一致性）──
TON_PER_TRUCK = 33
TON_PER_TRAIN_NODE = 65
# 每班12小时产量（吨），来自得率表绿色行：一厂43%=1155,二厂43%=1540,一厂45%=1080,二厂45%=1440
SHIFT_PROD_12H = {("一厂", "43%"): 1155, ("二厂", "43%"): 1540,
                  ("一厂", "45%"): 1080, ("二厂", "45%"): 1440}
# 每小时合理最大车数（产能上限）：一厂 3 车/hr，二厂 4 车/hr
HOURLY_CAP = {"一厂": 3, "二厂": 4}
INV_INIT = {"43%": 500.0, "45%": 500.0}
INV_LIMIT = {"43%": 1500.0, "45%": 2000.0}
# 26号14:00 前历史已排（不可覆盖，计入总需求满足量）
HIST = {"43%": {"tons": 561.0, "xiaobao": 17, "tunbao": 0, "sanliao": 0},
        "45%": {"tons": 924.0, "xiaobao": 7,  "tunbao": 0, "sanliao": 21}}
# 过瘤胃预留（43% 方向，硬约束下限）
GULUOMEI_26BAI_MIN = 100.0   # 26白续排段 ≥100 吨
GULUOMEI_LATER_MIN = 200.0   # 26夜及后续每个 43% 班次 ≥200 吨

# 评分权重（用户需求：库存平滑 0.40 + 安全 0.35 + 均衡 0.25 加权最大）
W_SMOOTH, W_SAFETY, W_BALANCE = 0.40, 0.35, 0.25
MAX_INV_STD = 3500.0
SAFE_MIN_INV = 300.0
BALANCE_BASE = 2800.0
EPS = 1e-6


# ── 从 data/ xlsx 独立读入需求（防作弊：需求不硬编码，从原始数据读）────────────
def _parse_yield_sheet(wb):
    """从「得率」sheet 读每厂每蛋白的每班产量/车数/得率。

    表结构：蛋白列为合并单元格（块首行给值、其余留空），厂名标在第 7 列，
    「每班」列形如 '1155/34'（吨数/车数）。
    """
    ws = wb["得率"]
    cur = None
    out = {}
    for r in range(1, ws.max_row + 1):
        a = ws.cell(r, 1).value
        if isinstance(a, (int, float)) and float(a) in (0.43, 0.45):
            cur = "%d%%" % int(round(float(a) * 100))
        tag = ws.cell(r, 7).value
        if isinstance(tag, str) and tag.strip() in ("一厂", "二厂") and cur:
            m = re.match(r"\s*([\d.]+)\s*/\s*([\d.]+)", str(ws.cell(r, 5).value))
            if m:
                out[(tag.strip(), cur)] = {
                    "tons": float(m.group(1)),
                    "trucks": float(m.group(2)),
                    "yield": float(ws.cell(r, 3).value or 0),
                }
    return out


def _parse_unit_tons(wb):
    """从「排产模拟」备注里反解 吨/车 与 吨/节（形如 '货车29车/957吨、火车3节/195吨'）。"""
    ws = wb["排产模拟"]
    truck, node = set(), set()
    for r in range(1, ws.max_row + 1):
        for c in range(1, ws.max_column + 1):
            v = ws.cell(r, c).value
            if not isinstance(v, str):
                continue
            m = re.search(r"货车(\d+)车/(\d+)吨", v)
            if m and int(m.group(1)) > 0:
                truck.add(round(int(m.group(2)) / int(m.group(1))))
            m = re.search(r"火车(\d+)节/(\d+)吨", v)
            if m and int(m.group(1)) > 0:
                node.add(round(int(m.group(2)) / int(m.group(1))))
    return (truck.pop() if len(truck) == 1 else 33,
            node.pop() if len(node) == 1 else 65)


def _parse_guluomei(wb):
    """从「排产模拟」的 '过-100' / '过-200' 标注读过瘤胃下限档位。"""
    ws = wb["排产模拟"]
    vals = set()
    for r in range(1, ws.max_row + 1):
        for c in range(1, ws.max_column + 1):
            v = ws.cell(r, c).value
            if isinstance(v, str):
                m = re.fullmatch(r"过-(\d+)", v.strip())
                if m:
                    vals.add(float(m.group(1)))
    if not vals:
        return 100.0, 200.0
    return min(vals), max(vals)


def load_demand(data_dir):
    import openpyxl
    path = os.path.join(data_dir, XLSX_NAME)
    wb = openpyxl.load_workbook(path, data_only=True)
    ws = wb["模拟报量"]

    def col2(r):
        return ws.cell(r, 2).value

    def num(v):
        if v is None:
            return 0
        s = re.sub(r"[^\d.]", "", str(v))
        return float(s) if s else 0

    dem = {
        "43%": {"total": num(col2(2)), "cars": num(col2(3)),
                "xiaobao": num(col2(5)), "tunbao": num(col2(6)),
                "sanliao": num(col2(7)), "train": num(col2(8))},
        "45%": {"total": num(col2(11)), "cars": num(col2(12)),
                "xiaobao": num(col2(14)), "tunbao": num(col2(15)),
                "sanliao": num(col2(16)), "train": num(col2(17))},
    }
    # 续排剩余需求 = 总需求 - 历史已排
    rem = {}
    for pk in ("43%", "45%"):
        h = HIST[pk]
        rem[pk] = {
            "total": dem[pk]["total"] - h["tons"],
            "xiaobao": int(dem[pk]["xiaobao"] - h["xiaobao"]),
            "tunbao": int(dem[pk]["tunbao"] - h["tunbao"]),
            "sanliao": int(dem[pk]["sanliao"] - h["sanliao"]),
            "train": int(dem[pk]["train"]),
        }
    # ── 以下参数一律从 xlsx 解析，不写死（换数据即自动跟随）──
    global SHIFT_PROD_12H, HOURLY_CAP, TON_PER_TRUCK, TON_PER_TRAIN_NODE
    global GULUOMEI_26BAI_MIN, GULUOMEI_LATER_MIN

    ytab = _parse_yield_sheet(wb)
    if ytab:
        SHIFT_PROD_12H = {k: v["tons"] for k, v in ytab.items()}
        cap = {}
        for (fac, _prot), v in ytab.items():
            hourly = int(math.ceil(v["trucks"] / 12.0))
            cap[fac] = max(cap.get(fac, 0), hourly)
        if cap:
            HOURLY_CAP = cap

    tpt, tpn = _parse_unit_tons(wb)
    TON_PER_TRUCK, TON_PER_TRAIN_NODE = tpt, tpn

    g_lo, g_hi = _parse_guluomei(wb)
    GULUOMEI_26BAI_MIN, GULUOMEI_LATER_MIN = g_lo, g_hi

    return dem, rem


# ── 归一化 ───────────────────────────────────────────────────────────────────
def norm_protein(x):
    s = str(x).strip()
    if "43" in s:
        return "43%"
    if "45" in s:
        return "45%"
    return s


def norm_factory(x):
    s = str(x).strip()
    if "一" in s or s in ("1", "F1", "厂一"):
        return "一厂"
    if "二" in s or s in ("2", "F2", "厂二"):
        return "二厂"
    return s


def as_bool(x):
    if isinstance(x, bool):
        return x
    s = str(x).strip().lower()
    return s in ("true", "1", "是", "封存", "yes", "y")


def load_solution(plan_path):
    with open(plan_path, encoding="utf-8") as f:
        obj = json.load(f)
    if isinstance(obj, dict) and "shifts" in obj:
        return list(obj["shifts"])
    if isinstance(obj, list):
        return list(obj)
    raise ValueError("solution.json 缺少 shifts 字段")


def shift_hours(sh):
    """把选手 shift 的 hours 归一为逐小时 (xiaobao,tunbao,sanliao) 列表。
    兼容：给了 hours 明细 -> 直接用；只给了 shift 级聚合车数 -> 折成单一"合并小时"。"""
    hours = sh.get("hours")
    out = []
    if isinstance(hours, list) and hours:
        for h in hours:
            out.append((
                int(h.get("xiaobao", 0) or 0),
                int(h.get("tunbao", 0) or 0),
                int(h.get("sanliao", 0) or 0),
            ))
        return out
    # 聚合兜底
    xb = int(sh.get("xiaobao_total", sh.get("xiaobao", 0)) or 0)
    tb = int(sh.get("tunbao_total", sh.get("tunbao", 0)) or 0)
    sl = int(sh.get("sanliao_total", sh.get("sanliao", 0)) or 0)
    return [(xb, tb, sl)]


def evaluate(plan_path, data_dir):
    metrics = {"overall_score": 0.0, "validity_score": 0.0, "quality_score": 0.0}
    try:
        dem, rem = load_demand(data_dir)

        if not os.path.isfile(plan_path):
            metrics["error_info"] = {"constraint": ["缺少 solution.json"]}
            return metrics

        shifts = load_solution(plan_path)
        violations = []

        # ── 硬约束0：班次结构必须完整覆盖固定的 20 条 (shift,factory) 且唯一 ──
        # 建模前提由结构给定；选手若缺项/多报/篡改结构 -> 判死。
        provided = {}
        for sh in shifts:
            key = (str(sh.get("shift_name", "")).strip(),
                   norm_factory(sh.get("factory", "")))
            if key in provided:
                violations.append(f"班次重复: {key}")
            provided[key] = sh

        required_keys = [(s[0], s[1]) for s in SHIFT_STRUCT]
        missing = [k for k in required_keys if k not in provided]
        if missing:
            violations.append(f"缺失班次: {missing[:6]}")
        unknown = [k for k in provided if k not in set(required_keys)]
        if unknown:
            violations.append(f"未知/多余班次: {unknown[:6]}")

        if violations:
            metrics["error_info"] = {"constraint": violations[:8]}
            return metrics

        # ── 逐班校验决策变量 schema + 建模前提字段 ──
        # 累计器
        acc = {"43%": {"xiaobao": 0, "tunbao": 0, "sanliao": 0, "train": 0, "tons": 0.0},
               "45%": {"xiaobao": 0, "tunbao": 0, "sanliao": 0, "train": 0, "tons": 0.0}}
        # 按 SHIFT_STRUCT 顺序遍历以保证库存轧账时间轴正确
        inv = {"43%": INV_INIT["43%"], "45%": INV_INIT["45%"]}
        min_inv = {"43%": inv["43%"], "45%": inv["45%"]}
        combined_trace = []
        max_shift_disp = 0.0
        train43 = 0
        train45 = 0

        for (sname, fac, prot, n_hours, blocked_req) in SHIFT_STRUCT:
            sh = provided[(sname, fac)]
            blk = as_bool(sh.get("blocked", False))

            # 建模前提：blocked 标志必须与结构一致（27白必封存，其余必不封存）——防作弊
            if blk != blocked_req:
                violations.append(f"{sname}{fac} blocked={blk} 应为 {blocked_req}")
                continue

            if blocked_req:
                # 27白整班封存：不得排任何车数/火车/过瘤胃，库存不变
                hrs = shift_hours(sh)
                if any(sum(h) > 0 for h in hrs) or int(sh.get("train_nodes", 0) or 0) > 0:
                    violations.append(f"{sname}{fac} 整班封存却排了货/火车（应留空）")
                combined_trace.append(inv["43%"] + inv["45%"])
                min_inv["43%"] = min(min_inv["43%"], inv["43%"])
                min_inv["45%"] = min(min_inv["45%"], inv["45%"])
                continue

            hrs = shift_hours(sh)
            # schema：hours 长度须等于该班有效小时数
            if len(hrs) != n_hours:
                violations.append(f"{sname}{fac} hours 长度={len(hrs)} 应为 {n_hours}")
                # 仍继续用给定 hrs 轧账，避免误放行
            # 每小时车数不超产能
            cap = HOURLY_CAP[fac]
            for i, (xb, tb, sl) in enumerate(hrs):
                if xb < 0 or tb < 0 or sl < 0:
                    violations.append(f"{sname}{fac} hr{i} 出现负车数")
                if xb + tb + sl > cap:
                    violations.append(f"{sname}{fac} hr{i}: {xb+tb+sl}车 > 产能上限{cap}车")

            trucks = sum(xb + tb + sl for (xb, tb, sl) in hrs)
            xb_t = sum(x for (x, _, _) in hrs)
            tb_t = sum(t for (_, t, _) in hrs)
            sl_t = sum(s for (_, _, s) in hrs)
            tn = int(sh.get("train_nodes", 0) or 0)
            if tn < 0:
                violations.append(f"{sname}{fac} 火车节数为负")
                tn = 0

            # 火车约束：只能排在 一厂 且蛋白与班次一致
            if tn > 0 and fac != "一厂":
                violations.append(f"{sname}{fac} 火车排在了非一厂({fac})")
            if prot == "43%":
                train43 += tn
            else:
                train45 += tn

            # 过瘤胃预留（仅 43% 方向有下限约束）
            gl = float(sh.get("guluomei_tons", 0) or 0)
            if gl < 0:
                violations.append(f"{sname}{fac} 过瘤胃为负")
                gl = 0.0
            if prot == "43%":
                need = GULUOMEI_26BAI_MIN if sname == "26白" else GULUOMEI_LATER_MIN
                if gl + EPS < need:
                    violations.append(f"{sname}{fac} 过瘤胃预留 {gl:.0f} < {need:.0f}")
            # 过瘤胃预留吨数须真实存在于库存里（不能超过当班可用库存后仍出仓）——见库存轧账

            # 产量、出仓、库存轧账（独立重算）
            prod = SHIFT_PROD_12H[(fac, prot)] / 12.0 * n_hours
            train_tons = tn * TON_PER_TRAIN_NODE
            dispatch = trucks * TON_PER_TRUCK + train_tons   # 过瘤胃不出仓
            # 库存守恒：不能凭空放号（放号+预留不得超过 期初+产量）
            available = inv[prot] + prod
            if dispatch + gl > available + 1.0:
                violations.append(
                    f"{sname}{fac} 放号+预留 {dispatch+gl:.0f} > 期初+产量 {available:.0f}（凭空放号/库存为负）")
            inv[prot] = inv[prot] + prod - dispatch
            if inv[prot] < -1.0:
                violations.append(f"{sname}{fac} {prot}仓末库存为负: {inv[prot]:.0f}")
            if inv[prot] > INV_LIMIT[prot] + 0.5:
                violations.append(f"{sname} {prot}仓超上限{INV_LIMIT[prot]:.0f}: {inv[prot]:.0f}")

            min_inv["43%"] = min(min_inv["43%"], inv["43%"])
            min_inv["45%"] = min(min_inv["45%"], inv["45%"])
            combined_trace.append(inv["43%"] + inv["45%"])
            max_shift_disp = max(max_shift_disp, dispatch)

            acc[prot]["xiaobao"] += xb_t
            acc[prot]["tunbao"] += tb_t
            acc[prot]["sanliao"] += sl_t
            acc[prot]["train"] += tn
            acc[prot]["tons"] += dispatch

        # ── 火车节数须精确匹配需求（防少排/多排）──
        if train43 != rem["43%"]["train"]:
            violations.append(f"43%火车节数 {train43} ≠ 需求 {rem['43%']['train']}")
        if train45 != rem["45%"]["train"]:
            violations.append(f"45%火车节数 {train45} ≠ 需求 {rem['45%']['train']}")

        # ── 需求覆盖（分包装 + 总吨；续排口径=总需求-历史）──
        for pk in ("43%", "45%"):
            r = rem[pk]
            g = acc[pk]
            if g["tons"] + 0.5 < r["total"]:
                violations.append(f"{pk} 出仓总量不足: {g['tons']:.0f} < {r['total']:.0f}")
            if g["xiaobao"] < r["xiaobao"]:
                violations.append(f"{pk} 小包不足: {g['xiaobao']} < {r['xiaobao']}")
            if g["tunbao"] < r["tunbao"]:
                violations.append(f"{pk} 吨包不足: {g['tunbao']} < {r['tunbao']}")
            if g["sanliao"] < r["sanliao"]:
                violations.append(f"{pk} 散料不足: {g['sanliao']} < {r['sanliao']}")

        if violations:
            metrics["error_info"] = {"constraint": violations[:8]}
            return metrics

        metrics["validity_score"] = 1.0

        # ── quality：加权综合分（maximize）──
        n = len(combined_trace)
        mean_c = sum(combined_trace) / n
        std_c = (sum((x - mean_c) ** 2 for x in combined_trace) / n) ** 0.5
        smoothness = max(0.0, 1.0 - std_c / MAX_INV_STD)
        safety = (min(min_inv["43%"] / SAFE_MIN_INV, 1.0) * 0.5
                  + min(min_inv["45%"] / SAFE_MIN_INV, 1.0) * 0.5)
        balance = max(0.0, 1.0 - max_shift_disp / BALANCE_BASE)
        player_score = 100.0 * (W_SMOOTH * smoothness + W_SAFETY * safety + W_BALANCE * balance)

        ref = load_reference()
        quality = player_score / ref if ref > 0 else 0.0
        metrics["quality_score"] = round(quality, 6)
        metrics["overall_score"] = round(quality, 6)
        metrics["player_score"] = round(player_score, 4)
        metrics["reference_value"] = ref
        metrics["sub_scores"] = {
            "smoothness": round(smoothness, 4),
            "safety": round(safety, 4),
            "balance": round(balance, 4),
            "inv_std": round(std_c, 1),
            "min_inv43": round(min_inv["43%"], 1),
            "min_inv45": round(min_inv["45%"], 1),
            "max_shift_tons": round(max_shift_disp, 1),
        }
    except Exception as e:
        metrics["error_info"] = {"exception": str(e), "traceback": traceback.format_exc()}
    return metrics


def load_reference():
    ref_path = os.path.join(_HERE, "baseline", "reference_metrics.json")
    if os.path.isfile(ref_path):
        try:
            return float(json.load(open(ref_path))["reference_value"])
        except Exception:
            pass
    return 86.52  # 回退锚点：源 session 演化最优分


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
