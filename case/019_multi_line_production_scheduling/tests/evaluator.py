"""
EXTRACTOR_SPEC:
  plan_file: solution.json
  required_columns: [fg_schedule, semi_schedule]
  notes: >
    The agent must produce a JSON file describing the full production schedule.
    Expected format:
      {
        "fg_schedule": [
          {"date": "2025-08-01", "line": "1#", "model": "101A", "qty": 0},
          ...   // 每条成品产线每天每机型的产量（qty 为整数，可为 0）
        ],
        "semi_schedule": [
          {"date": "2025-07-31", "semi": "AA", "self_qty": 500, "buy_qty": 0},
          ...   // 每条半成品产线每天的自制量 self_qty 与 BB 外购量 buy_qty；从 7/31 开始
        ]
      }

    字段说明：
    - fg_schedule[].date: ISO 日期字符串 'YYYY-MM-DD'，范围 2025-08-01 ~ 2025-08-31。
    - fg_schedule[].line: 成品产线，取值 '1#' / '2#' / '3#'。
    - fg_schedule[].model: 机型，取值 101A..110A。
    - fg_schedule[].qty: 当天该(产线,机型)的成品产量（整数）。同一(date,line,model)多行会被累加。
    - semi_schedule[].date: ISO 日期，含 '2025-07-31'（提前期首日）到 '2025-08-31'。
    - semi_schedule[].semi: 半成品类型 'AA' / 'BB' / 'CC'。
    - semi_schedule[].self_qty: 当天该半成品自制量（整数）。
    - semi_schedule[].buy_qty: 当天 BB 外购量（仅 BB 行有意义，整数）。

    常见 agent 产出规整方式：
    - 若产出为 CSV（fg_schedule.csv / semi_schedule.csv），列名对齐后转成上述两个数组。
    - 日期若为 '2025/08/01' 或 Excel 序列号，统一转成 'YYYY-MM-DD'。
    - line 若为 'L1'/'line1' 等，映射为 '1#'/'2#'/'3#'。
    - 缺失的 (date,line,model) 组合可不写（评估器按 0 处理）；负数产量应截为 0。
    - penalties/final_inv/delay_events 等诊断字段可选，评估器不读、会独立重算。

    用 Bash 跑 python3 转换后，把 solution.json 写入输出目录。
"""
import argparse
import json
import os
import re
import datetime
import math
import traceback
from pathlib import Path
from collections import defaultdict

PLAN_FILE = "solution.json"
_HERE = os.path.dirname(os.path.abspath(__file__))
PERFECT_Q = 1e6
_DATA = os.path.join(_HERE, "..", "data")
_XLSX_NAME = "production_input.xlsx"

# ─── 非业务数据常量（计划逻辑固有，非从 data 读的业务实体） ───
FG_LINES = ['1#', '2#', '3#']

# ─── 以下业务数据全部从 data/ 的 Excel 各 sheet 解析后填充（见 load_data） ───
# 计划日历
START = END = None
DAYS = []
# 成品/机型
MODELS = []
MODEL_LINES = {}          # 机型 → 可生产产线列表        ← 产能&产线
MODEL_UPH = {}            # 机型 → UPH                    ← 产能&产线
FG_INIT = {}              # 机型 → 7/31 期初库存          ← 库存
DEMAND = {}               # 机型 → 31 天每日需求          ← 客户需求
# 产线/工时
CHANGEOVER_MIN = {}       # 产线 → 换产分钟               ← 产能&产线（合并规则文字）
DAILY_HOURS = None        # 每日可用工时                  ← 产能&产线（文字）
FG_STOP_DAY = None        # 成品线停工日                  ← 产能&产线（文字）
SEMI_STOP_DAY = None      # 半成品线停工日                ← 产能&产线（文字）
# 半成品
SEMI_UPH = {}             # 半成品 → UPH                  ← 产能&产线
SEMI_MODELS = {}          # 半成品 → 消耗它的成品机型     ← BOM（反推）
SEMI_INIT_730 = {}        # 半成品 → 7/30 期初库存        ← 库存
SEMI_SELFMAX = {}         # 半成品 → 自制上限硬帽(层2零件) ← BOM
# BOM 零件联动上限
AA_GROUP_CAP = None       # 101A+102A+103A 当日上限 (零件C/D min) ← BOM
CC_GROUP_CAP = None       # 109A+110A 当日上限 (零件K/L min)      ← BOM
GH_GROUP_CAP = None       # 107A+108A(旧BOM) 当日上限 (零件G/H)   ← BOM
MN_DATE_LIMITS = {}       # 108A 新BOM 零件M/N 分日期段上限       ← BOM
# 罚款单价 / 阈值                                        ← 评分规则
BB_BUY_MAX = None
BB_BUY_COST = None
DELAY_COST = None
CHANGEOVER_COST = None
LINE2_COST = None
EXCESS_INV_COST = None
EXCESS_INV_THRESHOLD = None


def _excel_serial_to_date(n):
    """Excel 序列号（1900 日期系统）→ datetime.date。"""
    return datetime.date(1899, 12, 30) + datetime.timedelta(days=int(round(float(n))))


def load_data(data_dir):
    """从 data/ 的 Excel 各 sheet 解析全部业务数据，填充模块级全局。

    Sheet 布局要点（格式不规整）：
      客户需求  : 表头在第2行（第1行全 nan），日期列是 Excel 序列号；末尾有'总计'/备注行。
      库存      : 左块 成品产品型号/7/31库存，右块 半成品型号/7/30库存，前几行 nan。
      产能&产线 : 左块 成品UPH+可生产产线，右块 半成品UPH；换产时间/工时/停工日以文字给出。
      BOM       : 3 个纵向列块，每块堆叠多个机型的两层 BOM；108A 分 ECN前/后两块。
      评分规则  : 事项/损失金额 5 行。
    """
    import pandas as pd
    xlsx = os.path.join(data_dir, _XLSX_NAME)

    # ---------- 客户需求 ----------
    df = pd.read_excel(xlsx, sheet_name='客户需求', header=None)
    hdr = next(i for i in range(len(df)) if str(df.iat[i, 1]).strip() == '成品产品型号')
    date_cols, dates = [], []
    for c in range(2, df.shape[1]):
        v = df.iat[hdr, c]
        if pd.notna(v):
            date_cols.append(c)
            dates.append(_excel_serial_to_date(v))
    models, demand = [], {}
    for i in range(hdr + 1, len(df)):
        name = df.iat[i, 1]
        if pd.isna(name):
            continue
        name = str(name).strip()
        if not re.match(r'^\d{3}A$', name):      # 跳过 '总计' 及备注行
            continue
        models.append(name)
        demand[name] = [int(df.iat[i, c]) if pd.notna(df.iat[i, c]) else 0 for c in date_cols]

    # ---------- 库存 ----------
    df = pd.read_excel(xlsx, sheet_name='库存', header=None)
    hr = hc = sr = sc = None
    for i in range(len(df)):
        for c in range(df.shape[1]):
            t = str(df.iat[i, c]).strip()
            if t == '成品产品型号':
                hr, hc = i, c
            elif t == '半成品型号':
                sr, sc = i, c
    fg_init, semi_init = {}, {}
    for i in range(hr + 1, len(df)):
        m = df.iat[i, hc]
        if pd.notna(m) and re.match(r'^\d{3}A$', str(m).strip()):
            fg_init[str(m).strip()] = int(df.iat[i, hc + 1])
    for i in range(sr + 1, len(df)):
        m = df.iat[i, sc]
        if pd.notna(m) and str(m).strip() in ('AA', 'BB', 'CC'):
            semi_init[str(m).strip()] = int(df.iat[i, sc + 1])

    # ---------- 产能&产线 ----------
    df = pd.read_excel(xlsx, sheet_name='产能&产线', header=None)
    h = fc = scol = None
    for i in range(len(df)):
        row = [str(x).strip() for x in df.iloc[i]]
        if '成品产品型号' in row:
            h = i
            fc = row.index('成品产品型号')
            scol = row.index('半成品型号') if '半成品型号' in row else None
            break
    model_uph, model_lines, semi_uph = {}, {}, {}
    for i in range(h + 1, len(df)):
        m = df.iat[i, fc]
        if pd.notna(m) and re.match(r'^\d{3}A$', str(m).strip()):
            mm = str(m).strip()
            model_uph[mm] = int(df.iat[i, fc + 1])
            model_lines[mm] = ['%s#' % p for p in re.findall(r'(\d)#', str(df.iat[i, fc + 2]))]
        if scol is not None:
            sm = df.iat[i, scol]
            if pd.notna(sm) and str(sm).strip() in ('AA', 'BB', 'CC'):
                semi_uph[str(sm).strip()] = int(df.iat[i, scol + 1])
    alltext = '\n'.join(str(x) for x in df.values.flatten() if pd.notna(x))
    changeover = {'%s#' % a: int(b) for a, b in
                  re.findall(r'(\d)#成品产线所有机型之间每次转换需要(\d+)min', alltext)}
    daily_hours = float(re.search(r'每天开工时间([\d.]+)h', alltext).group(1))
    mfg = re.search(r'成品产线[^除]*除(\d+)/(\d+)外', alltext)
    msemi = re.search(r'半品产线[^除]*除(\d+)/(\d+)外', alltext)
    fg_stop = datetime.date(2025, int(mfg.group(1)), int(mfg.group(2)))
    semi_stop = datetime.date(2025, int(msemi.group(1)), int(msemi.group(2)))

    # ---------- BOM ----------
    df = pd.read_excel(xlsx, sheet_name='BOM', header=None)
    headers = [(i, c) for i in range(len(df)) for c in range(df.shape[1])
               if str(df.iat[i, c]).strip() == '型号']
    semi_of_model, semi_l2, l1_old, l1_new, mn, bb_buy_max = {}, {}, {}, {}, {}, None
    for (hi, hc2) in headers:
        model = str(df.iat[hi + 1, hc2]).strip()
        ecn = None
        for lc in range(hc2):
            v = df.iat[hi, lc]
            if pd.notna(v) and 'ECN' in str(v):
                ecn = str(v).strip()
        is_new = ecn is not None and 'ECN后' in ecn
        r = hi + 2
        while r < len(df):
            name = df.iat[r, hc2]
            if pd.isna(name) or str(name).strip() == '型号':
                break
            name = str(name).strip()
            lvl = int(df.iat[r, hc2 + 1]) if pd.notna(df.iat[r, hc2 + 1]) else None
            mm = str(df.iat[r, hc2 + 4]).strip()
            if name.startswith('半成品'):
                semi_of_model.setdefault(model, name.replace('半成品', ''))
                mb = re.search(r'外购每天(\d+)', mm)
                if mb:
                    bb_buy_max = int(mb.group(1))
            elif name.startswith('零件'):
                part = name.replace('零件', '')
                if lvl == 1:
                    if re.search(r'\d+/\d+-\d+/\d+:\d+', mm):        # M/N 分日期段
                        lst = []
                        for line in mm.split('\n'):
                            mt = re.match(r'(\d+)/(\d+)-(\d+)/(\d+):(\d+)', line.strip())
                            if mt:
                                s = datetime.date(2025, int(mt.group(1)), int(mt.group(2)))
                                e = datetime.date(2025, int(mt.group(3)), int(mt.group(4)))
                                lst.append(((s, e), int(mt.group(5))))
                        mn[part] = lst
                        (l1_new if is_new else l1_old).setdefault(model, {})[part] = None
                    else:
                        (l1_new if is_new else l1_old).setdefault(model, {})[part] = int(float(mm))
                elif lvl == 2:
                    semi_l2.setdefault(semi_of_model.get(model), {})[part] = int(float(mm))
            r += 1
    semi_models = {}
    for m, s in semi_of_model.items():
        semi_models.setdefault(s, []).append(m)
    for s in semi_models:
        semi_models[s] = sorted(semi_models[s])
    semi_selfmax = {s: min(v.values()) for s, v in semi_l2.items()}      # AA:min(A=500,B=700)=500
    # 层1零件联动上限（同组机型当日总产量帽 = min 该组零件上限）
    aa_group_cap = min(l1_old['101A'].values())   # 零件C/D → 650
    cc_group_cap = min(l1_old['109A'].values())   # 零件K/L → 2000
    gh_group_cap = min(l1_old['107A'].values())   # 零件G/H → 2000（107A+108A旧BOM 共享）

    # ---------- 评分规则 ----------
    df = pd.read_excel(xlsx, sheet_name='评分规则', header=None)
    costs = {}
    for i in range(len(df)):
        ev, amt = df.iat[i, 2], df.iat[i, 3]
        if pd.notna(ev) and pd.notna(amt) and str(amt).strip().replace('.', '').isdigit():
            costs[str(ev).strip()] = int(float(amt))
    def _cost(pred):
        return next(v for k, v in costs.items() if pred(k))
    delay_cost = _cost(lambda k: '客户需求日期' in k or '超过客户需求' in k)
    changeover_cost = _cost(lambda k: '转产' in k)
    line2_cost = _cost(lambda k: '2#' in k)
    excess_cost = _cost(lambda k: '库存超出' in k or '超出200' in k)
    bb_buy_cost = _cost(lambda k: 'BB外购' in k or '半成品BB' in k)
    excess_threshold = next(int(re.search(r'超出(\d+)台', k).group(1))
                            for k in costs if re.search(r'超出(\d+)台', k))

    # ---------- 写入模块级全局 ----------
    g = globals()
    g['DAYS'] = dates
    g['START'], g['END'] = dates[0], dates[-1]
    g['MODELS'] = models
    g['DEMAND'] = demand
    g['MODEL_LINES'] = model_lines
    g['MODEL_UPH'] = model_uph
    g['FG_INIT'] = fg_init
    g['CHANGEOVER_MIN'] = changeover
    g['DAILY_HOURS'] = daily_hours
    g['FG_STOP_DAY'] = fg_stop
    g['SEMI_STOP_DAY'] = semi_stop
    g['SEMI_UPH'] = semi_uph
    g['SEMI_MODELS'] = semi_models
    g['SEMI_INIT_730'] = semi_init
    g['SEMI_SELFMAX'] = semi_selfmax
    g['AA_GROUP_CAP'] = aa_group_cap
    g['CC_GROUP_CAP'] = cc_group_cap
    g['GH_GROUP_CAP'] = gh_group_cap
    g['MN_DATE_LIMITS'] = mn
    g['BB_BUY_MAX'] = bb_buy_max
    g['BB_BUY_COST'] = bb_buy_cost
    g['DELAY_COST'] = delay_cost
    g['CHANGEOVER_COST'] = changeover_cost
    g['LINE2_COST'] = line2_cost
    g['EXCESS_INV_COST'] = excess_cost
    g['EXCESS_INV_THRESHOLD'] = excess_threshold


def d_str(d):
    return d.strftime('%Y-%m-%d')


def is_fg_open(d):
    return d != FG_STOP_DAY


def is_semi_open(d):
    return d != SEMI_STOP_DAY


def _ecn_switch_day():
    """108A 新 BOM 生效首日 = M/N 分段上限里最早的起始日（BOM 解析所得）。"""
    starts = [s for lst in MN_DATE_LIMITS.values() for (s, _e), _l in lst]
    return min(starts)


def is_new_bom(d):
    return d >= _ecn_switch_day()


def mn_limit(d):
    """108A 新 BOM 零件 M/N 当日上限（取 M/N 各段的最小值，含义等价，二者相同）。"""
    lim = None
    for part_ranges in MN_DATE_LIMITS.values():
        for (s, e), l in part_ranges:
            if s <= d <= e:
                lim = l if lim is None else min(lim, l)
    return lim if lim is not None else 10 ** 9


def _norm_date(s):
    """把常见日期格式规整为 ISO 'YYYY-MM-DD'；失败则原样返回。"""
    s = str(s).strip()
    for fmt in ('%Y-%m-%d', '%Y/%m/%d', '%Y.%m.%d'):
        try:
            return datetime.datetime.strptime(s, fmt).date().strftime('%Y-%m-%d')
        except ValueError:
            continue
    return s


def check_and_score(result):
    """校验 9 类硬约束并独立重算 5 类罚款。返回 (violations, metrics_dict)。"""
    violations = []

    # ── 构建查找表 ──
    fg = {}
    for row in result.get('fg_schedule', []):
        if not isinstance(row, dict):
            violations.append("成品排产存在非对象条目")
            continue
        date_s = _norm_date(row.get('date', ''))
        line = str(row.get('line', '')).strip()
        model = str(row.get('model', '')).strip()
        try:
            raw_qty = float(row.get('qty', 0))
        except (TypeError, ValueError):
            violations.append(f"成品产量不是数值: {row!r}")
            continue
        if (not math.isfinite(raw_qty) or raw_qty < 0
                or abs(raw_qty - round(raw_qty)) > 1e-9):
            violations.append(f"成品产量必须是非负整数: {row!r}")
            continue
        if date_s not in {d_str(d) for d in DAYS} or line not in FG_LINES or model not in MODELS:
            violations.append(f"成品排产的日期/产线/机型不在计划范围: {row!r}")
            continue
        key = (date_s, line, model)
        fg[key] = fg.get(key, 0) + int(round(raw_qty))

    semi = {}
    for row in result.get('semi_schedule', []):
        if not isinstance(row, dict):
            violations.append("半成品排产存在非对象条目")
            continue
        date_s = _norm_date(row.get('date', ''))
        semi_type = str(row.get('semi', '')).strip()
        try:
            raw_sq = float(row.get('self_qty', 0))
            raw_bq = float(row.get('buy_qty', 0))
        except (TypeError, ValueError):
            violations.append(f"半成品产量不是数值: {row!r}")
            continue
        if any(not math.isfinite(v) or v < 0 or abs(v - round(v)) > 1e-9
               for v in (raw_sq, raw_bq)):
            violations.append(f"半成品自制量和外购量必须是非负整数: {row!r}")
            continue
        allowed_semi_dates = {d_str(DAYS[0] - datetime.timedelta(days=1))} | {d_str(d) for d in DAYS}
        if date_s not in allowed_semi_dates or semi_type not in SEMI_MODELS:
            violations.append(f"半成品排产的日期/类型不在计划范围: {row!r}")
            continue
        key = (date_s, semi_type)
        sq = int(round(raw_sq))
        bq = int(round(raw_bq))
        if key in semi:
            semi[key] = (semi[key][0] + sq, semi[key][1] + bq)
        else:
            semi[key] = (sq, bq)

    # ── H1. 产线-机型匹配 ──
    for (date_s, line, model), qty in fg.items():
        if qty > 0 and line not in MODEL_LINES.get(model, []):
            violations.append(f"产线匹配违规: {date_s} {model} 不可在 {line} 生产")

    # ── H2. 成品产线停工日 ──
    for (date_s, line, model), qty in fg.items():
        try:
            d = datetime.date.fromisoformat(date_s)
        except ValueError:
            continue
        if d == FG_STOP_DAY and qty > 0:
            violations.append(f"成品停工日违规: {date_s} {line} {model} qty={qty}")

    # ── H3. 半成品产线停工日 ──
    for (date_s, semi_type), (self_qty, _) in semi.items():
        try:
            d = datetime.date.fromisoformat(date_s)
        except ValueError:
            continue
        if d == SEMI_STOP_DAY and self_qty > 0:
            violations.append(f"半成品停工日违规: {date_s} {semi_type} self_qty={self_qty}")

    # ── H4. 产线产能（工时约束，含换产扣时） ──
    line_day_models = defaultdict(list)
    for (date_s, line, model), qty in fg.items():
        if qty > 0:
            line_day_models[(date_s, line)].append((model, qty))
    for (date_s, line), pairs in line_day_models.items():
        try:
            d = datetime.date.fromisoformat(date_s)
        except ValueError:
            continue
        if not is_fg_open(d):
            continue
        if line not in CHANGEOVER_MIN:
            continue
        n_models = len(pairs)
        co_hours = (n_models - 1) * CHANGEOVER_MIN[line] / 60.0 if n_models > 1 else 0
        prod_hours = sum(qty / MODEL_UPH[m] for m, qty in pairs if m in MODEL_UPH)
        total_hours = prod_hours + co_hours
        if total_hours > DAILY_HOURS + 1e-6:
            violations.append(f"产线产能违规: {date_s} {line} 需{total_hours:.2f}h > {DAILY_HOURS}h")

    # ── H5. 零件C/D: AA 组机型当日总产量 ≤ AA_GROUP_CAP ──
    aa_models = SEMI_MODELS['AA']
    date_aa_prod = defaultdict(int)
    for (date_s, line, model), qty in fg.items():
        if model in aa_models:
            date_aa_prod[date_s] += qty
    for date_s, total in date_aa_prod.items():
        if total > AA_GROUP_CAP:
            violations.append(f"零件C/D违规: {date_s} AA组总产量{total} > {AA_GROUP_CAP}")

    # ── H6. 零件G/H: BB 组机型当日总产量 ≤ GH_GROUP_CAP/天 ──
    # BOM: 104A/105A/106A/107A/108A 均挂零件 G/H；108A 转新 BOM 后改耗 M/N，不再占 G/H。
    gh_models = SEMI_MODELS['BB']
    for date_s in set(d for d, l, m in fg):
        try:
            d = datetime.date.fromisoformat(date_s)
        except ValueError:
            continue
        total_gh = 0
        for m in gh_models:
            if m == '108A' and is_new_bom(d):
                continue
            total_gh += sum(fg.get((date_s, l, m), 0) for l in FG_LINES)
        if total_gh > GH_GROUP_CAP:
            violations.append(f"零件G/H违规: {date_s} BB组总产量{total_gh} > {GH_GROUP_CAP}")

    # ── H7. 零件K/L: CC 组机型(109A+110A)当日总产量 ≤ CC_GROUP_CAP/天 ──
    cc_models = SEMI_MODELS['CC']
    date_cc_prod = defaultdict(int)
    for (date_s, line, model), qty in fg.items():
        if model in cc_models:
            date_cc_prod[date_s] += qty
    for date_s, total in date_cc_prod.items():
        if total > CC_GROUP_CAP:
            violations.append(f"零件K/L违规: {date_s} CC组总产量{total} > {CC_GROUP_CAP}")

    # ── H8. 零件M/N: 108A新BOM ──
    for date_s in set(d for d, l, m in fg):
        try:
            d = datetime.date.fromisoformat(date_s)
        except ValueError:
            continue
        if is_new_bom(d):
            qty_218 = sum(fg.get((date_s, l, '108A'), 0) for l in FG_LINES)
            lim = mn_limit(d)
            if qty_218 > lim:
                violations.append(f"零件M/N违规: {date_s} 108A={qty_218} > {lim}")

    # ── H9. 半成品自制产能上限（工时×UPH，AA 另受层2零件硬帽） ──
    for (date_s, semi_type), (self_qty, _) in semi.items():
        if semi_type not in SEMI_UPH:
            continue
        cap = int(SEMI_UPH[semi_type] * DAILY_HOURS)
        if semi_type in SEMI_SELFMAX:
            cap = min(cap, SEMI_SELFMAX[semi_type])
        if self_qty > cap:
            violations.append(f"{semi_type}产能违规: {date_s} self_qty={self_qty} > {cap}")

    # ── H10. 外购校验：只有 BB 有外购渠道，且 ≤ 30/天 ──
    # 2025-08-23 修:原实现只校验 semi_type == 'BB' 的 buy_qty,而 H11 逐日轧账对**所有**
    # 半成品都执行 semi_inv[s] += self_q + buy_q。结果 AA / CC 可以随手申报任意 buy_qty
    # 白拿库存,而 H9 只管 self_qty、罚款也只累加 BB —— 实测 {"semi":"AA","buy_qty":999999}
    # 配同样的 CC 行能拿到 validity=1.0、零 violation、零成本,本题的核心瓶颈(AA 与 CC 的
    # 自制产能)被完全绕过。EXTRACTOR_SPEC 本来就写明 buy_qty「仅 BB 行有意义」。
    for (date_s, semi_type), (_, buy_qty) in semi.items():
        if semi_type == 'BB':
            if buy_qty > BB_BUY_MAX:
                violations.append(f"BB外购违规: {date_s} buy_qty={buy_qty} > {BB_BUY_MAX}")
        elif buy_qty != 0:
            violations.append(
                f"外购渠道违规: {date_s} {semi_type} buy_qty={buy_qty}，"
                f"只有 BB 可以外购，其余半成品必须为 0")

    # ── H11. 半成品提前期 + 库存充足（逐日轧账） ──
    # 半成品提前 1 天生产：计划首日前一天(DAYS[0]-1)的产出计入期初可用库存。
    semi_inv = dict(SEMI_INIT_730)
    semi_types = list(SEMI_MODELS.keys())
    d_lead_s = d_str(DAYS[0] - datetime.timedelta(days=1))
    for s in semi_types:
        self_q, buy_q = semi.get((d_lead_s, s), (0, 0))
        semi_inv[s] += self_q + buy_q
    for di, d in enumerate(DAYS):
        date_s = d_str(d)
        for s, ms in SEMI_MODELS.items():
            consumed = sum(sum(fg.get((date_s, l, m), 0) for l in FG_LINES) for m in ms)
            if consumed > semi_inv[s] + 1e-6:
                violations.append(f"半成品不足违规: {date_s} {s} 需{consumed} 但库存仅{semi_inv[s]}")
            semi_inv[s] = max(0, semi_inv[s] - consumed)
        for s in semi_types:
            self_q, buy_q = semi.get((date_s, s), (0, 0))
            semi_inv[s] += self_q + buy_q

    if violations:
        return violations, None

    # ── 独立重算 5 类罚款 ──
    # 1. 延期罚款
    fg_inv = dict(FG_INIT)
    cum_dem = {m: 0 for m in MODELS}
    delay_events = []
    for di, d in enumerate(DAYS):
        date_s = d_str(d)
        for m in MODELS:
            cum_dem[m] += DEMAND[m][di]
            prod = sum(fg.get((date_s, l, m), 0) for l in FG_LINES)
            fg_inv[m] = fg_inv.get(m, 0) + prod - DEMAND[m][di]
            if fg_inv[m] < 0:
                delay_events.append({"date": date_s, "model": m})
    delay_penalty = len(delay_events) * DELAY_COST

    # 2. 转产罚款
    changeover_count = 0
    prev_models = {l: None for l in FG_LINES}
    for d in DAYS:
        if not is_fg_open(d):
            continue
        date_s = d_str(d)
        for line in FG_LINES:
            active = [m for m in MODELS if fg.get((date_s, line, m), 0) > 0
                      and line in MODEL_LINES[m]]
            if not active:
                continue
            if prev_models[line] is not None and active[0] != prev_models[line]:
                changeover_count += 1
            changeover_count += len(active) - 1
            prev_models[line] = active[-1]
    changeover_penalty = changeover_count * CHANGEOVER_COST

    # 3. 2# 产线开工费
    line2_days = sum(
        1 for d in DAYS
        if is_fg_open(d) and any(fg.get((d_str(d), '2#', m), 0) > 0 for m in MODELS)
    )
    line2_penalty = line2_days * LINE2_COST

    # 4. 期末超库存
    excess_inv_penalty = sum(
        max(0, fg_inv.get(m, 0) - EXCESS_INV_THRESHOLD) * EXCESS_INV_COST
        for m in MODELS
    )

    # 5. BB外购费
    bb_buy_total = sum(
        buy_q for (date_s, semi_type), (self_q, buy_q) in semi.items()
        if semi_type == 'BB'
    )
    bb_buy_penalty = bb_buy_total * BB_BUY_COST

    total_penalty = (delay_penalty + changeover_penalty + line2_penalty
                     + excess_inv_penalty + bb_buy_penalty)

    metrics = {
        "total_penalty": total_penalty,
        "delay_penalty": delay_penalty,
        "delay_events": len(delay_events),
        "changeover_penalty": changeover_penalty,
        "changeover_count": changeover_count,
        "line2_penalty": line2_penalty,
        "line2_days": line2_days,
        "excess_inv_penalty": excess_inv_penalty,
        "bb_buy_penalty": bb_buy_penalty,
        "bb_buy_total": bb_buy_total,
        "final_inv": {m: fg_inv.get(m, 0) for m in MODELS},
    }
    return [], metrics


def load_baseline():
    with open(os.path.join(_HERE, "baseline", "reference_metrics.json")) as f:
        return float((lambda _d:_d.get("reference_value",_d.get("baseline_cost")))(json.load(f)))


def evaluate(file_path, data_dir=None):
    metrics = {"validity_score": 0.0, "quality_score": 0.0, "overall_score": 0.0, "error_info": {}}
    try:
        load_data(data_dir or _DATA)
        baseline_cost = load_baseline()

        if not os.path.exists(file_path):
            metrics["error_info"] = {"fatal": [f"File not found: {file_path}"]}
            return metrics
        with open(file_path, "r", encoding="utf-8") as f:
            sub = json.load(f)

        if not isinstance(sub, dict) or "fg_schedule" not in sub:
            metrics["error_info"] = {"fatal": ["Missing 'fg_schedule' in solution.json"]}
            return metrics
        if "semi_schedule" not in sub:
            metrics["error_info"] = {"fatal": ["Missing 'semi_schedule' in solution.json"]}
            return metrics
        if not isinstance(sub["fg_schedule"], list) or not isinstance(sub["semi_schedule"], list):
            metrics["error_info"] = {"schema": ["fg_schedule/semi_schedule must be arrays"]}
            return metrics

        violations, m = check_and_score(sub)
        if violations:
            metrics["error_info"] = {"constraint": violations[:10],
                                     "violation_count": len(violations)}
            return metrics

        metrics["validity_score"] = 1.0

        total_penalty = m["total_penalty"]
        # 最小化目标：quality = baseline / player；不截断，选手更优时可 >1。
        # total_penalty 为 0 时使用统一的有限完美解哨兵。
        if total_penalty > 0:
            quality = baseline_cost / total_penalty
        else:
            quality = PERFECT_Q
        metrics["quality_score"] = round(quality, 6)
        metrics["overall_score"] = round(quality, 6)
        metrics["baseline_cost"] = baseline_cost
        metrics["total_penalty"] = total_penalty
        metrics["player_objective"] = total_penalty
        metrics["reference_value"] = baseline_cost
        metrics.update({k: m[k] for k in (
            "delay_penalty", "delay_events", "changeover_penalty", "changeover_count",
            "line2_penalty", "line2_days", "excess_inv_penalty", "bb_buy_penalty",
            "bb_buy_total")})

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
