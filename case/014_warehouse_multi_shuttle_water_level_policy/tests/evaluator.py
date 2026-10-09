# -*- coding: utf-8 -*-
"""
evaluator.py —— FDE-Bench多穿库水位策略 case 评估器(benchmark-online·闭卷·QAS 真实库口径版)。

本 case 与同目录的 `warehouse_water_level_policy`(MILP 仿真口径)是**姊妹 case**:
选手契约完全相同(交纯函数 `policy(ctx) -> {sku_id: W*}`),但评估**吃真实库冻结数据**、
按客户运营口径 HIT/WASTE/FILL 打分,不跑 MILP。真值来自现场取数冻结的 fixture
`tests/_private/ground_truth.csv`(13 个真实工作日 2026-07-15~07-29,剔周日 07-19/07-26)。

【STO 口径,2026-07-30 更新对齐已上线交付】fixture 除逐窗 SIG(DS/FD 剩余需求)外,另冻结
两路 STO(仓间调拨)真值:kind=STO(当天 STO 出货,进 ACT 出货真值)、kind=STOREM(逐窗 STO
待发需求)。评估器逐窗把 STOREM 折进 ctx.rem、并把纯 STO SKU 补进 `_window_skus`——精确复刻
已上线交付 `ctx_reader` 的 `SHIPMENT_TYPE IN('DS','FD','STO')` 需求视野(见现场记忆⑬)。这样
policy 决策时能看见 STO 需求、评估时 ACT 也含 STO,分子分母对称;否则纯 STO SKU 永远算漏推
(=07-29 FILL 崩的根因,现场记忆⑫)。

EXTRACTOR_SPEC:
  plan_file: solution.py
  required_columns: []
  notes: >
    选手产物是一个 Python 模块 solution.py,里面暴露一个纯函数 policy(ctx) -> {sku_id: W*}
    (各 SKU 的多穿库箱库水位上限,箱,>=0)。这不是表格,没有列。
    你要做的:在 workspace 里找到选手最终的策略代码文件(通常叫 solution.py / policy.py /
    submission.py,或 _agent_summary.md 指认的那个),【原样拷贝】成 output 目录下的 solution.py。
    - 不要执行它,不要改任何一行,不要重跑/重新优化(评估器自己会 import 并调用 policy)。
    - 若有多个候选文件,用 _agent_summary.md 判断选手最终选定的那个;拿不准就选暴露了
      顶层可调用 policy(ctx) 的那个 .py。
    - 若找不到任何暴露 policy(ctx) 的文件,status=failed。

== 任务(闭卷·交策略函数·评估器驱动)==
  选手交纯函数 policy(ctx) -> {sku_id: W*}。评估器逐(天×4h窗)从真实库冻结快照重建 ctx、
  喂给 policy 拿 W*,跨窗聚合成当日水位(各窗最大)、推荐并集、码板量;再对比当天【真实出货】
  算客户三指标 HIT / WASTE / FILL。真实出货(ACT)是私有标签,只在 tests/_private/ 内、
  **绝不进 ctx** —— 结构性防泄漏(选手看得到 ctx 里的信号,永远看不到当天真出货)。

== ctx 里有什么(决策时可见,causal)==
  当窗富信号:rem/rem_bulk/rem_parcel/rem_ds/rem_fd/rem_2nd(各口径剩余需求)、air(空运急件)、
  weight/n_orders(订单画像)、priority(优先级)、shipped_today(今日已出)、picked_prior(先前已拣)、
  dps_today/dps_next(排产投料前瞻)、box_age/asrs_age(库龄)、asrs_avail/asrs_box(立库可用)、
  s_ms/s_as(箱库/立库现货)、P(板规)、water_hist(历史水位)、ess(未开单净需求前瞻)、
  prod_ahead(当日剩余计划投料)。字段稀疏(0 值留空=缺省),policy 自行组合利用,任一列缺→缺省不崩。

== 评分(越大越好,2026-07-30 v3 口径:三门等权重 + day_score 波动罚)==
  逐日:day_score = 100×(1−WASTE)            # WASTE 主奖励,全程给下压梯度(连续奖励,保留)
                  − 200×max(0, WASTE−0.40)   # WASTE 门:>40 每超 0.01 罚 2 分
                  − 200×max(0, 0.60−HIT)     # HIT 门:<60 每缺 0.01 罚 2 分(与 WASTE 门等权重)
                  − 200×max(0, 0.70−FILL)    # FILL 门:<70 每缺 0.01 罚 2 分(与 WASTE 门等权重)
                  + [10×max(0,HIT−0.60) + 10×max(0,FILL−0.70)]     # 达标小额加分
                  − 50×max(0,(码板−PAL_CAP)/PAL_CAP)               # 码板超码垛能力相对超额罚
     缺货硬阈值:当日缺货率>0.50 → day_score = 100 − 1000×(缺货率−0.50)(灾难性缺货垫底)。
  综合:combined = mean(day_scores) − 0.6×pstdev(day_scores)   # 跨日 day_score 标准差波动罚,奖励全程稳
  评估日=fixture 内全部真实工作日(自动剔周日);无真实出货的日不计入。

== 对外接口(benchmark-online 约定)==
  evaluate(submission_dir, data_dir) -> {"validity_score":0/1, "quality_score":float,
                                         "overall_score":float, "error_info":{...}}
  quality = combined / reference_value(reference=交付冠军 21.1255,启发式非 MIP 最优,见 baseline)。
  真值(SIG+ACT+PACK+STO)由评估器私有持有(tests/_private/ground_truth.csv),独立重算,不信任选手自报。
"""
import os
import sys
import csv
import json
import time
import argparse
import datetime
import statistics
import traceback
import importlib.util
from collections import defaultdict

HERE = os.path.dirname(os.path.abspath(__file__))
PLAN_FILE = "solution.py"
GT_PATH = os.path.join(HERE, "_private", "ground_truth.csv")   # 私有真值(SIG ctx + ACT 标签 + PACK + STO 两侧)

# ─────────────────────── 决策/评分常量(2026-07-30 v3:三门等权重 200 + 跨日 day_score 标准差波动罚)───────────────────────
DECISION_HOURS = [0, 4, 8, 12, 16, 20]          # 每天 6 个 4h 决策窗
WD_CN = ["周一", "周二", "周三", "周四", "周五", "周六", "周日"]

KPI_HIT_TGT   = 0.60    # HIT 门:命中率过 60 即达标
KPI_FILL_TGT  = 0.70    # FILL 门:满足率过 70 即达标
KPI_WASTE_TGT = 0.40    # WASTE 目标线:>此值重罚(客户"不想看到 WASTE>40")
W_WASTE_REW   = 100.0   # ★WASTE 主奖励:day 基分 = 100×(1−WASTE),全程越低越高(连续奖励,保留)
W_WASTE_OVER  = 200.0   # ★WASTE>40 重罚:每超 0.01 罚 2 分(WASTE 门的有效斜率=200)
W_HIT_PEN     = 200.0   # ★HIT<60 门:每缺 0.01 扣 2 分(2026-07-30 与 WASTE 门等权重 100→200)
W_FILL_PEN    = 200.0   # ★FILL<70 门:每缺 0.01 扣 2 分(2026-07-30 与 WASTE 门等权重 100→200)
W_HIT_BONUS   = 10.0    # HIT>60 每超 1.0 小额加分
W_FILL_BONUS  = 10.0    # FILL>70 每超 1.0 小额加分
W_STD_PEN     = 0.6     # ★跨日 day_score 标准差波动罚(2026-07-30 启用,λ=0.6):奖励"全程稳"而非只稳 WASTE
W_RANGE_PEN   = 0.0     # 跨天 WASTE 极差罚(2026-07-30 关闭,由 day_score std 取代)
SHORT_HARD    = 0.50    # 缺货硬阈值:客户自身 FILL~66%(缺货 34%),放到 50% 只杀灾难性缺货
PAL_CAP       = 1440.0 * 4.0   # 码板每窗上限箱=码垛 1440/h×4h=5760;超则按相对超额罚
W_PAL_PEN     = 50.0    # 码板超上限惩罚权重

# ─────────────────────── 从 fixture 重建 ctx 的字段集(与 dump_mock/replay_mock 冻结口径一致)───────────────────────
SIG = ["rem", "rem_bulk", "rem_parcel", "rem_ds", "rem_fd", "rem_2nd", "air", "weight", "n_orders",
       "priority", "shipped_today", "picked_prior", "dps_today", "dps_next", "box_age", "asrs_age",
       "asrs_avail", "asrs_box", "s_ms", "s_as", "P", "water_hist", "ess", "prod_ahead"]


# ─────────────────────── 读取私有 fixture(SIG 逐窗 + ACT 私有标签 + PACK)───────────────────────
def _load_fixture(path):
    """解析 ground_truth.csv → (windows, wskus, act, pack, sto_rem)。稀疏存储:0 值留空(priority/P 例外恒存)。
       与 replay_cmp2_days.py 逐字节同口径。
       ACT 已含 STO 出货(现场取数时 ACT 真值本就 SHIPMENT_TYPE IN(DS,FD,STO)),故 kind=STO 仅作
       诊断/审计,不再二次累加进 act;kind=STOREM 是逐窗 STO 待发需求,评估器折进 ctx.rem(见 _build_ctx)。"""
    windows = defaultdict(lambda: defaultdict(dict))   # day -> (hour,w) -> feat -> {sku:val}
    wskus = defaultdict(lambda: defaultdict(list))     # day -> (hour,w) -> [sku]
    act = defaultdict(dict)                             # day -> {sku: act_box}   ← 私有标签,绝不进 ctx(已含 STO)
    pack = {}                                           # sku -> packspec(箱→pcs)
    sto_rem = defaultdict(lambda: defaultdict(dict))   # day -> (hour,w) -> {sku: STO待发箱}(折进 ctx.rem)
    # clean fixture 保留 UTF-8 BOM；使用 utf-8-sig 仅去除文件标记，不改变任何数据值。
    with open(path, encoding="utf-8-sig") as fh:
        rd = csv.DictReader((ln for ln in fh if not ln.startswith("#")))
        for r in rd:
            k = r["kind"]
            if k == "SIG":
                day = r["day"]; key = (float(r["hour"]), int(r["w"])); sku = r["sku"]
                for feat in SIG:
                    v = r.get(feat, "")
                    if v not in ("", None):
                        fv = float(v)
                        if fv != 0 or feat in ("priority", "P"):
                            windows[day][key].setdefault(feat, {})[sku] = fv
                wskus[day][key].append(sku)
            elif k == "ACT":
                act[r["day"]][r["sku"]] = float(r["act_box"] or 0)
            elif k == "PACK":
                pv = float(r["packspec"] or 0)
                if pv > 0:
                    pack[r["sku"]] = pv
            elif k == "STOREM":
                day = r["day"]; key = (float(r["hour"]), int(r["w"])); sku = r["sku"]
                q = float(r.get("rem") or 0)
                if q > 0:
                    sto_rem[day][key][sku] = q
            # kind=STO(STO 出货)ACT 已含,不重复累加;kind=ESSAUD 审计,评估不用 → 跳过
    return windows, wskus, act, pack, sto_rem


def _build_ctx(windows, wskus, sto_rem, day, key):
    """逐窗重建喂 policy 的 ctx(决策时可见量;ACT 永不在内)。与 replay_cmp2_days.build_ctx(+STO)一致。
       STO 折进 rem:把本窗 STOREM 待发量叠进 ctx.rem、纯 STO SKU 补进 _window_skus —— 复刻已上线
       交付 ctx_reader 的 SHIPMENT_TYPE IN(DS,FD,STO) 需求视野,使 policy 能看见 STO 需求。"""
    d = {feat: dict(mp) for feat, mp in windows[day][key].items()}   # 深拷贝各 feat(避免污染 fixture)
    sk = list(wskus[day][key])
    srem = sto_rem[day].get(key, {})
    if srem:
        d.setdefault("rem", {})
        seen = set(sk)
        for sku, q in srem.items():
            if q > 0:
                d["rem"][sku] = d["rem"].get(sku, 0.0) + q
                if sku not in seen:
                    sk.append(sku); seen.add(sku)
    d["_window_skus"] = sk
    d["hour"] = key[0]
    d["w"] = key[1]
    return d


# ─────────────────────── 逐日打分(与 evaluator_qas._eval_day 同口径,码板从 SIG s_ms 现算)───────────────────────
def _eval_day(day, policy, windows, wskus, sto_rem, act, pack, pack_dft):
    def _pk(i):
        v = pack.get(i, 0)
        return float(v) if v and float(v) > 0 else pack_dft

    water_day = {}          # 各 SKU 当日水位(各窗最大)= 推荐 Rec Carton Qty
    rec_skus = set()        # 各窗推荐并集 = HasRec
    pal_down_max = 0.0      # 各窗最大码板量(Σ 现货−水位,水位<现货)—— 受码垛能力约束
    # 修复：原实现不校验 policy 返回的 SKU 是否在候选清单里，跨日/凭空的 SKU 也计入
    # 推荐并集与水位。口径：当日各窗候选清单（含 STO 补进的 SKU）的并集；某窗推荐了
    # 当日别的窗在册的 SKU，与在那个窗推荐它等价（推荐集取六窗并集、水位取六窗最大），
    # 故按当日累计候选集判成员关系，不在集合内的一律忽略。
    day_cands = set()
    for key in windows[day]:
        day_cands.update(_build_ctx(windows, wskus, sto_rem, day, key).get("_window_skus", []))
    for key in sorted(windows[day], key=lambda x: x[0]):     # 逐窗 hour 升序
        W = policy(_build_ctx(windows, wskus, sto_rem, day, key))
        if not isinstance(W, dict):
            W = {}
        s_ms_win = windows[day][key].get("s_ms", {})         # 本窗箱库现货(码板量基准)
        pal_down = 0.0
        for i, wv in W.items():
            if i not in day_cands:
                continue
            wn = float(wv) if wv is not None else 0.0
            if not (wn > 0):
                continue
            rec_skus.add(i)
            if wn > water_day.get(i, 0.0):                   # 各窗最大水位 = Rec Carton Qty
                water_day[i] = wn
            ms_i = float(s_ms_win.get(i, 0) or 0)
            if ms_i > wn:                                    # 水位低于现货 → 码板量 = 现货−水位
                pal_down += (ms_i - wn)
        if pal_down > pal_down_max:
            pal_down_max = pal_down

    a = act.get(day, {})
    ship = set(i for i, v in a.items() if v > 1e-6)          # HasDemand = 当天真出货 SKU
    hit_set = rec_skus & ship
    dead = rec_skus - ship
    hit = (len(hit_set) / len(ship)) if ship else 0.0        # HIT = 命中/有需求(recall)
    # WASTE over HasRec(含 Dead): Σmax(0,Rec−Act)pcs / ΣRec pcs
    srec = wst = 0.0
    for i in rec_skus:
        pk = _pk(i); rq = water_day.get(i, 0.0) * pk; av = a.get(i, 0.0) * pk
        srec += rq; wst += max(0.0, rq - av)
    waste = (wst / srec) if srec > 1e-6 else 0.0
    # FILL: Σ_{HasRec} min(Rec,Act)pcs / Σ_{HasDemand} Act pcs
    sfill = sum(min(water_day.get(i, 0.0) * _pk(i), a.get(i, 0.0) * _pk(i)) for i in rec_skus)
    sdem = sum(a.get(i, 0.0) * _pk(i) for i in ship)
    fill = (sfill / sdem) if sdem > 1e-6 else 1.0
    short = max(0.0, 1.0 - fill)
    dow_idx = datetime.date.fromisoformat(day).weekday()
    return dict(dow=WD_CN[dow_idx], HIT=hit, WASTE=waste, FILL=fill, 缺货率=short,
                码板=round(pal_down_max), n_rec=len(rec_skus), n_ship=len(ship),
                n_hit=len(hit_set), n_dead=len(dead), n_missed=len(ship - rec_skus))


def _day_score(k):
    # 2026-07-30 v3:WASTE 连续奖励 100×(1−WASTE) 保留;三条 KPI 门等权重(HIT/FILL/WASTE 各斜率 200)。
    # 2026-08-09 补丁:所有分支末尾 clip 到 [0, +inf),避免罚项失控让整体分数变负。
    if k["缺货率"] > SHORT_HARD:
        # 2026-08-23 修:原式 100−1000×(缺货率−0.5) 是**从满分起步**往下掉,导致缺货率
        # 刚跨过 0.50 反而拿到近 100 的日分,而常规分支在 FILL=0.50 处光 FILL 门就先扣
        # 40 分(约 30 分)——形成一条"越差越高分"的向上悬崖,且日分恒定还免掉了跨日
        # 波动罚。灾难性缺货按定义就该判 0,不再给任何残值。
        return 0.0
    base = W_WASTE_REW * (1.0 - k["WASTE"])                        # ★WASTE 主奖励:越低越高
    waste_over = W_WASTE_OVER * max(0.0, k["WASTE"] - KPI_WASTE_TGT)  # WASTE 门:>40 每超 0.01 罚 2 分
    gate = (W_HIT_PEN  * max(0.0, KPI_HIT_TGT  - k["HIT"])            # HIT 门:<60 每缺 0.01 罚 2 分
            + W_FILL_PEN * max(0.0, KPI_FILL_TGT - k["FILL"]))        # FILL 门:<70 每缺 0.01 罚 2 分
    bonus = (W_HIT_BONUS  * max(0.0, k["HIT"]  - KPI_HIT_TGT)
             + W_FILL_BONUS * max(0.0, k["FILL"] - KPI_FILL_TGT))
    pal_pen = W_PAL_PEN * max(0.0, (k.get("码板", 0) - PAL_CAP) / PAL_CAP)
    return max(0.0, base - waste_over - gate + bonus - pal_pen)


def _combined(per_day):
    ds = [_day_score(k) for k in per_day.values()]
    base = sum(ds) / len(ds) if ds else 0.0
    if len(per_day) >= 2:
        # ★2026-07-30 v3:波动罚改为跨日 day_score 标准差(奖励"全程稳",不只稳 WASTE 一个维度)。
        if W_STD_PEN:
            base -= W_STD_PEN * statistics.pstdev(ds)
        # 旧口径 WASTE 极差罚(W_RANGE_PEN,现默认 0=关):留作可回退旋钮。
        if W_RANGE_PEN:
            vs = [k["WASTE"] for k in per_day.values()]
            base -= W_RANGE_PEN * (max(vs) - min(vs))
    # 2026-08-10 补丁:日分已各自 clip 到 [0,+inf),但波动罚在均值之后再减,
    # 仍可把总分压成负数(实测 -5.54 / -0.22)。下限夹在聚合层,不设上限——
    # 超过 reference 仍可 >1。
    return max(0.0, base)


# ─────────────────────── 选手 policy 加载 ───────────────────────
def _load_policy(submission_dir):
    path = os.path.join(submission_dir, PLAN_FILE)
    if not os.path.isfile(path):
        raise FileNotFoundError(f"缺 {PLAN_FILE}(选手须在 submission 目录暴露 policy(ctx))")
    spec = importlib.util.spec_from_file_location("_lz_qas_submission", path)
    mod = importlib.util.module_from_spec(spec)
    sys.modules["_lz_qas_submission"] = mod
    spec.loader.exec_module(mod)
    pol = getattr(mod, "policy", None)
    if not callable(pol):
        raise AttributeError(f"{PLAN_FILE} 未暴露可调用的 policy(ctx)")
    return pol


def _load_reference():
    with open(os.path.join(HERE, "baseline", "reference_metrics.json"), encoding="utf-8") as f:
        d = json.load(f)
    return float(d["reference_value"]), d.get("direction", "higher_is_better")


# ─────────────────────── 对外接口 ───────────────────────
def evaluate(submission_dir, data_dir=None):
    """benchmark-online 契约:{validity_score, quality_score, overall_score, error_info}(+ 附加诊断字段)。
       data_dir 仅为签名兼容;真值由评估器私有持有(§6 评估器驱动闭卷范式)。"""
    t0 = time.time()
    m = {"validity_score": 0.0, "quality_score": 0.0, "overall_score": 0.0, "error_info": {}}
    ref_val, direction = _load_reference()
    m["reference_value"] = ref_val
    m["direction"] = direction
    try:
        policy = _load_policy(submission_dir)
    except Exception as e:
        m["error_info"] = {"fatal": [f"选手加载失败: {e}"]}
        return m
    try:
        windows, wskus, act, pack, sto_rem = _load_fixture(GT_PATH)
    except Exception as e:
        m["error_info"] = {"fatal": [f"私有 fixture 读取失败: {e}"]}
        return m
    _pv = [float(v) for v in pack.values() if v and float(v) > 0]
    pack_dft = statistics.median(_pv) if _pv else 1.0

    try:
        per_day = {}
        for day in sorted(windows):
            r = _eval_day(day, policy, windows, wskus, sto_rem, act, pack, pack_dft)
            if r["n_ship"] > 0:              # 无真实出货日(周日/数据缺)剔除
                per_day[day] = r
    except Exception:
        m["error_info"] = {"exception": "policy 运行异常", "traceback": traceback.format_exc()}
        return m
    if not per_day:
        m["error_info"] = {"fatal": ["无有效评估日(所有日无真实出货)"]}
        return m

    combined = _combined(per_day)
    worst_short = max(k["缺货率"] for k in per_day.values())
    # validity:policy 结构性可运行且产出有效水位即视为合法解;缺货以 in-score 大负分惩罚(客户自身即高缺货)。
    m["validity_score"] = 1.0
    quality = combined / ref_val if ref_val else 0.0        # 不加 min(,1) 截断
    m["quality_score"] = round(quality, 6)
    m["overall_score"] = round(m["validity_score"] * quality, 6)
    m["combined_score"] = round(combined, 4)
    m["cost_time"] = round(time.time() - t0, 3)
    if worst_short > SHORT_HARD:
        m["error_info"] = {"warn": [f"最差日缺货 {worst_short:.1%}>阈值 {SHORT_HARD:.0%}(已在 day_score 内重罚)"]}
    m["metric"] = {
        "综合分": round(combined, 4),
        "最差日缺货率": round(worst_short, 4),
        "日均WASTE": round(sum(k["WASTE"] for k in per_day.values()) / len(per_day), 4),
        "日均HIT": round(sum(k["HIT"] for k in per_day.values()) / len(per_day), 4),
        "日均FILL": round(sum(k["FILL"] for k in per_day.values()) / len(per_day), 4),
        "per_day": {d: {kk: (round(vv, 4) if isinstance(vv, float) else vv) for kk, vv in k.items()}
                    for d, k in per_day.items()},
    }
    return m


def main():
    ap = argparse.ArgumentParser(description="FDE-Bench水位策略 QAS 真实库口径评估器")
    ap.add_argument("--submission-dir", required=True, help="含 solution.py(暴露 policy(ctx))的目录")
    ap.add_argument("--data-dir", default=os.path.join(HERE, "..", "data"), help="公开数据契约目录(签名兼容)")
    a = ap.parse_args()
    print(json.dumps(evaluate(a.submission_dir, a.data_dir), ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
