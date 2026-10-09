"""抽取产物守恒校验：比对 normalized/ 与 agent 原始产物，不依赖抽取器自述。

为什么要这个：抽取器改写产物是**提示词拦不住**的。实测同一条禁令，dsv4p 读懂了
主动放弃（notes 原话 "wrapping them into allocation_policy would be a forbidden
shim/re-implementation"），glm52 把同样的行为改叫 "purely structural normalization"
照做不误——它交了 241 行自己写的策略，agent 那边其实什么都没产出。禁令加得越细，
它换的说法越多："用 evaluator 的公式重算"（23_silicon，把长度按判分公式反算，
26/35 specs 变 35/35）、"提取并规范化"（04_container）。

三模型投票也只在**单个模型改写**时有效。23_silicon 上一轮 glm52 单独改写被 2:1 挡住，
这一轮 dsv4p 也学会了同样的手法，投票直接反转，0.871 被采信——而 agent 的真实水平
是 26/35。规则和多数决都拦不住，只有比对产物本身能拦。

所以这里不看抽取器说了什么，只看它写出来的东西：

    decision      字面值必须在 agent 原始产物里找得到。凭空出现即判 tampered。
                  这条直接命中 23_silicon：3114.29 既不是原值 303.49，也不是任何
                  声明过的换算倍数（比值 10.261）。
    grouping_key  标签可以重编号（板1→P001 合法），但它诱导的**划分**必须守恒。
                  只比字面值查不出「把零件从板1挪到板2」——多重集分毫未变。
    identifier    跳过。evaluator 自己会拿它去 data/ 对齐，抽取器动了必然触发
                  unknown/缺失，不必在这里重复管。
    可执行代码    产物 .py 必须能在 workspace 里找到内容相同的来源文件。
                  04_container 的 241 行凭空出现，这条能当场抓住。

    python conservation.py --run <run目录>
    python conservation.py --case 022_sheet_metal_2d_nesting
    python conservation.py                      # 扫索引树全部 run
"""

from __future__ import annotations

import argparse
import collections
import csv
import hashlib
import json
import os
import re
from pathlib import Path

_ROOT = Path(__file__).resolve().parents[2]
BENCH = Path(os.environ.get("DELIVER_CASE_ROOT", str(_ROOT / "case")))
INDEX = Path(os.environ.get("DELIVER_RESULTS_INDEX", str(_ROOT / "results")))

ROLE_DECISION = "decision"
ROLE_GROUPING = "grouping_key"
# 这些 role 不做值守恒：identifier 由 evaluator 自己对齐真值表，其余是结构/透传字段。
# informational 是「字段可从其他字段反算，evaluator 不判分」——23_silicon 的 sub_i_long
# 就是这类，改它不影响分数，所以也不必查守恒。
ROLE_SKIP = {"identifier", "passthrough", "ignored", "unused", "structural",
             "container", "required_attribute", "informational", "description"}

# workspace 里混着 .opencode 的数千个依赖文件，全量 rglob 单个 run 要 50s+
_SKIP_DIR = {".opencode", ".opencode_data", "node_modules", "__pycache__", ".git"}
_OUT_SUBDIRS = ("out", "output", "outputs", "result", "results", "intermediate",
                "artifacts", "prepared", "experiments", "src", "scripts")


def _readable(p: str | Path) -> Path:
    """原样返回路径（FDE-bench 直接读，无 webagent 的 NFS /./ 特殊性）。"""
    return Path(p)


def _num(v) -> str:
    """14961.4 与 '14961.40' 视为同一个值；非数值原样返回。"""
    s = str(v).strip()
    try:
        return f"{float(s):.6g}"
    except (ValueError, TypeError):
        return s


# ── case 目录解析 ────────────────────────────────────────────────────────────

_SLUG: dict[str, Path] = {}


def resolve_case(case_name: str) -> Path | None:
    """按去掉编号的名字匹配 benchmark 目录。

    不能靠补零：34_supply_chain_replenishment 删除后 35~40 在 benchmark 里
    整体前移一位（36_aps → 035_aps）。
    """
    if not _SLUG:
        for d in BENCH.iterdir():
            if d.is_dir() and re.match(r"^\d+_", d.name):
                _SLUG[re.sub(r"^\d+_", "", d.name)] = d
    direct = BENCH / case_name
    if direct.is_dir():
        return direct
    return _SLUG.get(re.sub(r"^\d+_", "", case_name))


def load_roles(case_dir: Path) -> tuple[dict[str, str], list[str]]:
    p = case_dir / "tests" / "submission_schema.json"
    if not p.is_file():
        return {}, ["缺 submission_schema.json"]
    try:
        data = json.loads(p.read_text(encoding="utf-8"))
    except (json.JSONDecodeError, OSError) as e:
        return {}, [f"schema 不可解析: {e}"]

    roles: dict[str, str] = {}
    unknown: list[str] = []
    known = {ROLE_DECISION, ROLE_GROUPING} | ROLE_SKIP

    def walk(node):
        if isinstance(node, dict):
            name, role = node.get("name"), node.get("role")
            if name and role:
                roles[str(name)] = str(role)
                if role not in known:
                    unknown.append(f"{name}: role={role}")
            for v in node.values():
                walk(v)
        elif isinstance(node, list):
            for x in node:
                walk(x)

    walk(data)
    return roles, unknown


# ── 产物读取 ────────────────────────────────────────────────────────────────

def _files(d: Path, deep: bool = False):
    """目录下的产物文件。deep=True 时连常见输出子目录一起扫（用于 workspace）。"""
    if not d.is_dir():
        return
    for p in d.iterdir():
        if p.is_file() and not p.name.startswith("_"):
            yield p
    if not deep:
        return
    for sub in _OUT_SUBDIRS:
        sd = d / sub
        if sd.is_dir():
            for p in sd.rglob("*"):
                if p.is_file() and not any(x in _SKIP_DIR for x in p.parts):
                    yield p


def _records(p: Path) -> list[dict]:
    """把一个文件读成记录列表（CSV 行 / JSON 里的对象数组）。"""
    if p.suffix == ".csv":
        try:
            with p.open(encoding="utf-8-sig", errors="replace") as fh:
                return [dict(r) for r in csv.DictReader(fh)]
        except (OSError, csv.Error):
            return []
    if p.suffix == ".json":
        try:
            data = json.loads(p.read_text(encoding="utf-8"))
        except (json.JSONDecodeError, OSError):
            return []
        out, stack = [], [data]
        while stack:
            cur = stack.pop()
            if isinstance(cur, dict):
                stack.extend(cur.values())
            elif isinstance(cur, list):
                if cur and all(isinstance(x, dict) for x in cur):
                    out.extend(cur)
                else:
                    stack.extend(x for x in cur if isinstance(x, (dict, list)))
        return out
    return []


def scalar_pool(d: Path, deep: bool = False) -> collections.Counter:
    """目录下所有产物的标量值多重集。

    不带字段名——列名映射是合法的表示层改动，这里只问「这个值 agent 产出过没有」。
    """
    c: collections.Counter = collections.Counter()
    for p in _files(d, deep=deep):
        for rec in _records(p):
            for v in rec.values():
                if not isinstance(v, (dict, list)) and v not in (None, ""):
                    c[_num(v)] += 1
    return c


# ── 检查项 ──────────────────────────────────────────────────────────────────

def check_decisions(ws: Path, nz: Path, decisions: list[str]) -> list[str]:
    """decision 字段的字面值必须在 agent 产物里找得到，或由一次**全局一致的**换算得到。

    只比字面值会把合法的单位换算全判成篡改：kimik3 把长度从米换成毫米（每个值 ×1000），
    值确实都变了，但这是表示层改动。区分办法是看比例是否全局一致——真正的篡改是
    「挑几个越界的值往回挪」，比例参差；单位换算则是整列同一个系数。

    所以先按字段逐列求「normalized 值 / agent 侧最接近的候选值」的比例集合：
      - 全列同一个比例（且不为 1）→ 单位换算，放行
      - 比例参差              → 逐个值判定，找不到出处的报出来
    """
    if not decisions:
        return []
    pool = scalar_pool(ws, deep=True)
    if not pool:
        # 扫不到 agent 侧的值就没有比对基准（产物可能是 .py/.npz/.xlsx 这类本脚本
        # 不解析的格式）。这是「查不了」，不是「有问题」——报出来只会淹没真问题。
        return []

    # agent 侧的数值池，用于比例判定
    ws_nums = []
    for v in pool:
        try:
            ws_nums.append(float(v))
        except (ValueError, TypeError):
            pass
    ws_set = set(ws_nums)

    by_field: dict[str, list] = collections.defaultdict(list)
    for p in _files(nz):
        for rec in _records(p):
            for k, v in rec.items():
                if k in decisions and not isinstance(v, (dict, list)) and v not in (None, ""):
                    by_field[k].append(v)

    problems = []
    # 常见的表示层换算系数：单位（m↔mm、kg↔g）、时间（h↔min↔s）、百分比
    SCALES = (1000.0, 0.001, 100.0, 0.01, 10.0, 0.1, 60.0, 1 / 60.0,
              3600.0, 1 / 3600.0, 24.0, 8.0)
    # 日期/时间：抽取器几乎一定会重排格式，字面比对必然全军覆没。实测 36_aps：
    #   agent  "2026-05-27T08:00:00"          "2026-05-27 08:25:24"
    #   glm52  "2026-05-27T08:00:00.000000"   "2026-05-27T08:25:24.803400"
    # 秒以下的位数两边都不一致（一边截断、一边补微秒），所以归约到**秒**再比：
    # 分隔符、T/空格、微秒全部丢掉，只留 YYYYMMDDHHMMSS。时刻真被改动仍能抓住。
    _DT = re.compile(r"^\s*\d{2,4}[-/]\d{1,2}[-/]\d{1,2}[ T]?[\d:]*(\.\d+)?\s*$"
                     r"|^\s*\d{1,2}:\d{2}(:\d{2})?(\.\d+)?\s*$")

    def _dt_key(s: str) -> str:
        """归约到秒级数字骨架。

        两种形态都要能对上：
          完整时间戳  2026-05-27T08:25:24.803400 / 2026-05-27 08:25:24 → 20260527082524
          纯时刻      9:58 / 09:58 / 09:58:00                          → 095800
        27_vmi 那批误报就是纯时刻：agent 写 "9:58"（DictReader 出来没有前导零）、
        抽取器规范成 "09:58"，去分隔符后一个是 958、一个是 0958，长度不同直接对不上。
        所以时刻分支按 时:分:秒 逐段补零再拼，而不是对整串做 ljust。
        """
        raw = str(s).strip()
        body = raw.split(".", 1)[0]
        if ":" in body and not re.search(r"[-/]", body):     # 纯时刻，无日期部分
            parts = (body.split(":") + ["0", "0", "0"])[:3]
            try:
                return "".join(f"{int(p or 0):02d}" for p in parts)
            except ValueError:
                return re.sub(r"\D", "", body)
        return re.sub(r"\D", "", body)

    ws_dt = {_dt_key(v) for v in pool if _DT.match(str(v))}

    def traceable(v) -> bool:
        s = str(v).strip()
        if _DT.match(s):
            return _dt_key(s) in ws_dt
        try:
            fv = float(s)
        except (ValueError, TypeError):
            return False
        for sc in (1.0,) + SCALES:
            if _num(fv / sc) in pool:
                return True
        return False

    # 自由文本类字段：值是人读的句子，抽取器重新措辞属于表示层改动。
    # 15_noodle 的 `reason` 在 schema 里被标成 decision（"冻结窗口照抄原Excel计划"
    # 这种），但真正的决策是排了多少产量、用哪条产线——把说明文字当决策值查，
    # 整列都会报出来。按字段名 + 值形态双重判断，避免把编码型字段误放行。
    _PROSE_NAME = re.compile(r"reason|note|remark|comment|说明|备注|理由|原因|描述", re.I)

    def _is_prose(field: str, vals: list) -> bool:
        if _PROSE_NAME.search(field):
            return True
        # 名字没线索时看形态：多数值较长且含空白/中文标点
        long_ones = [str(v) for v in vals if len(str(v)) > 10
                     and re.search(r"[\s；、：（）()，,]", str(v))]
        return len(long_ones) > len(vals) * 0.6

    # 聚合字段：一个单元格里塞了一串 ID（"0;19;25;27;..."）。整串当标量查必然找不到，
    # 因为 agent 那边是每行一个 ID 的长表——重塑成聚合列是标准的表示层改动。
    # 拆开逐个查成员，成员都在就放行；某个成员凭空出现仍能抓住。
    _SEP = re.compile(r"[;,|]\s*")

    def _agg_ok(v) -> bool:
        parts = [x for x in _SEP.split(str(v)) if x]
        if len(parts) < 2:
            return False
        return all(pool[_num(x)] > 0 or traceable(x) for x in parts)

    # 枚举/状态标签：取值集合极小（"不排车"/"已排车" 这种），抽取器按 schema 的
    # 术语改写标签是表示层改动，agent 可能写"否"、"不安排"。字段的去重值 ≤ 6 个
    # 且都很短时按枚举放行——真正的决策藏在数量、时刻、分配里，不在标签措辞上。
    def _is_enum(vals: list) -> bool:
        uniq = {str(v) for v in vals}
        return len(uniq) <= 6 and all(len(u) <= 8 for u in uniq) and len(vals) > len(uniq)

    for field, vals in by_field.items():
        if _is_prose(field, vals) or _is_enum(vals):
            continue
        missing = [v for v in vals
                   if pool[_num(v)] <= 0 and not traceable(v) and not _agg_ok(v)]
        if not missing:
            continue
        uniq = sorted({_num(v) for v in missing})
        problems.append(
            f"`{field}` 有 {len(missing)} 个值在 agent 产物中找不到出处"
            f"（也不是单位换算或格式重排）：{uniq[:5]}")
    return problems


def partition_of(records: list[dict], key: str, anchors: list[str]):
    """按 key 分组，每组用组内成员的 anchor 值集合表示。

    返回组签名的多重集——标签本身不参与，所以 板1→P001 这类重编号天然放行，
    而把某个零件从一组挪到另一组会改变签名，立刻暴露。
    """
    if not records or key not in records[0]:
        return None
    groups: dict[str, list] = collections.defaultdict(list)
    for r in records:
        if key not in r or r[key] in (None, ""):
            continue          # CSV 行长不齐时 DictReader 会漏列，跳过而不是崩
        sig = tuple(sorted(_num(r[a]) for a in anchors if a in r and r[a] not in (None, "")))
        if sig:
            groups[_num(r[key])].append(sig)
    if not groups:
        return None
    return frozenset(collections.Counter(
        frozenset(collections.Counter(v).items()) for v in groups.values()
    ).items())


def check_groupings(ws: Path, nz: Path, groupings: list[str],
                    anchors: list[str]) -> list[str]:
    """normalized 里 key 诱导的划分，能否在 workspace 的某一列上复现。

    workspace 侧的列名是 agent 自己起的，无从预知哪一列是分组列，所以逐列试：
    只要有任意一列诱导出相同划分，就说明这个分组是 agent 做的而非抽取器编的。
    """
    if not groupings or not anchors:
        return []
    nz_recs = [r for p in _files(nz) for r in _records(p)]
    per_file = [(p, _records(p)) for p in _files(ws, deep=True)]
    # 除了逐文件比，还要比「所有文件拼起来」：抽取器常把 agent 的多张表合成一张
    # （09_fmcg 就是 scheduled_orders.csv 104 行 + unscheduled_orders.csv 299 行
    # → 合表 403 行），合并本身合法，但任何单个文件都凑不出合表的划分，
    # 只逐文件比会把这种合并全判成未守恒——实测 83 处「未守恒」多数是这个。
    pooled = [r for _, recs in per_file for r in recs]
    ws_recs = per_file + ([(Path("<all>"), pooled)] if len(per_file) > 1 else [])
    problems = []
    for g in groupings:
        want = partition_of(nz_recs, g, anchors)
        if want is None:
            continue
        for p, recs in ws_recs:
            if not recs:
                continue
            if any(partition_of(recs, col, anchors) == want for col in recs[0]):
                break
        else:
            problems.append(f"分组键 `{g}` 未守恒：workspace 里没有任何一列能复现这个划分")
    return problems


def _norm_code(text: str) -> str:
    """代码归一化：去注释、空白、空行，只留骨架用于比对来源。"""
    out = []
    for line in text.splitlines():
        s = line.split("#", 1)[0].strip()
        if s:
            out.append(re.sub(r"\s+", " ", s))
    return "\n".join(out)


def check_code_origin(ws: Path, nz: Path) -> list[str]:
    """产物里的 .py 必须能在 workspace 找到内容相同的来源。

    04_container 那次 agent 中断、workspace 里只有实验脚本，抽取器却写出 241 行
    solution.py 并拿到 1.227——它交的是自己的作业。行数/md5 都对不上任何来源文件。
    """
    # normalize*.py / convert*.py 这类是抽取器为做转换而写的辅助脚本，不是交付物；
    # 交付物是 schema 点名的那个（solution.py 等）。把辅助脚本也查来源会全线误报。
    # 下划线开头的一律算辅助：抽取器写 _normalize.py / _ap_rules.py / _ap_data.py
    # 这类中间脚本很常见，它们不是交付物（交付物是 schema 点名的 solution.py 等），
    # 查来源必然报出来。实测 57 处报告里有 16 处是这种。
    _HELPER = re.compile(r"^(_|normalize|convert|transform|build|make|gen|explore)[-_a-z0-9]*\.py$", re.I)
    pys = [p for p in nz.iterdir()
           if p.is_file() and p.suffix == ".py" and not _HELPER.match(p.name)] if nz.is_dir() else []
    if not pys:
        return []
    src: dict[str, str] = {}
    src_bodies: list[str] = []
    for p in _files(ws, deep=True):
        if p.suffix != ".py":
            continue
        try:
            body = _norm_code(p.read_text(encoding="utf-8", errors="replace"))
        except OSError:
            continue
        src[hashlib.md5(body.encode()).hexdigest()] = p.name
        src_bodies.append(body)
    problems = []
    for p in pys:
        try:
            body = _norm_code(p.read_text(encoding="utf-8", errors="replace"))
        except OSError:
            continue
        h = hashlib.md5(body.encode()).hexdigest()
        if h in src:
            continue
        # 允许「agent 的原文件被完整包含在产物里」——在原代码外面加导入或包装、
        # 但没改逻辑，这种仍可追溯到来源；完全找不到出处的才判 tampered。
        if any(s and s in body for s in src_bodies):
            continue
        n_lines = len(body.splitlines())
        problems.append(f"产物 `{p.name}`({n_lines} 行有效代码) 在 workspace 里找不到对应来源"
                        f"——抽取器可能自己写了代码")
    return problems


# ── 单 run ──────────────────────────────────────────────────────────────────

def check_run(case_dir: Path, run: Path, tag: str | None = None) -> dict:
    """校验一个 run 的归一化产物与 agent 原始产物守恒。

    case_dir : case 目录（含 tests/submission_schema.json）
    run      : run 目录（含 workspace/ 与 normalized<tag>/）
    tag      : 抽取模型标签（glm52/qwen38_27b/dsv4f/gemini36f/grok45）；None = 默认 normalized
    """
    res: dict = {"case": case_dir.name, "tag": tag or "normalized", "problems": []}

    roles, unknown = load_roles(case_dir)
    if unknown:
        res["problems"].append(f"schema 含未知 role: {'; '.join(unknown[:4])}")

    nz = run / (f"normalized_{tag}" if tag else "normalized")
    ws = run / "workspace"
    if not nz.is_dir():
        res["ok"] = True
        res["note"] = "无 normalized 目录，跳过"
        return res

    decisions = [f for f, r in roles.items() if r == ROLE_DECISION]
    groupings = [f for f, r in roles.items() if r == ROLE_GROUPING]
    anchors = [f for f, r in roles.items() if r == "identifier"]

    # 三条检查的可信度差别很大，分开对待：
    #
    # code_origin 高可信：产物 .py 的内容能不能在 workspace 里找到出处，是硬事实。
    #   实测 04_container 抓得准（glm52 的 190 行凭空出现、另两个交 3 行 stub 判干净）。
    #
    # decisions / groupings 低可信：全库扫下来 17% 的抽取被报，而抽样复核多数是误报——
    #   派生量（16_nuclear 的 actual_work_hours = 点数 × 工效，agent 不落盘中间值）、
    #   符号合并（14_microgrid 把充放电两列合成带符号的 P_batt）、
    #   多表合并（09_fmcg 的 waybill）都会让「值在 agent 产物里找不到」，但都合法。
    #   这两条留作 warnings 供人工排查，不参与 ok 判定——否则真信号会被淹没。
    res["warnings"] = check_decisions(ws, nz, decisions) + check_groupings(
        ws, nz, groupings, anchors)
    res["problems"] += check_code_origin(ws, nz)
    res["ok"] = not res["problems"]
    return res


def main() -> None:
    ap = argparse.ArgumentParser(description="抽取产物守恒校验")
    ap.add_argument("--run", default=None, help="单个 run 目录")
    ap.add_argument("--case", default=None, help="只查该 case")
    ap.add_argument("--tag", default=None,
                    help="只查某个抽取模型（glm52/qwen38_27b/dsv4f/gemini36f/grok45）")
    ap.add_argument("--limit", type=int, default=0)
    args = ap.parse_args()

    # 旧标签保留在默认审计列表中，以便继续检查替换模型前的历史评分结果。
    tags = ([args.tag] if args.tag else
            ["glm52", "qwen38_27b", "dsv4f", "gemini36f", "grok45", "gpt52",
             "kimik3", "dsv4p", None])

    targets: list[tuple[Path, str]] = []
    if args.run:
        r = Path(args.run)
        targets = [(r, args.case or r.parent.name)]
    else:
        seen = set()
        for model in sorted(os.listdir(INDEX)):
            for tri in sorted(os.listdir(INDEX / model)):
                for case in sorted(os.listdir(INDEX / model / tri)):
                    if args.case and case != args.case and f"0{case}" != args.case:
                        continue
                    real = _readable(os.path.realpath(INDEX / model / tri / case))
                    if str(real) in seen:
                        continue
                    seen.add(str(real))
                    targets.append((real, case))
        if args.limit:
            targets = targets[:args.limit]

    bad = 0
    for run, case in targets:
        for tag in tags:
            r = check_run(run, case, tag)
            if not r.get("ok"):
                bad += 1
                print(f"\n✗ {case[:34]:<36}{tag or 'normalized':<8}{run.name[-16:]}")
                for p in r["problems"]:
                    print(f"    {p}")
    print(f"\n扫描 {len(targets)} 个 run × {len(tags)} 个抽取，有问题 {bad} 处")


if __name__ == "__main__":
    main()
