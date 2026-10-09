"""归一化守恒校验：extractor 只能改表示，不能改事实。

prompt 里的白名单是软约束——模型可能遵守也可能不遵守。这里做硬校验，两道：

1. **token 多重集**：抓「凭空造内容」（补行、填值、重算）。
2. **分组结构**：抓「重新安排内容」。

第 2 道是 2026-08-13 补的，起因是第 1 道**结构性地拦不住重排**：

    源:    车1=[A,B,C]  车2=[D,E]     多集 {A,B,C,D,E}
    输出:  车1=[A,B,D]  车2=[C,E]     多集 {A,B,C,D,E}   ← 一致，放行

实测 results/smoke/03_city_* 四个条件里三个中招：agent 已写出目标 schema 的
routes.json，extractor 却输出了不同分组，notes 还写着 "already conforms to the
required schema"，status=success，分数照常算出来，全程无声。C 条件更甚——
源 routes.csv 有权威 stop_seq，extractor 输出的路线 0 含 167 个客户（权威 133）。

顺序仍然允许改（路线内访问顺序的重排是合法的格式操作），**分组不允许**。
"""

from __future__ import annotations

import csv
import io
import json
import re
from collections import Counter, defaultdict
from pathlib import Path

# 形如 "0001" / 12345 的实体标识；用于比对决策集合
_TOKEN = re.compile(r"[A-Za-z0-9_\-]{2,}")

# 依赖/缓存目录：.agent_home 下有数万个 node_modules 文件，
# 全量遍历会让校验挂死（实测扫描超 2 分钟无输出）
_SKIP_DIR = {".agent_home", "node_modules", ".opencode_data",
             "__pycache__", ".git", "data"}


def _tokens(path: Path) -> Counter:
    """把任意文本/JSON/CSV 文件拍平成 token 多重集。"""
    try:
        raw = path.read_text(encoding="utf-8", errors="replace")
    except Exception:
        return Counter()
    return Counter(_TOKEN.findall(raw))


def _dir_tokens(d: Path, skip: set[str] | None = None) -> Counter:
    skip = skip or set()
    total = Counter()
    if not d.is_dir():
        return total
    for p in d.rglob("*"):
        if any(part in _SKIP_DIR for part in p.parts):
            continue
        if not p.is_file():
            continue
        if p.name in skip or p.name.startswith("_"):
            continue
        if p.suffix in (".py", ".log", ".md", ".txt", ".png", ".pyc"):
            continue
        total += _tokens(p)
    return total



# ── 分组结构 ──────────────────────────────────────────────────────────────────

def _norm(v) -> str:
    """ID 归一：'123' 与 123 与 ' 123 ' 视为同一个。

    int/str 混用是 agent 产物里最常见的表示差异，也是白名单里 cast_id_type
    这条修复的由来——它属于「改表示」，不该被结构校验误判成「改事实」。
    """
    s = str(v).strip()
    if s.endswith(".0") and s[:-2].isdigit():
        s = s[:-2]
    return s


def _groups_from_json(obj) -> list[frozenset] | None:
    """从 JSON 里找出「一组一组的 ID」。

    认两种形状（覆盖本 benchmark 的产物形态）：
        {"routes": [{"customers": [id, ...]}, ...]}
        {"routes": [[id, ...], ...]}
    """
    if isinstance(obj, dict):
        for val in obj.values():
            got = _groups_from_json(val)
            if got:
                return got
        return None
    if not isinstance(obj, list) or not obj:
        return None

    groups: list[frozenset] = []
    for item in obj:
        if isinstance(item, list):
            inner = item
        elif isinstance(item, dict):
            inner = next((v for v in item.values()
                          if isinstance(v, list)
                          and all(not isinstance(x, (list, dict)) for x in v)), None)
            if inner is None:
                return None
        else:
            return None
        groups.append(frozenset(_norm(x) for x in inner))
    return groups if len(groups) > 1 else None


def _groups_from_csv(text: str) -> list[frozenset] | None:
    """CSV：按「取值重复的列」分组，收集「取值近乎唯一的列」作为 ID。

    典型形态 (vehicle_id, stop_seq, 网点编码, ...)：vehicle_id 重复 -> 分组键，
    网点编码基本唯一 -> 成员 ID。
    """
    try:
        rows = list(csv.DictReader(io.StringIO(text)))
    except Exception:
        return None
    if len(rows) < 4 or not rows[0]:
        return None

    cols = list(rows[0].keys())
    n = len(rows)
    uniq = {c: len({_norm(r.get(c, "")) for r in rows}) for c in cols}

    # 分组键：取值数介于 [2, n/3]，避免把 ID 列或常量列当成分组键
    keys = [c for c in cols if 2 <= uniq[c] <= max(2, n // 3)]
    # 成员列：取值数最接近行数的那列
    members = [c for c in cols if uniq[c] >= n * 0.8]
    if not keys or not members:
        return None
    key, member = keys[0], members[0]

    buckets: dict[str, set] = defaultdict(set)
    for r in rows:
        v = _norm(r.get(member, ""))
        if v:
            buckets[_norm(r.get(key, ""))].add(v)
    groups = [frozenset(s) for s in buckets.values() if s]
    return groups if len(groups) > 1 else None


def _groups_of_file(path: Path) -> list[frozenset] | None:
    try:
        raw = path.read_text(encoding="utf-8", errors="replace")
    except Exception:
        return None
    if path.suffix == ".json":
        try:
            return _groups_from_json(json.loads(raw))
        except json.JSONDecodeError:
            return None
    if path.suffix == ".csv":
        return _groups_from_csv(raw)
    return None


def _as_partition(groups: list[frozenset]) -> tuple:
    """归一成可比较的分组签名（组间无序、组内无序——顺序允许改）。"""
    return tuple(sorted(tuple(sorted(g)) for g in groups))


def check_structure(workspace: Path, normalized: Path) -> dict:
    """校验 normalized 的分组与 workspace 里某个源文件一致。

    判定刻意保守——**只在能拿出反证时才判违规**：
      找不到 normalized 的分组结构   -> skip（不是所有 case 都有分组语义）
      workspace 里没有可比的候选源   -> skip（无从对照，不能凭空定罪）
      有候选源且其中之一与之相同     -> ok
      有候选源但全都不同             -> 违规（extractor 重排了解）

    "可比" = 成员 ID 集合完全相同。只有 ID 集相同、分组不同，才能断定是重排，
    而非「抽取了另一份不同的产物」。
    """
    out_groups, out_file = None, None
    for p in sorted(normalized.glob("*")):
        if p.is_file() and not p.name.startswith("_"):
            out_groups = _groups_of_file(p)
            if out_groups:
                out_file = p.name
                break
    if not out_groups:
        return {"checked": False, "reason": "normalized 无可识别的分组结构"}

    out_ids = frozenset().union(*out_groups)
    out_sig = _as_partition(out_groups)

    candidates = []
    for p in workspace.rglob("*"):
        if any(part in _SKIP_DIR for part in p.parts) or not p.is_file():
            continue
        if p.suffix not in (".json", ".csv") or p.name.startswith("_"):
            continue
        g = _groups_of_file(p)
        if not g:
            continue
        if frozenset().union(*g) != out_ids:
            continue                     # ID 集不同 -> 不是同一份决策，无从比对
        candidates.append((p.name, _as_partition(g)))

    if not candidates:
        return {"checked": False, "reason": "workspace 无 ID 集相同的候选源"}

    for name, sig in candidates:
        if sig == out_sig:
            return {"checked": True, "ok": True,
                    "matched_source": name, "output_file": out_file}

    return {
        "checked": True, "ok": False,
        "output_file": out_file,
        "sources_compared": [n for n, _ in candidates],
        "detail": "分组与源不一致：extractor 重新安排了 agent 的解",
    }


def check_conservation(workspace: Path, normalized: Path) -> dict:
    """比对 normalized 与 workspace 的 token 多重集。

    返回 {"ok": bool, "invented": [...], "dropped_ratio": float, ...}

    invented —— normalized 里出现、workspace 里从未出现的 token。这是最硬的
    违规信号：extractor 凭空造了 agent 没写过的内容（补行、填值、重算）。

    dropped —— workspace 有而 normalized 没有的 token。这个宽容得多：extractor
    本来就要丢掉日志、调试字段、agent 的中间产物，故只在丢失比例极高时告警。
    """
    ws = _dir_tokens(workspace, skip={"customers.csv"})
    nm = _dir_tokens(normalized)
    if not nm:
        return {"ok": True, "note": "normalized 为空，无需校验"}

    invented = [t for t in nm if t not in ws]
    # 数字与 ID 的凭空出现最可疑；纯英文单词多为 schema 关键字（routes/customers）
    suspicious = [t for t in invented if any(c.isdigit() for c in t)]

    kept = sum(1 for t in nm if t in ws)
    ratio = kept / max(1, len(nm))

    structure = check_structure(workspace, normalized)
    regrouped = bool(structure.get("checked") and not structure.get("ok"))

    return {
        "ok": not suspicious and not regrouped,
        "invented_tokens": suspicious[:20],
        "invented_count": len(suspicious),
        "preserved_ratio": round(ratio, 3),
        "structure": structure,
        "regrouped": regrouped,
    }


def verdict(extract_status: str, conservation: dict) -> str:
    """综合 extractor 自报状态与守恒校验，给出最终判定。"""
    if not conservation.get("ok", True):
        return "tampered"          # 凭空造内容或重排了解，该 run 不可用
    return extract_status
