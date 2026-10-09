"""从 agent 运行产物中采集 token / cache / 费用。

opencode 把每个 step 的 token 明细写进它自己的 SQLite（含 cache read/write），
但 cost 字段恒为 0——自建 provider 没有单价表。因此费用在此按 config.toml
的 [pricing] 段自行折算。

计价规则来自 webapp-experiment-log/模型价格表.md：
  1. 分档模型（gpt-5.6-sol >272k、gemini-3.1-pro >200k）按**每个 step** 的
     input+cache_read 判档，故费用必须逐 step 累加，不能先汇总再算。
  2. 无独立缓存创建价的模型（kimi-k3 / glm 系 / kimi-k2.7），cache_write
     按输入价兜底。

其他脚手架（OpenHands 等）若无同等数据源，返回 None，聚合时自动跳过而非记 0，
避免把"没采到"伪装成"没花钱"。
"""

from __future__ import annotations

import json
import sqlite3
from pathlib import Path

_PER_M = 1_000_000


def _find_db(run_dir: Path) -> Path | None:
    """定位本次 run 的 opencode SQLite。

    权威位置是 environment.py 建的私有 XDG：run_dir/.agent_home/data/opencode/。
    另一处 workspace/.opencode_data/ 是更早的布局，留作兼容——写死单一路径
    曾让 token/费用**静默**采不到（usage.json 里 token_usage 恒为 null，
    而 cli.py 只在"有非 None 字段"时才写，于是连字段都不出现，毫无报错）。
    """
    for pattern in (".agent_home/**/opencode/*.db",
                    "workspace/.opencode_data/opencode/*.db"):
        hits = sorted(run_dir.glob(pattern))
        if hits:
            return hits[0]
    return None




def _harness_secs(run_dir: Path | None) -> float:
    """clarify.json 里 harness 侧（detect + answerer）的累计用时。"""
    if run_dir is None:
        return 0.0
    path = Path(run_dir) / "clarify.json"
    if not path.is_file():
        return 0.0
    try:
        return float(json.loads(path.read_text(encoding="utf-8")).get("harness_secs") or 0.0)
    except (OSError, ValueError, json.JSONDecodeError):
        return 0.0


def _text_clarify_anchor_ms(run_dir: Path | None) -> int | None:
    """clarify.json 里文本通路最后一次回灌答案的时刻（epoch ms）；没有则 None。"""
    if run_dir is None:
        return None
    path = Path(run_dir) / "clarify.json"
    if not path.is_file():
        return None
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, ValueError, json.JSONDecodeError):
        return None
    stamps = [r.get("answered_at_ms") for r in data.get("rounds", [])
              if isinstance(r.get("answered_at_ms"), (int, float))]
    return int(max(stamps)) if stamps else None


def _phase_split(db: Path, run_dir: Path | None = None, *, had_clarify: bool = False) -> dict | None:
    """opencode：切分澄清 vs 求解。

    澄清结束锚点 = max(最后一条 question tool 时间戳, 最后一次文本回灌时刻)。
    两个通道都可能出现（实测 glm-5.3 一次 run 走 tool、下一次走正文），
    只认 tool 会把正文提问的 run 记成 clarify=0，再经 cli._sync 覆盖掉 loop
    记的真实时长。

    had_clarify=True 且找不到任何锚点 → clarify 记 None（"切不出来"），
    不记 0（"没澄清"）；调用方据此不去覆盖 clarify.json。
    返回 {"clarify_input": N, "solve_input": N, "clarify_secs": N, "solve_secs": N, ...}
    """
    import sqlite3
    try:
        c = sqlite3.connect(f"file:{db}?mode=ro", uri=True)

        qr = c.execute(
            "SELECT time_created FROM part"
            " WHERE json_extract(data,'$.type')='tool'"
            "   AND json_extract(data,'$.tool')='question'"
            " ORDER BY time_created DESC LIMIT 1"
        ).fetchone()
        anchors = [x for x in (qr[0] if qr else None, _text_clarify_anchor_ms(run_dir))
                   if x is not None]
        q_end = max(anchors) if anchors else None

        # 所有 step-finish 按时间排序
        rows = c.execute(
            "SELECT time_created,"
            " json_extract(data,'$.tokens.input'),"
            " json_extract(data,'$.tokens.output'),"
            " json_extract(data,'$.tokens.reasoning'),"
            " json_extract(data,'$.tokens.cache.read')"
            " FROM part"
            " WHERE json_extract(data,'$.type')='step-finish'"
            " ORDER BY time_created"
        ).fetchall()
        if not rows:
            return None

        if q_end is None:
            ts_first, ts_last = rows[0][0], rows[-1][0]
            if had_clarify:
                # 有澄清轮次却没有任何锚点（旧 clarify.json 无 answered_at_ms 等）：
                # 切不出来就说切不出来，别把它记成"零澄清"
                return {"clarify_tokens": None, "clarify_secs": None,
                        "solve_tokens": None, "solve_secs": None}
            # 没有澄清（Hidden/Full）→ 整个 run 都是求解阶段
            solve_input = sum(r[1] or 0 for r in rows)
            solve_output = sum(r[2] or 0 for r in rows)
            solve_reasoning = sum(r[3] or 0 for r in rows)
            return {
                "clarify_tokens": 0, "clarify_secs": 0.0,
                "solve_tokens":   solve_input + solve_output + solve_reasoning,
                "solve_secs":     round((ts_last - ts_first) / 1000, 1),
                "clarify_input": 0, "clarify_output": 0, "clarify_reasoning": 0,
                "solve_input": solve_input, "solve_output": solve_output,
                "solve_reasoning": solve_reasoning,
            }

        ts_first, ts_last = rows[0][0], rows[-1][0]
        clarify_input  = sum(r[1] or 0 for r in rows if r[0] <= q_end)
        solve_input    = sum(r[1] or 0 for r in rows if r[0] >  q_end)
        clarify_ms     = max(0, q_end - ts_first)
        solve_ms       = max(0, ts_last - q_end)
        clarify_output = sum(r[2] or 0 for r in rows if r[0] <= q_end)
        solve_output   = sum(r[2] or 0 for r in rows if r[0] >  q_end)
        clarify_reasoning = sum(r[3] or 0 for r in rows if r[0] <= q_end)
        solve_reasoning   = sum(r[3] or 0 for r in rows if r[0] >  q_end)

        return {
            "clarify_tokens": clarify_input + clarify_output + clarify_reasoning,
            "clarify_secs":   round(clarify_ms / 1000, 1),
            "solve_tokens":   solve_input + solve_output + solve_reasoning,
            "solve_secs":     round(solve_ms / 1000, 1),
            "clarify_input":  clarify_input,
            "clarify_output": clarify_output,
            "clarify_reasoning": clarify_reasoning,
            "solve_input":    solve_input,
            "solve_output":   solve_output,
            "solve_reasoning": solve_reasoning,
        }
    except Exception:
        return None
def _read_steps(db: Path) -> list[dict]:
    """取出每个 step-finish 的 token 明细。"""
    steps: list[dict] = []
    conn = sqlite3.connect(f"file:{db}?mode=ro", uri=True)
    try:
        for (data,) in conn.execute("SELECT data FROM part"):
            try:
                part = json.loads(data)
            except json.JSONDecodeError:
                continue
            if part.get("type") != "step-finish":
                continue
            tk = part.get("tokens") or {}
            cache = tk.get("cache") or {}
            steps.append({
                "input": tk.get("input", 0),
                "output": tk.get("output", 0),
                "reasoning": tk.get("reasoning", 0),
                "cache_read": cache.get("read", 0),
                "cache_write": cache.get("write", 0),
            })
    finally:
        conn.close()
    return steps


def _step_cost(step: dict, prices: dict) -> float:
    """单个 step 的费用，按需切换分档单价。"""
    rates = prices
    threshold = prices.get("tier_threshold")
    if threshold and (step["input"] + step["cache_read"]) > threshold:
        rates = {**prices, **(prices.get("above") or {})}

    # 无独立缓存创建价的模型，cache_write 按输入价兜底
    write_rate = rates.get("cache_write")
    if write_rate is None:
        write_rate = rates.get("input", 0)

    return (
        step["input"] / _PER_M * rates.get("input", 0)
        + step["output"] / _PER_M * rates.get("output", 0)
        + step["cache_read"] / _PER_M * (rates.get("cache_read") or 0)
        + step["cache_write"] / _PER_M * write_rate
    )


def _phase_token_total(part: dict) -> int | None:
    """统一阶段 token 定义，同时保留真实的 0。

    旧写法 ``sum(...) or None`` 会把「明确为 0」误写成未知，导致无澄清 run
    在 OpenCode 是 0、其他脚手架却是 None。
    """
    values = [part.get(k) for k in ("input", "output", "reasoning")
              if part.get(k) is not None]
    return sum(values) if values else None


def _usage_contract(payload: dict) -> dict:
    """补齐所有脚手架共享的 usage 契约与可比性元数据。

    保留历史字段不改名；新增字段只用于告诉分析端「是否采到、单位是什么、
    精度如何」。缺数据仍为 None，绝不补成 0。
    """
    out = dict(payload)
    tokens = out.get("tokens") if isinstance(out.get("tokens"), dict) else None
    phase = out.get("phase") if isinstance(out.get("phase"), dict) else None
    steps_unit = out.get("steps_unit")
    steps = tokens.get("steps") if tokens else None
    llm_calls = steps if steps_unit == "llm_call" else None
    out["llm_calls"] = llm_calls
    out.setdefault("source", None)
    out.setdefault("precision", None)
    out.setdefault("steps_unit", None)
    out.setdefault("phase", None)
    out.setdefault("phase_precision", None)
    out.setdefault("cost_precision", None)
    out.setdefault("calls", [])
    out.setdefault("effective_models", [])

    def token_available(name: str) -> bool:
        return tokens is not None and tokens.get(name) is not None

    out["availability"] = {
        "tokens": bool(tokens and any(token_available(k)
                                      for k in ("input", "output", "reasoning"))),
        "input_tokens": token_available("input"),
        "output_tokens": token_available("output"),
        "reasoning_tokens": token_available("reasoning"),
        "cache_read_tokens": token_available("cache_read"),
        "cache_write_tokens": token_available("cache_write"),
        "cache_hit_rate": out.get("cache_hit_rate") is not None,
        "cost_usd": out.get("cost_usd") is not None,
        "llm_calls": llm_calls is not None,
        "phase_tokens": bool(phase and all(
            phase.get(k) is not None for k in ("clarify_tokens", "solve_tokens"))),
        "phase_time": bool(phase and all(
            phase.get(k) is not None for k in ("clarify_secs", "solve_secs"))),
    }
    out["units"] = {
        "tokens": "token",
        "cache_hit_rate": "ratio",
        "cost_usd": "usd",
        "llm_calls": "llm_call",
        "phase_time": "second",
    }
    return out


def _backend_usage(backend_usage: dict, model: str | None, empty: dict) -> dict:
    """Convert optional backend metadata to the public usage schema.

    This is deliberately a fallback only.  ``collect_usage`` continues to use
    OpenCode's SQLite as the authority whenever a DB is present.
    """
    tokens = dict(backend_usage.get("tokens") or {})
    if not tokens or not any(tokens.get(k) is not None for k in
                             ("input", "output", "reasoning", "cache_read", "cache_write")):
        return empty
    for key in ("input", "output", "reasoning", "cache_read", "cache_write"):
        tokens.setdefault(key, None)
    tokens.setdefault("steps", None)
    inp, cached = tokens.get("input"), tokens.get("cache_read")
    hit_rate = None
    if inp is not None and cached is not None and (inp + cached):
        # 与 opencode 分支**同一个定义**：input 不含缓存命中，分母是 input + cache_read。
        # 后端账本（UsageLedger）已按各家协议把 input 归一成不含缓存
        # （OpenAI / Gemini 减掉 cached；Anthropic 本就不含），这里不再分口径。
        # 2026-09-02 之前这里是 cached / input，对 Anthropic 系（input 不含缓存）
        # 会系统性偏高，与 opencode 的数不可比。
        hit_rate = round(cached / (inp + cached), 4)

    prices, price_key = _resolve_prices(model)
    precision = backend_usage.get("precision", "native_event")
    steps_unit = backend_usage.get("steps_unit", "llm_call")
    cost = None
    if prices is not None:
        # Use call-level metadata when available, which preserves the
        # per-step tier rule used by OpenCode.  A third-party backend that only
        # supplies an aggregate still gets a transparent aggregate estimate.
        details = backend_usage.get("_steps") or []
        if details:
            cost = round(sum(_step_cost({k: step.get(k) or 0 for k in
                                         ("input", "output", "cache_read", "cache_write")}, prices)
                             for step in details), 6)
        else:
            cost = round(_step_cost({k: tokens.get(k) or 0 for k in
                                     ("input", "output", "cache_read", "cache_write")}, prices), 6)

    phase = backend_usage.get("phase")
    if phase and isinstance(phase, dict) and any(k in phase for k in ("clarify", "solve")):
        # UsageLedger keeps phase records nested; expose the same flat names as
        # the existing OpenCode event-timeline output.
        c = phase.get("clarify") or {}
        s = phase.get("solve") or {}
        phase = {
            "clarify_tokens": _phase_token_total(c),
            "solve_tokens": _phase_token_total(s),
            "clarify_secs": phase.get("clarify_secs"),
            "solve_secs": phase.get("solve_secs"),
            "clarify_input": c.get("input"), "clarify_output": c.get("output"),
            "clarify_reasoning": c.get("reasoning"),
            "solve_input": s.get("input"), "solve_output": s.get("output"),
            "solve_reasoning": s.get("reasoning"),
        }
    result = {
        "tokens": tokens,
        "cache_hit_rate": hit_rate,
        "cost_usd": cost,
        "pricing_model": price_key,
        "pricing_warning": (f"no pricing entry for model {model!r}"
                             if prices is None else None),
        "phase": phase,
        "source": backend_usage.get("source", "backend_metadata"),
        "precision": precision,
        # steps 单位：llm_call（逐模型调用）| send（整轮聚合，gemini）。跨脚手架比 steps
        # 前先看它；opencode SQLite 分支恒为 llm_call
        "steps_unit": steps_unit,
        "phase_precision": (backend_usage.get("phase_precision", "send_boundary")
                            if phase is not None else None),
        # 聚合 token 总数仍可信；只有需要逐调用判档的费用会受 send 粒度影响。
        "cost_precision": ("aggregate_estimate"
                           if cost is not None and prices and prices.get("tier_threshold")
                           and steps_unit == "send"
                           else "partial" if cost is not None and precision == "partial"
                           else "exact" if cost is not None else None),
        "calls": backend_usage.get("calls") or [],
        "effective_models": backend_usage.get("effective_models") or [],
    }
    return _usage_contract(result)


def _resolve_prices(model: str | None) -> tuple[dict, str] | tuple[None, None]:
    """按模型名查单价。

    容忍两类写法：provider 前缀（responses/kimi-k3）与变体后缀
    （gpt-5.6-sol/high、gpt-5.6-sol-high）。变体与底层模型同单价，
    差异仅在 reasoning_effort，故取最长前缀匹配。
    """
    if not model:
        return None, None
    from ..config import pricing as _pricing
    pricing = {k: v for k, v in _pricing().items() if not k.startswith("_")}

    # 逐段剥离 provider 前缀，直到命中；再退化为前缀匹配以吸收变体后缀
    parts = model.split("/")
    for i in range(len(parts)):
        candidate = "/".join(parts[i:])
        if candidate in pricing:
            return pricing[candidate], candidate
        matches = [k for k in pricing if candidate.startswith(k)]
        if matches:
            key = max(matches, key=len)
            return pricing[key], key
    return None, None


def collect_usage(run_dir: Path, model: str | None = None,
                  backend_usage: dict | None = None) -> dict:
    """汇总一次 run 的 token 用量与费用。

    字段缺失时用 None 表示"未采集到"，与 0 区分开。
    """
    empty = _usage_contract({
        "tokens": None, "cache_hit_rate": None, "cost_usd": None,
        "pricing_model": None, "pricing_warning": None,
    })

    db = _find_db(run_dir)
    if db is None:
        return _backend_usage(backend_usage, model, empty) if backend_usage else empty
    try:
        steps = _read_steps(db)
    except sqlite3.Error:
        return empty
    if not steps:
        return empty

    tot = {k: sum(s[k] for s in steps) for k in
           ("input", "output", "reasoning", "cache_read", "cache_write")}
    tot["steps"] = len(steps)

    # cache 命中率：命中 token 占全部读入 token 的比例。
    # opencode 的 input 已扣除缓存命中部分，故分母是 input + cache_read。
    read_total = tot["input"] + tot["cache_read"]
    hit_rate = round(tot["cache_read"] / read_total, 4) if read_total else None

    prices, price_key = _resolve_prices(model)
    cost = round(sum(_step_cost(s, prices) for s in steps), 6) if prices else None

    # opencode：按 question tool / 文本回灌 的时间锚点切分澄清 vs 求解阶段
    had_clarify = False
    cj = Path(run_dir) / "clarify.json"
    if cj.is_file():
        try:
            had_clarify = json.loads(cj.read_text(encoding="utf-8")).get("total_rounds", 0) > 0
        except (OSError, ValueError, json.JSONDecodeError):
            had_clarify = False
    phase = _phase_split(db, run_dir, had_clarify=had_clarify) if db else None
    # agent 在 question 工具上阻塞等 answerer 的时间混在它的墙钟里；文本通路的
    # 后端算的是子进程时长、天然不含。减掉才能跨脚手架比澄清耗时。
    if phase and phase.get("clarify_secs"):
        harness = _harness_secs(run_dir)
        if harness:
            phase["clarify_secs"] = round(max(0.0, phase["clarify_secs"] - harness), 1)
            phase["harness_secs"] = harness

    return _usage_contract({
        "tokens": tot,
        "steps_unit": "llm_call",
        "cache_hit_rate": hit_rate,
        "cost_usd": cost,
        "pricing_model": price_key,
        # 不为未知模型猜单价；显式记录原因，避免报表里的 None 被误读成 0。
        "pricing_warning": (f"no pricing entry for model {model!r}"
                            if prices is None else None),
        "source": "opencode_sqlite",
        "precision": "native_event",
        "phase_precision": "event_timeline" if phase is not None else None,
        "cost_precision": "exact" if cost is not None else None,
        # 阶段统计由 OpenCode SQLite 事件时间线计算。
        "phase": phase,
    })
