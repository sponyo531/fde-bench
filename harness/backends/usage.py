"""Backend usage metadata helpers.

The runner deliberately treats usage as optional.  A backend may expose token
metadata in a JSON event stream, while another one may expose none at all.  In
the latter case fields remain ``None`` instead of being reported as zero.
"""

from __future__ import annotations

from collections.abc import Mapping


# 各脚手架 usage 对象的键名。dsh（pi-ai TokenUsage）是 camelCase + Tokens 后缀
# （inputTokens / cacheReadTokens…）；kimi 的 wire.jsonl usage.record 是
# inputOther / output / inputCacheRead / inputCacheCreation；codex rollout 是
# cached_input_tokens / cache_write_input_tokens。2026-09-02 之前这些都不在表里，
# 对应后端的 tokens 或缓存命中恒 None。
_KEYS = {
    "input": ("input", "input_tokens", "prompt_tokens", "promptTokenCount",
              "prompt_token_count", "inputTokens", "inputOther", "input_other"),
    "output": ("output", "output_tokens", "completion_tokens",
               "candidatesTokenCount", "candidates_token_count", "outputTokens"),
    "reasoning": ("reasoning", "reasoning_tokens", "reasoningTokenCount",
                  "reasoningTokens", "thoughtsTokenCount", "reasoning_output_tokens"),
    "cache_read": ("cache_read", "cache_read_tokens", "cacheRead",
                   "cache_read_input_tokens", "cached_tokens", "cacheReadTokens",
                   "cachedContentTokenCount", "cached", "cached_input_tokens",
                   "inputCacheRead", "input_cache_read"),
    "cache_write": ("cache_write", "cache_write_tokens", "cacheWrite",
                    "cache_creation_input_tokens", "cacheWriteTokens",
                    "cache_write_input_tokens", "inputCacheCreation", "input_cache_creation"),
}


def canonical_model_identity(value: str | None) -> str:
    """Return the comparable model id, independent of transport/provider names.

    The tested model is configured as e.g. ``direct/glm-5.3``.  Dedicated
    scaffolds report the same call using their transport namespace instead:
    Claude/Gemini/Codex usually emit a bare id, while LiteLLM/OpenHands emits
    ``openai/glm-5.3``.  Comparing those raw strings produced false
    ``model_mismatch`` flags even though the model id was unchanged.

    Keep the raw identities in the manifest for auditability; this function is
    only for the boolean comparison.  The final path component is sufficient
    for the model ids supported by this benchmark and still detects real
    fallbacks such as ``opencode/big-pickle``.
    """
    if not isinstance(value, str):
        return ""
    parts = [part.strip().lower() for part in value.split("/") if part.strip()]
    return parts[-1] if parts else ""


def models_mismatch(requested: str | None, observed: list[str] | tuple[str, ...]) -> bool:
    """Whether observed model ids differ from the requested model.

    An empty observation remains "unknown", not a mismatch.  Callers retain
    the raw observed list separately so provider/transport routing is never
    erased from the audit trail.
    """
    observed_ids = {canonical_model_identity(model) for model in observed}
    observed_ids.discard("")
    requested_id = canonical_model_identity(requested)
    return bool(observed_ids and requested_id and observed_ids != {requested_id})


def _number(value):
    if isinstance(value, bool):
        return None
    try:
        return int(value) if value is not None else None
    except (TypeError, ValueError):
        return None


def extract_usage(value) -> dict | None:
    """Find and normalize a usage object in a backend JSON payload.

    CLIs differ in their event envelopes (``usage``, ``usageMetadata``,
    ``response.usage`` and so on), so this intentionally walks mappings but
    only accepts objects containing at least one known token key.
    """
    if isinstance(value, Mapping):
        for name in ("usage", "usageMetadata", "token_usage", "tokenUsage", "tokens"):
            candidate = value.get(name)
            if isinstance(candidate, Mapping):
                got = normalize_usage(candidate)
                if got is not None:
                    return got
        got = normalize_usage(value)
        if got is not None:
            return got
        for child in value.values():
            got = extract_usage(child)
            if got is not None:
                return got
    elif isinstance(value, list):
        for child in value:
            got = extract_usage(child)
            if got is not None:
                return got
    return None


def normalize_usage(value: Mapping) -> dict | None:
    out = {}
    for field, names in _KEYS.items():
        found = None
        for name in names:
            if name in value:
                found = _number(value[name])
                if found is not None:
                    break
        out[field] = found
    # OpenAI sometimes reports cached tokens in prompt_tokens_details.
    details = value.get("prompt_tokens_details") or value.get("inputTokenDetails")
    if isinstance(details, Mapping) and out["cache_read"] is None:
        out["cache_read"] = _number(details.get("cached_tokens") or details.get("cacheRead"))
    if not any(v is not None for v in out.values()):
        return None
    return out


def extract_model(value) -> str | None:
    """Extract an observed model identity from a structured event."""
    if isinstance(value, Mapping):
        for key in ("effective_model", "model", "model_id", "modelID", "model_name"):
            val = value.get(key)
            if isinstance(val, str) and val.strip():
                provider = value.get("provider") or value.get("provider_id") or value.get("providerID")
                return f"{provider}/{val}" if provider and "/" not in val else val
        for child in value.values():
            got = extract_model(child)
            if got:
                return got
    elif isinstance(value, list):
        for child in value:
            got = extract_model(child)
            if got:
                return got
    return None


def structured_event(value) -> dict | None:
    """Normalize the small common subset of CLI JSONL events."""
    if not isinstance(value, Mapping):
        return None
    kind = str(value.get("type") or value.get("event") or value.get("kind") or "").lower()
    item = value.get("item") if isinstance(value.get("item"), Mapping) else value
    item_kind = str(item.get("type") or item.get("kind") or "").lower()
    combined = f"{kind} {item_kind}"
    if any(x in combined for x in ("tool_result", "tool.completed", "command_result")):
        tool = item.get("name") or item.get("tool") or item.get("type") or "unknown"
        return {"type": "tool_result", "tool": str(tool),
                "is_error": bool(item.get("is_error") or item.get("error"))}
    if any(x in combined for x in ("tool", "command_execution", "file_change", "function_call")):
        tool = item.get("name") or item.get("tool") or item.get("type") or "unknown"
        return {"type": "tool_call", "tool": str(tool), "is_error": False}
    if kind == "error" or "error" in combined or item.get("is_error") is True:
        return {"type": "tool_result", "tool": str(item.get("tool") or ""), "is_error": True}
    if any(x in combined for x in ("message", "text", "response")) and any(
            item.get(k) for k in ("text", "content", "message")):
        return {"type": "text", "tool": "", "is_error": False}
    return None


class UsageLedger:
    """Accumulate per-call usage without inventing missing fields."""

    def __init__(self, *, input_includes_cache_read: bool = False):
        self._calls: list[dict] = []
        self._phase: str | None = None
        # 归一口径：账本里的 input **永远不含** cache_read（与 opencode SQLite 一致）。
        # OpenAI / Gemini 的 prompt_tokens 含缓存命中，入账时减掉；Anthropic 的
        # input_tokens 本来就不含。这样 cache_hit_rate 全框架只有一个定义。
        self._input_includes_cache_read = input_includes_cache_read
        self._phase_override: dict | None = None

    def set_phase(self, phase: str | None) -> None:
        self._phase = phase

    def relabel_last(self, phase: str | None) -> None:
        if self._calls:
            self._calls[-1]["phase"] = phase

    def record(self, usage: Mapping | None, *, phase: str | None = None,
               elapsed_s: float | None = None, partial: bool = False,
               n_calls: int = 1, aggregate: bool = False) -> None:
        """入账一条 usage。

        n_calls：这条记录代表几次底层模型调用（claude 的 result.usage 是一次 send 的
        总量，但 result.num_turns 给了调用数；记进去 steps 就仍按"模型调用"计）。
        aggregate：这条是 send 级聚合、拆不出逐调用（gemini）——snapshot 的
        steps_unit 会标成 "send"，分档定价按聚合判档会偏高，报表据此标注。
        """
        normalized = normalize_usage(usage) if usage is not None else None
        if normalized is None:
            return
        if (self._input_includes_cache_read and normalized.get("input") is not None
                and normalized.get("cache_read")):
            normalized["input"] = max(0, normalized["input"] - normalized["cache_read"])
        self._calls.append({"usage": normalized, "phase": phase or self._phase,
                            "elapsed_s": elapsed_s, "partial": partial,
                            "n_calls": max(1, int(n_calls)), "aggregate": aggregate})

    def set_phase_override(self, phase: dict | None) -> None:
        """后端按自己的锚点算好的阶段切分（如 OpenHands 一次 send 含全程，按"最后
        一次作答时已发生的调用数"切）。非空时 snapshot 的 phase 用它，不用逐条标签。"""
        self._phase_override = phase

    def label_calls(self, n_clarify: int) -> None:
        """把前 n_clarify 次调用标为 clarify、其余 solve（按调用顺序）。"""
        for i, c in enumerate(self._calls):
            c["phase"] = "clarify" if i < n_clarify else "solve"

    @property
    def call_count(self) -> int:
        return len(self._calls)

    def label_from(self, start: int, phase: str) -> None:
        """把第 start 条起的全部调用标成 phase。文本多轮用：一次 send 可能含很多步
        （dsh 一次求解 94 步），标签要按"这轮回复是不是提问"整段打，不能只改最后一条。"""
        for c in self._calls[start:]:
            c["phase"] = phase

    @property
    def last_call_tokens(self) -> int | None:
        if not self._calls:
            return None
        u = self._calls[-1]["usage"]
        vals = [u[k] for k in ("input", "output", "reasoning") if u[k] is not None]
        return sum(vals) if vals else None

    def _sum(self, calls: list[dict]) -> dict:
        result = {}
        for key in ("input", "output", "reasoning", "cache_read", "cache_write"):
            vals = [c["usage"][key] for c in calls if c["usage"][key] is not None]
            result[key] = sum(vals) if vals else None
        result["steps"] = sum(c.get("n_calls", 1) for c in calls)
        return result

    def snapshot(self) -> dict | None:
        if not self._calls:
            return None
        tokens = self._sum(self._calls)
        phases = {}
        for phase in ("clarify", "solve"):
            selected = [c for c in self._calls if c.get("phase") == phase]
            if not selected:
                continue
            pu = self._sum(selected)
            phases[phase] = pu
            elapsed = [c["elapsed_s"] for c in selected if c.get("elapsed_s") is not None]
            if elapsed:
                phases[phase + "_secs"] = round(sum(elapsed), 1)
        # Keep call-level details in the in-memory snapshot so tiered pricing
        # can be calculated exactly.  The scoring layer consumes and removes
        # this private field; it is not written to usage.json.
        # 任一条是超时后的下界估计，整份就标 partial：报表按 precision 过滤/标注
        # ``aggregate`` 不是数据不可信，而是 CLI 只给 send 级总量、无法恢复逐调用
        # 明细（当前是 Gemini）。必须与逐调用 native_event 分开，否则分档费用和
        # llm_calls 会看起来具有并不存在的精度。超时下界 ``partial`` 优先级最高。
        precision = (
            "partial" if any(c.get("partial") for c in self._calls)
            else "aggregate" if any(c.get("aggregate") for c in self._calls)
            else "native_event"
        )
        # steps 的单位：全部记录都能落到模型调用 → llm_call；任一条是 send 级聚合 → send
        steps_unit = "send" if any(c.get("aggregate") for c in self._calls) else "llm_call"
        override = getattr(self, "_phase_override", None)
        if override:
            phases = dict(override)
        return {"tokens": tokens, "phase": phases or None,
                "steps_unit": steps_unit,
                "source": "backend_metadata", "precision": precision,
                "calls": [{"usage": dict(c["usage"]), "phase": c.get("phase"),
                           "elapsed_s": c.get("elapsed_s")} for c in self._calls],
                "_steps": [dict(c["usage"]) for c in self._calls]}


# ── 工具调用的跨脚手架归一 ──────────────────────────────────────────────────
# 各家工具集不同（codex 没有 read 工具、读文件靠 shell；claude 有 Read/Edit/Bash；
# opencode 有 read/write/edit/bash/glob/grep；dsh 有 bash/read/write/edit…），
# 原始 tool_calls 只能在同一脚手架内比。这里按"干了什么"归成五类，
# shell / edit 两类在所有脚手架都有明确对应，可跨脚手架比。
_TOOL_KINDS: dict[str, tuple[str, ...]] = {
    # 先判 other：todo / 计划 / 思考 / 提问 / 子 agent 这类"不碰文件系统"的工具，
    # 名字里常带 write/plan，不先拦会被兜底规则归成 edit
    "other": ("todowrite", "todoread", "write_todos", "task_tracker", "task", "agent", "subagent",
              "think", "question", "ask_user_question", "askuserquestion", "ask_user",
              "plan", "enter_plan_mode", "exit_plan_mode", "update_plan", "plan_execution",
              "finish", "skill", "activate_skill", "complete_task", "goal", "workflow",
              "todo", "browser", "topic"),
    "shell": ("bash", "shell", "execute_bash", "run_shell_command", "command_execution",
              "local_shell", "exec_command", "execute_ipython_cell", "python", "pwsh",
              "container.exec", "openhands-action", "cmdrunaction", "ipythonruncellaction"),
    "read": ("read", "read_file", "read_many_files", "cat", "view", "str_replace_editor:view",
             "notebookread", "list_directory", "ls", "filereadaction"),
    "edit": ("edit", "write", "write_file", "apply_patch", "file_change", "str_replace_editor",
             "multiedit", "notebookedit", "create_file", "replace", "patch",
             "fileeditaction", "filewriteaction"),
    "search": ("grep", "glob", "grep_search", "glob_files", "search_file_content", "find",
               "web_fetch", "webfetch", "web_search", "websearch", "google_web_search",
               "fetch", "search", "browseurlaction", "browseinteractiveaction",
               "webbrowseaction"),
}


def classify_tool(name: str) -> str:
    """工具名 → shell | read | edit | search | other（大小写不敏感，去掉 mcp 前缀）。"""
    n = (name or "").strip().lower()
    if n.startswith("mcp__"):
        n = n.split("__")[-1]
    for kind, names in _TOOL_KINDS.items():
        if n in names:
            return kind
    # 常见前缀/包含式兜底
    if any(x in n for x in ("bash", "shell", "command", "exec", "terminal")):
        return "shell"
    if any(x in n for x in ("edit", "write", "patch", "create")):
        return "edit"
    if any(x in n for x in ("read", "view", "list", "cat")):
        return "read"
    if any(x in n for x in ("grep", "glob", "search", "fetch", "find")):
        return "search"
    return "other"


def count_tool_kinds(events) -> dict[str, int]:
    """AgentEvent 列表 → 各类工具调用次数。"""
    out = {"shell": 0, "read": 0, "edit": 0, "search": 0, "other": 0}
    for e in events or ():
        if getattr(e, "type", "") == "tool_call":
            out[classify_tool(getattr(e, "tool", ""))] += 1
    return out
