"""六脚手架 usage 语义对齐：同一份「2 次模型调用、有缓存命中」的事实，经各家原生格式
进账本后，必须得到**同一 schema、同一口径**的 snapshot。

    - tokens 五键齐全；input 不含 cache_read（各家协议已归一）
    - steps 按模型调用数；拆不出的（gemini）steps_unit=send 而不是假装 llm_call
    - precision 标签：native_event / aggregate / partial
    - tool_calls 归类后 shell / edit 可跨脚手架比

这些是报表跨脚手架比较的前提；单元测试只测各家自己时看不出口径漂移。
"""

from __future__ import annotations

from pathlib import Path

import pytest

from harness.backends import _cli_spec as C
from harness.backends.base import AgentEvent, AgentSession
from harness.backends.claude_code import SPEC as CLAUDE
from harness.backends.gemini import SPEC as GEMINI
from harness.backends.kimi import SPEC as KIMI
from harness.backends.usage import UsageLedger, count_tool_kinds, models_mismatch

FIX = Path(__file__).parent / "fixtures"
KEYS = {"input", "output", "reasoning", "cache_read", "cache_write", "steps"}


class _Cfg:
    model = ""
    max_runtime_s = 60


def _cli(spec, name, extra=""):
    out = (FIX / f"{name}_stream.jsonl").read_text(encoding="utf-8") + extra
    s = C.CliAgentSession(spec, Path("/tmp"), _Cfg()); s.on_event = lambda e: None
    s._record_metadata(out, "", 10.0)
    return s.usage_snapshot()


def _fact_two_calls(ledger: UsageLedger, *, style: str):
    """同一事实的四种原生写法：两次调用，第二次命中 40 缓存。"""
    if style == "opencode_like":          # input 不含缓存（opencode SQLite / Anthropic / kimi / dsh）
        ledger.record({"input": 100, "output": 5, "cache_read": 0})
        ledger.record({"input": 60, "output": 5, "cache_read": 40})
    elif style == "openai_like":          # prompt_tokens 含 cached（codex / OpenHands via litellm）
        ledger.record({"prompt_tokens": 100, "completion_tokens": 5, "cached_tokens": 0})
        ledger.record({"prompt_tokens": 100, "completion_tokens": 5, "cached_tokens": 40})
    elif style == "dsh_like":
        ledger.record({"inputTokens": 100, "outputTokens": 5, "cacheReadTokens": 0})
        ledger.record({"inputTokens": 60, "outputTokens": 5, "cacheReadTokens": 40})
    elif style == "kimi_like":
        ledger.record({"inputOther": 100, "output": 5, "inputCacheRead": 0})
        ledger.record({"inputOther": 60, "output": 5, "inputCacheRead": 40})


def test_agent_send_counter_is_separate_from_llm_steps():
    class Dummy(AgentSession):
        async def send(self, message: str, **kwargs) -> str:
            self._note_agent_send()
            return message

        async def close(self) -> None:
            return None

    import asyncio
    session = Dummy()
    asyncio.run(session.send("first"))
    asyncio.run(session.send("second"))
    assert session.agent_send_count() == 2


@pytest.mark.parametrize("style,inclusive", [
    ("opencode_like", False), ("openai_like", True), ("dsh_like", False), ("kimi_like", False)])
def test_same_fact_same_numbers_across_protocols(style, inclusive):
    led = UsageLedger(input_includes_cache_read=inclusive)
    _fact_two_calls(led, style=style)
    snap = led.snapshot()
    tok = snap["tokens"]
    assert set(tok) == KEYS
    assert (tok["input"], tok["output"], tok["cache_read"], tok["steps"]) == (160, 10, 40, 2)
    assert snap["steps_unit"] == "llm_call" and snap["precision"] == "native_event"


def test_real_cli_fixtures_share_schema_and_declare_granularity(tmp_path, monkeypatch):
    claude = _cli(CLAUDE, "claude")
    gemini = _cli(GEMINI, "gemini")
    sid = "session_7c896b28-b9e3-4540-86d8-5b86bd430322"
    wire = tmp_path / "sessions/wd_x" / sid / "agents/main/wire.jsonl"
    wire.parent.mkdir(parents=True)
    wire.write_text((FIX / "kimi_wire.jsonl").read_text(encoding="utf-8"), encoding="utf-8")
    monkeypatch.setenv("KIMI_CODE_HOME", str(tmp_path))
    kimi = _cli(KIMI, "kimi", extra='\n{"role":"meta","type":"session.resume_hint","session_id":"%s"}' % sid)

    for name, snap in (("claude", claude), ("gemini", gemini), ("kimi", kimi)):
        assert snap is not None, name
        assert set(snap["tokens"]) == KEYS, name
        assert {"steps_unit", "precision", "source", "phase"} <= set(snap), name
        # input 不含缓存：cache_hit = cache_read/(input+cache_read) 才有意义
        t = snap["tokens"]
        assert t["cache_read"] is None or t["input"] >= 0
    assert claude["steps_unit"] == "llm_call" and claude["tokens"]["steps"] == 2
    assert gemini["steps_unit"] == "send"                    # 拆不出逐调用，必须如实标
    assert gemini["precision"] == "aggregate"
    assert kimi["steps_unit"] == "llm_call" and kimi["tokens"]["steps"] == 2


def test_tool_calls_normalize_to_comparable_kinds():
    """各家工具名不同，归类后 shell / edit 计数相等。"""
    per_scaffold = {
        "claude-code": ["Bash", "Read", "Edit", "TodoWrite"],
        "opencode":    ["bash", "read", "edit", "todowrite"],
        "codex":       ["command_execution", "command_execution", "file_change", "update_plan"],
        "gemini":      ["run_shell_command", "read_file", "write_file", "write_todos"],
        "dsh":         ["bash", "read", "write", "task"],
        "openhands":   ["CmdRunAction", "FileReadAction", "FileEditAction", "BrowseURLAction"],
    }
    kinds = {k: count_tool_kinds([AgentEvent(type="tool_call", tool=t) for t in v])
             for k, v in per_scaffold.items()}
    for k in ("claude-code", "opencode", "gemini", "dsh"):
        assert kinds[k]["shell"] == 1 and kinds[k]["edit"] == 1 and kinds[k]["other"] == 1, (k, kinds[k])
    assert kinds["codex"]["shell"] == 2 and kinds["codex"]["edit"] == 1   # codex 读文件靠 shell
    assert kinds["openhands"] == {"shell": 1, "read": 1, "edit": 1,
                                    "search": 1, "other": 0}
    assert set(kinds["claude-code"]) == {"shell", "read", "edit", "search", "other"}


def test_phase_dict_of_nones_is_not_available():
    from harness.scoring.report import load_run
    import tempfile, json as _j
    with tempfile.TemporaryDirectory() as td:
        rd = Path(td)
        (rd / "manifest.json").write_text(_j.dumps({"run_id": "r", "case": "c", "condition": "Hidden",
                                                    "scaffold": "opencode", "model": "m"}))
        (rd / "usage.json").write_text(_j.dumps({"status": "ok", "agent_sends": 3,
                                                  "token_usage": {
            "tokens": {"input": 1, "output": 1, "reasoning": None, "cache_read": None,
                       "cache_write": None, "steps": 1},
            "steps_unit": "llm_call",
            "phase": {"clarify_secs": None, "solve_secs": None,
                      "clarify_tokens": None, "solve_tokens": None}}}))
        row = load_run(rd)
    assert row["phase_available"] is False
    assert row["tokens_total"] == 2
    assert row["agent_sends"] == 3 and row["llm_calls"] == 1


@pytest.mark.parametrize("requested,observed", [
    ("responses/gpt-5.6-sol", ["gpt-5.6-sol"]),
    ("chat/gemini-3.1-pro-preview", ["gemini-3.1-pro-preview"]),
    ("direct/glm-5.3", ["openai/glm-5.3"]),
    ("chat/deepseek-v4-pro", ["deepseek-v4-pro"]),
    ("responses/kimi-k3", []),                         # unknown is not mismatch
])
def test_model_mismatch_ignores_transport_namespace(requested, observed):
    assert models_mismatch(requested, observed) is False


def test_model_mismatch_still_detects_real_fallback():
    assert models_mismatch("direct/glm-5.3", ["opencode/big-pickle"]) is True
    assert models_mismatch("direct/glm-5.3", ["openai/glm-5.3", "openai/other-model"]) is True
