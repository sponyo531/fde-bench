"""codex 逐 step 用量来自 rollout 的 token_count，续接轮只记新增。"""

from __future__ import annotations

import json
import asyncio
import time
from pathlib import Path

from harness.backends.codex.session import CodexSession


def _tc(inp, cached, out):
    return json.dumps({"type": "event_msg", "payload": {"type": "token_count", "info": {
        "last_token_usage": {"input_tokens": inp, "cached_input_tokens": cached,
                             "output_tokens": out, "reasoning_output_tokens": 0}}}})


def test_rollout_steps_are_recorded_per_call_and_cursor_advances(tmp_path: Path, monkeypatch):
    tid = "01a0-thread"
    s = CodexSession(tmp_path, None)
    s.session_id = tid
    # rollout 落在 session 自己的独立 CODEX_HOME 里（不再读运行者的 ~/.codex）
    f = s._codex_home / "sessions/2026/09/02" / f"rollout-2026-09-02T00-00-00-{tid}.jsonl"
    f.parent.mkdir(parents=True)
    f.write_text("\n".join([_tc(1000, 0, 10), _tc(3000, 1000, 20)]) + "\n", encoding="utf-8")
    assert s._record_rollout_steps(5.0) is True
    tok = s.usage_snapshot()["tokens"]
    assert tok["steps"] == 2
    # OpenAI 口径 input 含 cached；账本归一后 input 不含：1000 + (3000-1000)
    assert tok["input"] == 3000 and tok["cache_read"] == 1000 and tok["output"] == 30

    # 第二轮 send 后 rollout 追加了一条，只记新增那条
    with f.open("a", encoding="utf-8") as fh:
        fh.write(_tc(500, 0, 5) + "\n")
    assert s._record_rollout_steps(2.0) is True
    assert s.usage_snapshot()["tokens"]["steps"] == 3
    assert s._record_rollout_steps(1.0) is False        # 没有新增 → 让调用方退回 stdout 总量


def test_missing_rollout_falls_back(tmp_path: Path, monkeypatch):
    monkeypatch.setenv("CODEX_HOME", str(tmp_path))
    s = CodexSession(tmp_path, None)
    s.session_id = "nope"
    assert s._record_rollout_steps(1.0) is False


def test_codex_home_points_provider_at_gateway(tmp_path):
    s = CodexSession(tmp_path, None)
    cfg = (s._codex_home / "config.toml").read_text(encoding="utf-8")
    assert 'model_provider = "benchmark_responses"' in cfg and 'wire_api = "responses"' in cfg
    assert 'base_url = "' in cfg
    assert 'env_key = "DELIVER_RESPONSES_API_KEY"' in cfg and "sk-" not in cfg
    assert 'web_search = "disabled"' in cfg
    assert s._env()["CODEX_HOME"] == str(s._codex_home)
    # Keep CODEX_HOME beside this run rather than in the process /tmp root;
    # Codex CLI creates PATH helper binaries under this directory.
    assert s._codex_home.parent == tmp_path.parent
    assert s._codex_home.name.startswith(".codex_home_")


def test_timeout_harvests_fresh_rollout_before_home_cleanup(tmp_path, monkeypatch):
    """A solve timeout keeps rollout usage instead of retaining clarification only."""
    class Cfg:
        model = "responses/gpt-5.6-sol"
        max_runtime_s = 60

    tid = "01a0-timeout-thread"
    s = CodexSession(tmp_path, Cfg()); s.session_id = tid
    f = s._codex_home / "sessions/2026/09/04" / f"rollout-x-{tid}.jsonl"
    f.parent.mkdir(parents=True)
    f.write_text(_tc(100, 20, 5) + "\n", encoding="utf-8")
    s.set_usage_phase("solve")
    s._runtime_deadline = time.monotonic() - 1

    def must_not_run(*args, **kwargs):
        raise AssertionError("expired budget must not launch another CLI process")

    monkeypatch.setattr("harness.backends._proc.run_pg", must_not_run)
    assert s._send("continue") == ""
    snap = s.usage_snapshot()
    assert s.killed_reason == "max_runtime"
    assert snap["tokens"]["input"] == 80 and snap["phase"]["solve"]["output"] == 5
    asyncio.run(s.close())
