"""deepseek_harness 后端的纯函数部分（不起进程）。协议/多轮行为由 e2e 覆盖。"""

from __future__ import annotations

import time
from pathlib import Path

from harness.backends.deepseek_harness import _split_model, _write_home, DeepSeekHarnessSession
from harness.backends.usage import UsageLedger


def test_split_model_keeps_provider_route():
    assert _split_model("chat/deepseek-v4-pro") == ("chat", "deepseek-v4-pro")
    assert _split_model("deepseek-v4-pro") == ("chat", "deepseek-v4-pro")


def test_write_home_declares_openai_compat_route(tmp_path: Path):
    _write_home(tmp_path, "chat", "deepseek-v4-pro")
    y = (tmp_path / "settings.yaml").read_text(encoding="utf-8")
    assert "api: openai-completions" in y
    assert "apiKeyEnv:" in y and "sk-" not in y            # 只写环境变量名，不写密钥
    assert "- id: deepseek-v4-pro" in y
    assert "agent-default-model:" in y and "model: deepseek-v4-pro" in y


def test_dsh_is_text_channel_only():
    """SDK 模式下 ask_user_question 无 answerer 必报错——必须走文本多轮。"""
    assert DeepSeekHarnessSession.supports_native_clarify is False


def test_collect_text_takes_last_assistant_text_block_only():
    s = DeepSeekHarnessSession.__new__(DeepSeekHarnessSession)
    s._events = [
        {"type": "assistant/message", "data": {"message": {"content": [
            {"type": "reasoning", "text": "thinking..."},
            {"type": "text", "text": "第一段"}]}}},
        {"type": "tool/call", "data": {}},
        {"type": "assistant/message", "data": {"message": {"content": [
            {"type": "text", "text": "最终回复？"}]}}},
        {"type": "turn/end", "data": {"reason": {"kind": "completed"}}},
    ]
    s._saw_usage = False
    assert s._collect_text(0, 0.0) == "最终回复？"          # reasoning 不算、取最后一条


def _usage_session():
    s = DeepSeekHarnessSession.__new__(DeepSeekHarnessSession)
    s.on_event = lambda e: None
    s.killed_reason = None
    s.degraded_reason = None
    s._usage_ledger = UsageLedger()
    s._usage_pending = {}
    s._last_call_tokens = None
    s._saw_usage = False
    return s


def test_dsh_final_usage_replaces_streaming_sample():
    """dsh reports one attempt in chunk and message; it must be billed once."""
    s = _usage_session()
    usage = {"inputTokens": 100, "outputTokens": 20, "cacheReadTokens": 30}
    s._on_session_event({"type": "assistant/chunk", "data": {
        "turn": 1, "step": 2, "chunk": {"type": "usage", "usage": usage}}})
    s._on_session_event({"type": "assistant/message", "data": {
        "turn": 1, "step": 2, "usage": usage, "message": {"content": []}}})
    s._on_session_event({"type": "step/end", "data": {"turn": 1, "step": 2}})

    assert s.usage_snapshot()["tokens"] == {
        "input": 100, "output": 20, "reasoning": None,
        "cache_read": 30, "cache_write": None, "steps": 1,
    }


def test_dsh_chunk_usage_survives_without_final_message():
    s = _usage_session()
    s._on_session_event({"type": "assistant/chunk", "data": {
        "turn": 3, "step": 4, "chunk": {"type": "usage", "usage": {
            "inputTokens": 7, "outputTokens": 2}}}})
    s._on_session_event({"type": "turn/end", "data": {
        "turn": 3, "reason": {"kind": "error", "error": {"message": "gateway"}}}})

    assert s.usage_snapshot()["tokens"]["steps"] == 1
    assert "gateway" in s.degraded_reason


def test_dsh_max_tokens_is_agent_outcome_not_backend_failure():
    s = _usage_session()
    s._on_session_event({"type": "turn/end", "data": {
        "turn": 1, "reason": {"kind": "max-tokens"}}})
    assert s.degraded_reason is None


def test_registry_exposes_full_name_and_alias():
    from harness.backends.registry import available
    names = available()
    assert "deepseek-harness" in names and "dsh" in names


def test_runspec_rejects_deepseek_without_chat_route(tmp_path: Path):
    import pytest
    from harness.run import RunSpec, assert_deepseek_route
    with pytest.raises(ValueError, match="chat/"):
        RunSpec(case=tmp_path, condition="Hidden", model="direct/deepseek-v4-pro")
    RunSpec(case=tmp_path, condition="Hidden", model="chat/deepseek-v4-pro")
    assert_deepseek_route(None); assert_deepseek_route("responses/kimi-k3")


def test_initialize_retries_while_plugins_still_loading(monkeypatch, tmp_path):
    """进程刚起、插件未加载完时 initialize 收到 "no adapter registered"：要重试而不是当配置错。"""
    from harness.backends import _retry as R
    monkeypatch.setattr(R.time, "sleep", lambda s: None)
    s = DeepSeekHarnessSession.__new__(DeepSeekHarnessSession)
    s._workspace = tmp_path; s._provider = "chat"; s._model = "deepseek-v4-pro"
    s.on_event = lambda e: None; s.session_id = "s-x"
    calls = {"n": 0}

    def fake_request(method, params, *, timeout):
        calls["n"] += 1
        if calls["n"] < 3:
            raise RuntimeError('dsh initialize 失败: {"code": -32603, "message": "no adapter registered for provider \\"chat\\""}')
        return {"serverInfo": {"name": "deepseek-harness-sdk-runtime", "version": "0.0.1"}}
    s._request = fake_request
    # 直接调用 start() 里那段：复用同样的 retry 参数
    from harness.backends._retry import retry_transient
    res = retry_transient(lambda: s._request("initialize", {}, timeout=1), label="t", attempts=6,
                          extra_transient=("no adapter registered",))
    assert res["serverInfo"]["name"] == "deepseek-harness-sdk-runtime" and calls["n"] == 3


def test_dsh_text_turns_share_one_run_deadline():
    """An expired run does not enqueue another continuation with a fresh budget."""
    class Proc:
        returncode = None
        def poll(self): return None

    class Cfg:
        max_runtime_s = 60
        idle_timeout_s = 60

    s = DeepSeekHarnessSession.__new__(DeepSeekHarnessSession)
    s._proc = Proc(); s._config = Cfg(); s._events = []; s._saw_usage = False
    s._runtime_deadline = time.monotonic() - 1
    s.killed_reason = None
    killed = []
    s._kill = lambda: killed.append(True)
    s._request = lambda *args, **kwargs: (_ for _ in ()).throw(
        AssertionError("expired budget must not enqueue another prompt"))

    assert s._send("continue") == ""
    assert s.killed_reason == "max_runtime" and killed == [True]
