"""OpenHands SSE transport adapter: opt-in, atomic tools, usage, and retries."""
import asyncio
import json
import os
import subprocess
import sys
from pathlib import Path
from types import ModuleType, SimpleNamespace

import pytest

from harness.backends.base import RunnerConfig
from harness.backends.openhands import driver
from harness.backends.openhands.session import _OpenHandsSession


class ConnectionError(Exception):
    def __init__(self, message, **kwargs):
        super().__init__(message)


class Stream:
    def __init__(self, chunks, error=None):
        self.chunks, self.error, self.closed = chunks, error, False

    def __iter__(self):
        yield from self.chunks
        if self.error:
            raise self.error

    def close(self):
        self.closed = True


def install_fakes(monkeypatch, stream, response=None):
    calls, builds = [], []
    module = ModuleType("openhands.llm.llm")

    def original(*args, **kwargs):
        calls.append((args, kwargs))
        return stream

    def build(chunks, messages):
        builds.append((chunks, messages))
        return response or SimpleNamespace(choices=[object()], usage=None)

    module.litellm_completion = original
    monkeypatch.setitem(sys.modules, "openhands.llm.llm", module)
    monkeypatch.setitem(sys.modules, "litellm", SimpleNamespace(
        stream_chunk_builder=build, APIConnectionError=ConnectionError,
        Usage=lambda **values: SimpleNamespace(**values)))
    return module, original, calls, builds


def test_stream_assembled_inside_retry_boundary(monkeypatch, tmp_path):
    usage = {"prompt_tokens": 42, "completion_tokens": 12, "total_tokens": 54,
             "prompt_tokens_details": {"cached_tokens": 20},
             "completion_tokens_details": {"reasoning_tokens": 8}}
    chunks = [
        {"choices": [{"delta": {"reasoning_content": "reason"}, "finish_reason": None}]},
        {"choices": [{"delta": {"tool_calls": [{"index": 0, "function": {
            "name": "execute_bash", "arguments": '{"command":'}}]}, "finish_reason": None}]},
        {"choices": [{"delta": {"tool_calls": [{"index": 0, "function": {
            "arguments": '"true"}'}}]}, "finish_reason": "tool_calls"}]},
        {"choices": [], "usage": usage},
    ]
    stream = Stream(chunks)
    response = SimpleNamespace(choices=[object()], usage=None)
    module, original, calls, builds = install_fakes(monkeypatch, stream, response)
    stats = {}
    sidecar = tmp_path / "stream.json"
    restore = driver._install_chat_streaming_compat(stats, str(sidecar))
    messages = [{"role": "user", "content": "private task"}]
    result = module.litellm_completion(
        model="openai/qwen3.8-max", messages=messages, api_key="secret",
        timeout=3600, max_completion_tokens=131072, temperature=0.0, top_p=1.0,
        stream_options={"existing_option": True})
    assert result is response
    assert builds == [(chunks, messages)]
    sent = calls[0][1]
    assert sent["stream"] is True
    assert sent["stream_options"] == {"existing_option": True, "include_usage": True}
    assert (sent["timeout"], sent["max_completion_tokens"], sent["temperature"], sent["top_p"]) == (3600, 131072, 0.0, 1.0)
    assert vars(result.usage) == usage
    assert (stats["started"], stats["completed"], stats["failed"]) == (1, 1, 0)
    assert stats["provider_usage_received"] and stream.closed
    assert "secret" not in sidecar.read_text() and "private task" not in sidecar.read_text()
    assert json.loads(sidecar.read_text())["state"] == "completed"
    restore()
    assert module.litellm_completion is original


@pytest.mark.parametrize("chunks", [[], [{"choices": [{"delta": {"content": "partial"}, "finish_reason": None}]}]])
def test_truncated_stream_never_returns_partial_response(monkeypatch, chunks):
    stream = Stream(chunks)
    module, _, _, builds = install_fakes(monkeypatch, stream)
    stats = {}
    restore = driver._install_chat_streaming_compat(stats)
    try:
        with pytest.raises(ConnectionError, match="finish_reason"):
            module.litellm_completion(model="openai/qwen3.8-max", messages=[])
        assert builds == []
        assert stats["completed"] == 0 and stats["failed"] == 1 and stream.closed
    finally:
        restore()


def test_iteration_error_propagates_for_existing_retry(monkeypatch):
    error = TimeoutError("connection stalled")
    stream = Stream([{"choices": []}], error)
    module, _, _, builds = install_fakes(monkeypatch, stream)
    stats = {}
    restore = driver._install_chat_streaming_compat(stats)
    try:
        with pytest.raises(TimeoutError) as caught:
            module.litellm_completion(model="openai/qwen3.8-max", messages=[])
        assert caught.value is error and not builds and stream.closed
        stream.error = None
        stream.chunks = [{"choices": [{"finish_reason": "stop"}]}]
        module.litellm_completion(model="openai/qwen3.8-max", messages=[])
        assert (stats["started"], stats["completed"], stats["failed"]) == (2, 1, 1)
    finally:
        restore()


@pytest.mark.parametrize("setting, expected", [(None, False), ("false", False), ("1", True), ("true", True)])
def test_session_stream_flag_is_explicit(monkeypatch, tmp_path, setting, expected):
    if setting is None:
        monkeypatch.delenv("DELIVER_OPENHANDS_STREAM", raising=False)
    else:
        monkeypatch.setenv("DELIVER_OPENHANDS_STREAM", setting)
    sent = []

    def run(args, **kwargs):
        sent.append(json.loads(kwargs["input"]))
        return subprocess.CompletedProcess(args, 0, json.dumps({"final_message": "", "diagnostics": {}}), "")

    monkeypatch.setattr("harness.backends._proc.run_pg", run)
    monkeypatch.setattr("harness.backends._retry.gateway", lambda _: ("https://gw/v1", "key"))
    session = _OpenHandsSession(tmp_path, RunnerConfig(
        runner_type="openhands", model="chat/qwen3.8-max", max_runtime_s=60))
    session.on_event = lambda _: None
    asyncio.run(session.send("task"))
    assert sent[0]["stream"] is expected
    assert session.backend_diagnostics["stream"] is expected
    assert sent[0]["max_output_tokens"] == 131072


def test_responses_streaming_fails_before_sending(monkeypatch, tmp_path):
    monkeypatch.setenv("DELIVER_OPENHANDS_STREAM", "true")
    monkeypatch.setattr("harness.backends._retry.gateway", lambda _: ("https://gw/v1", "key"))
    session = _OpenHandsSession(tmp_path, RunnerConfig(
        runner_type="openhands", model="responses/kimi-k3", max_runtime_s=60))
    with pytest.raises(ValueError, match="Chat Completions only"):
        asyncio.run(session.send("task"))
