"""Offline replay of the empty unknown terminal observed in Full/076/try1.

No provider, agent solver, subprocess, or evaluated artifact is executed here.
"""
import copy
import time
import urllib.error
from pathlib import Path
from types import SimpleNamespace

import pytest

from harness.backends.opencode.serve import OpenCodeServeSession
from harness.cli import (_completed_despite_nonzero_exit,
                         _record_backend_diagnostics)


def _terminal(mid, finish="unknown", parts=None, error=None):
    return {"info": {"id": mid, "role": "assistant", "finish": finish,
                     "time": {"created": 1000, "completed": 1001}, "error": error,
                     "tokens": {"input": 0, "output": 0, "reasoning": 0}},
            "parts": parts if parts is not None else [
                {"type": "step-start"}, {"type": "step-finish", "reason": finish}]}


class TerminalServer:
    def __init__(self, replies):
        self.replies = list(replies)
        self.messages = [_terminal("older", "tool-calls", [
            {"type": "text", "text": "Let me fix:"}])]
        self.posts = []
        self.statuses = {}  # Native OpenCode omits idle sessions.
        self.after_post = lambda: None
        self.on_safety_get = lambda: None

    def __call__(self, method, path, body=None, timeout=120):
        if method == "POST" and path.endswith("/abort"):
            return True
        if method == "POST" and path.endswith("/message"):
            self.posts.append((path, copy.deepcopy(body), timeout))
            reply = self.replies.pop(0)
            if isinstance(reply, Exception):
                raise reply
            if reply:
                self.messages.append(reply)
            self.after_post()
            return reply
        if method == "GET" and path == "/session/status":
            self.on_safety_get()
            return self.statuses
        if method == "GET" and path.endswith("/message"):
            return self.messages
        raise AssertionError((method, path))


def _session(replies):
    server = TerminalServer(replies)
    session = OpenCodeServeSession(Path("/tmp/offline-terminal-recovery"),
                                  SimpleNamespace(max_runtime_s=28800,
                                                  idle_timeout_s=10800,
                                                  max_turns=None))
    session.session_id = "same-session"
    session._api = server
    watches = []
    session._watch = lambda *args: watches.append(args)
    session.on_event = lambda event: None
    return session, server, watches


def test_empty_unknown_continues_same_session_with_original_budget_and_state():
    session, server, watches = _session([
        _terminal("empty"), _terminal("done", "stop", [{"type": "text", "text": "Done."}])])
    session._run_started_at = time.time() - 100
    original_start = session._run_started_at
    session.tool_turns = 69
    session._seen_parts.add("already-counted:running")
    answerer = object()
    session._answerer = answerer
    session._answered_clarify_questions = 7
    session.clarify_rounds.append({"id": "previous-round"})

    assert session._send("original task", None) == "Done."
    assert len(server.posts) == 2
    assert [p[0] for p in server.posts] == ["/session/same-session/message"] * 2
    assert [p[1] for p in server.posts] == [
        {"parts": [{"type": "text", "text": "original task"}]},
        {"parts": [{"type": "text", "text": session._CONTINUE_MESSAGE}]}]
    assert server.posts[1][2] <= server.posts[0][2] < 28800 + 600
    assert len(watches) == 1  # Same watcher also retains answered question IDs.
    assert watches[0][-1] == original_start == session._run_started_at
    assert session.tool_turns == 69
    assert "already-counted:running" in session._seen_parts
    assert session._answerer is answerer and session._answered_clarify_questions == 7
    assert session.clarify_rounds == [{"id": "previous-round"}]
    assert session.killed_reason is None and session.degraded_reason is None
    audit = session.backend_diagnostics["terminal_recovery"]
    assert audit["outcome"] == "recovered"
    assert audit["events"][0]["message_id"] == "empty"
    assert audit["events"][0]["continuation"] == 1


def test_persistent_empty_unknown_is_bounded_and_never_returns_stale_text():
    session, server, _ = _session([_terminal(str(i)) for i in range(4)])
    assert session._send("original task", None) == ""
    assert len(server.posts) == 3
    assert session._terminal_continuations == 2
    assert session.degraded_reason == "opencode_unknown_terminal"
    audit = session.backend_diagnostics["terminal_recovery"]
    assert audit["outcome"] == "not_recovered"
    assert audit["events"][-1]["reason"] == "continuation_limit"
    # A subsequent harness send cannot reset the session-wide recovery budget.
    assert session._send("another logical turn", None) == ""
    assert len(server.posts) == 4 and session._terminal_continuations == 2


@pytest.mark.parametrize("reply", [
    _terminal("normal", "stop"),
    _terminal("normal", "stop", [{"type": "text", "text": "Done."}]),
])
def test_normal_terminal_does_not_continue_or_fetch_old_text(reply):
    session, server, _ = _session([reply])
    expected = "Done." if reply["parts"][0]["type"] == "text" else ""
    assert session._send("task", None) == expected
    assert len(server.posts) == 1 and not session.backend_diagnostics


@pytest.mark.parametrize("part", [
    {"type": "text", "text": "Partial answer"},
    {"type": "reasoning", "text": "Still thinking"},
    {"type": "tool", "state": {"status": "completed"}},
])
def test_nonempty_unknown_is_not_automatically_continued(part):
    session, server, _ = _session([_terminal("partial", parts=[part])])
    session._send("task", None)
    assert len(server.posts) == 1
    assert session.degraded_reason == "opencode_unknown_terminal"
    assert session.backend_diagnostics["terminal_recovery"]["events"][-1]["reason"] == "nonempty_unknown_turn"


@pytest.mark.parametrize("name", ["ProviderAuthError", "APIError"])
def test_message_error_is_not_success_or_retried(name):
    session, server, _ = _session([_terminal("err", error={"name": name})])
    assert session._send("task", None) == ""
    assert len(server.posts) == 1
    assert session.degraded_reason == "opencode_message_error"
    assert session.backend_diagnostics["terminal_error"]["name"] == name


def test_http_error_is_not_caught_as_timeout_or_retried():
    session, server, _ = _session([urllib.error.HTTPError("http://offline", 401, "unauthorized", {}, None)])
    with pytest.raises(RuntimeError, match="HTTP 401"):
        session._send("task", None)
    assert session.killed_reason is None
    assert session.degraded_reason == "opencode_message_http_error"
    assert len(server.posts) == 1


def test_ambiguous_transport_failure_does_not_replay_post():
    session, server, _ = _session([urllib.error.URLError("connection closed")])
    session._send("task", None)
    assert len(server.posts) == 1
    assert not session.backend_diagnostics.get("terminal_recovery")


@pytest.mark.parametrize("reason", ["early_stop", "idle", "max_runtime", "max_turns"])
def test_watchdog_or_intentional_stop_cannot_be_resumed(reason):
    session, server, _ = _session([_terminal("empty")])
    server.after_post = lambda: setattr(session, "killed_reason", reason)
    session._send("task", {"question"})
    assert len(server.posts) == 1 and session.killed_reason == reason


@pytest.mark.parametrize("state", ["busy", "retry"])
def test_active_session_is_not_resumed(state):
    session, server, _ = _session([_terminal("empty")])
    server.statuses = {session.session_id: {"type": state}}
    session._send("task", None)
    assert len(server.posts) == 1
    assert session.backend_diagnostics["terminal_recovery"]["events"][-1]["reason"] == "session_not_idle"


@pytest.mark.parametrize("state", ["running", "pending"])
def test_unfinished_earlier_tool_prevents_continuation(state):
    session, server, _ = _session([_terminal("empty")])
    server.messages[0]["parts"].append({"type": "tool", "state": {"status": state}})
    session._send("task", None)
    assert len(server.posts) == 1
    assert session.backend_diagnostics["terminal_recovery"]["events"][-1]["reason"] == "unfinished_tool"


def test_changed_latest_message_prevents_continuation():
    session, server, _ = _session([_terminal("empty")])
    server.after_post = lambda: server.messages.append({"info": {"id": "other-user", "role": "user"}})
    session._send("task", None)
    assert len(server.posts) == 1
    assert session.backend_diagnostics["terminal_recovery"]["events"][-1]["reason"] == "terminal_message_changed"


def test_failed_safety_check_does_not_continue():
    session, server, _ = _session([_terminal("empty")])
    def failed_get():
        raise urllib.error.URLError("offline")
    server.on_safety_get = failed_get
    session._send("task", None)
    assert len(server.posts) == 1 and session.degraded_reason


def test_safety_checks_cannot_extend_original_deadline(monkeypatch):
    now = [1000.0]
    monkeypatch.setattr("harness.backends.opencode.serve.time.time", lambda: now[0])
    session, server, _ = _session([_terminal("empty")])
    session._run_started_at = now[0] - 28799
    server.on_safety_get = lambda: now.__setitem__(0, now[0] + 2)
    session._send("task", None)
    assert len(server.posts) == 1 and session.killed_reason == "max_runtime"


def test_no_post_when_original_budget_is_already_exhausted():
    session, server, _ = _session([])
    session._run_started_at = time.time() - 28801
    session._send("task", None)
    assert not server.posts and session.killed_reason == "max_runtime"


def test_no_continuation_past_original_tool_limit():
    session, server, _ = _session([_terminal("empty")])
    session._config.max_turns = 70
    session.tool_turns = 69
    server.after_post = lambda: setattr(session, "tool_turns", 70)
    session._send("task", None)
    assert len(server.posts) == 1 and session.killed_reason == "max_turns"


def test_absent_http_body_is_not_success_or_stale_text():
    session, server, _ = _session([None])
    assert session._send("task", None) == ""
    assert session.degraded_reason == "opencode_empty_http_response"
    assert len(server.posts) == 1


@pytest.mark.parametrize("status", ["degraded", "error"])
def test_opencode_sources_only_do_not_rescue_abnormal_terminal(status):
    assert not _completed_despite_nonzero_exit(
        status, ["solver/main.py", "solver/common.py"], backend_reason="opencode_unknown_terminal")
    assert _completed_despite_nonzero_exit(
        status, ["solver/main.py", "solution.json"], backend_reason="opencode_unknown_terminal")
    assert _completed_despite_nonzero_exit(status, ["main.py"], backend_reason="another_backend")


def test_successful_recovery_keeps_audit_in_usage(tmp_path):
    import json
    session, _, _ = _session([_terminal("empty"), _terminal("done", "stop")])
    session._send("task", None)
    assert session.degraded_reason is None
    (tmp_path / "usage.json").write_text(json.dumps({"status": "ok", "produced": ["solution.json"]}))
    _record_backend_diagnostics(tmp_path, session)
    data = json.loads((tmp_path / "usage.json").read_text())
    assert data["status"] == "ok" and data["produced"] == ["solution.json"]
    assert data["backend_diagnostics"]["terminal_recovery"]["outcome"] == "recovered"


@pytest.mark.parametrize("payload,recovered,expected_status", [
    (False, False, "degraded"), (True, False, "ok"), (True, True, "ok"),
])
def test_cli_preserves_terminal_outcome_and_artifacts(
        tmp_path, monkeypatch, payload, recovered, expected_status):
    import asyncio
    import json
    from harness import cli
    from harness.run import RunSpec

    case = tmp_path / "076_demo_clean"
    case.mkdir()
    run = tmp_path / "run"
    ws = run / "workspace"
    ws.mkdir(parents=True)
    (run / "manifest.json").write_text(json.dumps({"case": case.name, "model": "local"}))
    replies = [_terminal("empty"), _terminal("done", "stop")] if recovered else [
        _terminal(str(i)) for i in range(3)]
    session, server, _ = _session(replies)

    async def close():
        pass
    async def start(workspace):
        assert workspace == ws
        return session
    session.close = close
    def produce():
        (ws / "solver.py").write_text("# agent source, not a materialized answer\n")
        if payload:
            (ws / "solution.json").write_text('{"answer": []}')
    server.after_post = produce
    monkeypatch.setattr(cli, "setup_run", lambda *a: (run, ws, "original task"))
    monkeypatch.setattr(cli, "runner_from_config", lambda *a: SimpleNamespace(start=start))
    monkeypatch.setattr(cli.envmod, "build_env", lambda *a: {
        k: str(tmp_path / k) for k in ("XDG_CONFIG_HOME", "XDG_DATA_HOME", "XDG_CACHE_HOME")})
    monkeypatch.setattr(cli.envmod, "describe", lambda *a: {})
    monkeypatch.setattr(cli, "collect_usage", lambda *a, **kw: None)
    spec = RunSpec(case=case, condition="Full", scaffold="opencode", model="local")
    result = asyncio.run(cli._one(spec, tmp_path, 28800, skip_eval=True, use_sandbox=False))
    usage = json.loads((run / "usage.json").read_text())
    assert result["status"] == usage["status"] == expected_status
    assert "solver.py" in usage["produced"] and (ws / "solver.py").is_file()
    assert ("solution.json" in usage["produced"]) == payload
    audit = usage["backend_diagnostics"]["terminal_recovery"]
    assert audit["outcome"] == ("recovered" if recovered else "not_recovered")
    if not recovered:
        assert usage["backend_anomaly" if payload else "failure_reason"] == "opencode_unknown_terminal"
