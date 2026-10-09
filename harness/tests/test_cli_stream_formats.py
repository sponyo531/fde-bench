"""三家 CLI 的 stream-json 抽取：回复文本 / usage / 工具调用。

fixtures 保留实测 CLI 流中解析所需的消息形状和用量，去掉会话环境元数据。
之前三家都是纯文本 stdout：tokens/cache/cost 全 None，
tool_calls 记成假零。
"""

from __future__ import annotations

import asyncio
import json
import subprocess
import time
import tomllib
from pathlib import Path

import pytest

from harness.backends import _cli_spec as C
from harness.backends.claude_code import SPEC as CLAUDE
from harness.backends.gemini import SPEC as GEMINI
from harness.backends.kimi import SPEC as KIMI

FIX = Path(__file__).parent / "fixtures"


class _Cfg:
    model = ""
    max_runtime_s = 60


def _run(spec, name):
    out = (FIX / f"{name}_stream.jsonl").read_text(encoding="utf-8")
    s = C.CliAgentSession(spec, Path("/tmp"), _Cfg())
    events = []
    s.on_event = events.append
    s._record_metadata(out, "", 12.3)
    return C._clean(out, spec), s, events


@pytest.mark.parametrize("spec,name", [(CLAUDE, "claude"), (GEMINI, "gemini"), (KIMI, "kimi")])
def test_reply_text_is_last_assistant_message(spec, name):
    reply, _, _ = _run(spec, name)
    assert reply == "DONE-42"


@pytest.mark.parametrize("spec,name", [(CLAUDE, "claude"), (GEMINI, "gemini"), (KIMI, "kimi")])
def test_tool_calls_are_counted_and_backend_declares_it(spec, name):
    _, s, events = _run(spec, name)
    assert s.reports_tool_calls is True
    assert sum(e.type == "tool_call" for e in events) == 1


def test_claude_usage_is_send_total_with_anthropic_cache_semantics():
    _, s, _ = _run(CLAUDE, "claude")
    tok = s.usage_snapshot()["tokens"]
    # steps 取 result.num_turns（=2 次模型调用），与其他后端同单位
    assert tok == {"input": 4, "output": 104, "reasoning": None,
                   "cache_read": 52653, "cache_write": 52757, "steps": 2}
    assert s.usage_snapshot()["steps_unit"] == "llm_call"
    assert s.effective_models() == ["claude-sonnet-5"]


def test_gemini_usage_input_excludes_cache():
    _, s, _ = _run(GEMINI, "gemini")
    tok = s.usage_snapshot()["tokens"]
    assert tok["input"] == 12464 and tok["cache_read"] == 8058 and tok["output"] == 31
    assert s.effective_models() == ["gemini-3.1-pro-preview"]
    assert s.usage_snapshot()["steps_unit"] == "send"       # 整轮聚合，拆不出逐调用
    assert s.usage_snapshot()["precision"] == "aggregate"


def test_kimi_usage_comes_from_session_wire_log(tmp_path, monkeypatch):
    """kimi stdout 无 usage；从 ~/.kimi-code/sessions/*/session_<id>/agents/main/wire.jsonl 取逐 step。"""
    sid = "session_7c896b28-b9e3-4540-86d8-5b86bd430322"
    wire = tmp_path / "sessions" / "wd_x" / sid / "agents" / "main" / "wire.jsonl"
    wire.parent.mkdir(parents=True)
    wire.write_text((FIX / "kimi_wire.jsonl").read_text(encoding="utf-8"), encoding="utf-8")
    monkeypatch.setenv("KIMI_CODE_HOME", str(tmp_path))
    out = (FIX / "kimi_stream.jsonl").read_text(encoding="utf-8")
    out += "\n" + '{"role":"meta","type":"session.resume_hint","session_id":"%s"}' % sid
    s = C.CliAgentSession(KIMI, Path("/tmp"), _Cfg()); s.on_event = lambda e: None
    s._record_metadata(out, "", 9.0)
    tok = s.usage_snapshot()["tokens"]
    assert tok["steps"] == 2 and tok["input"] > 0 and tok["cache_read"] > 0
    assert s.effective_models() == ["responses/kimi-k3"]
    # 续接轮：没有新增 usage.record 时不重复入账
    s._record_metadata(out, "", 1.0)
    assert s.usage_snapshot()["tokens"]["steps"] == 2


def test_kimi_149_usage_comes_from_new_share_dir_layout(tmp_path, monkeypatch):
    """1.49.0 改用 KIMI_SHARE_DIR，并把 wire.jsonl 移到 session 目录根。"""
    sid = "6e93409d-0669-4ec6-a0da-cf430b1bfd7b"
    wire = tmp_path / "sessions" / "workdir_hash" / sid / "wire.jsonl"
    wire.parent.mkdir(parents=True)
    wire.write_text((FIX / "kimi_wire.jsonl").read_text(encoding="utf-8"), encoding="utf-8")
    monkeypatch.setenv("KIMI_SHARE_DIR", str(tmp_path))
    monkeypatch.delenv("KIMI_CODE_HOME", raising=False)

    out = (FIX / "kimi_stream.jsonl").read_text(encoding="utf-8")
    out += "\n" + '{"role":"meta","type":"session.resume_hint","session_id":"%s"}' % sid
    s = C.CliAgentSession(KIMI, Path("/tmp"), _Cfg())
    s.on_event = lambda e: None
    s._record_metadata(out, "", 9.0)

    assert s.usage_snapshot()["tokens"]["steps"] == 2


def test_kimi_runtime_config_exports_current_and_legacy_home(monkeypatch, tmp_path):
    from harness.backends import _retry as R

    class Cfg:
        model = "responses/kimi-k3"
        max_runtime_s = 60

    seen = []
    monkeypatch.setattr(R, "gateway", lambda model="":
                        (seen.append(model) or ("https://gateway.example/v1", "secret")))
    s = C.CliAgentSession(KIMI, tmp_path, Cfg())
    env = s._credential_env()

    assert env["KIMI_SHARE_DIR"] == env["KIMI_CODE_HOME"]
    config_path = Path(env["KIMI_SHARE_DIR"]) / "config.toml"
    config = tomllib.loads(config_path.read_text(encoding="utf-8"))
    assert seen == ["responses/kimi-k3"]
    assert config["providers"]["responses"]["type"] == "openai_responses"
    assert config["models"]["responses/kimi-k3"]["provider"] == "responses"

    asyncio.run(s.close())


def test_kimi_without_session_log_stays_none(tmp_path, monkeypatch):
    monkeypatch.setenv("KIMI_CODE_HOME", str(tmp_path))     # 空目录：找不到 wire.jsonl
    _, s, _ = _run(KIMI, "kimi")
    assert s.usage_snapshot() is None          # 定位不到会话文件就不伪造


def test_claude_timeout_without_result_records_partial_lower_bound():
    """进程被掐、result 未落地：按 message.id 汇总 assistant usage 快照，标 partial。"""
    lines = [l for l in (FIX / "claude_stream.jsonl").read_text(encoding="utf-8").splitlines()
             if '"type": "result"' not in l and '"type":"result"' not in l]
    s = C.CliAgentSession(CLAUDE, Path("/tmp"), _Cfg()); s.on_event = lambda e: None
    s._record_metadata("\n".join(lines), "", 30.0)
    snap = s.usage_snapshot()
    assert snap is not None and snap["precision"] == "partial"
    assert snap["tokens"]["cache_read"] > 0


def test_first_turn_session_in_use_switches_to_resume(monkeypatch, tmp_path):
    """首轮失败后重试撞 "Session ID … already in use" → 自动切 --resume 模板重试。"""
    import subprocess
    from harness.backends import _retry as R
    monkeypatch.setattr(R.time, "sleep", lambda s: None)
    calls = []

    def fake_run_pg(cmd, **kw):
        calls.append(cmd)
        if len(calls) == 1:
            return subprocess.CompletedProcess(cmd, 1, "", "API Error: 502 Bad Gateway")
        if len(calls) == 2 and "--session-id" in cmd:
            return subprocess.CompletedProcess(cmd, 1, "", "Error: Session ID x is already in use.")
        return subprocess.CompletedProcess(cmd, 0, '{"type":"result","result":"ok","usage":{"input_tokens":1,"output_tokens":1}}\n', "")

    monkeypatch.setattr("harness.backends._proc.run_pg", fake_run_pg)
    s = C.CliAgentSession(CLAUDE, tmp_path, _Cfg()); s.on_event = lambda e: None
    assert s._send("hi") == "ok"
    assert "--session-id" in calls[0] and "--session-id" in calls[1] and "--resume" in calls[2]


def test_kimi_usage_found_without_resume_hint_in_private_home(tmp_path):
    """超时掐掉 resume_hint 时，从本 session 独占目录里直接取 wire.jsonl。"""
    s = C.CliAgentSession(KIMI, tmp_path, _Cfg()); s.on_event = lambda e: None
    s._runtime_home = tmp_path / "kh"
    wire = s._runtime_home / "sessions/wd_x/session_abc/agents/main/wire.jsonl"
    wire.parent.mkdir(parents=True)
    wire.write_text((FIX / "kimi_wire.jsonl").read_text(encoding="utf-8"), encoding="utf-8")
    partial = '{"role":"assistant","content":"I have reviewed the data and"}'      # 无 resume_hint
    s._record_metadata(partial, "", 150.0)
    assert s.usage_snapshot()["tokens"]["steps"] == 2


def test_kimi_149_uses_print_ui_and_real_continue_flag():
    """1.49.0 的 -p 只是 prompt；JSON 需 --print，续接也不能误用小写 -c。"""
    assert "--print" in KIMI.first
    assert "--output-format" in KIMI.first
    assert "--print" in KIMI.resume
    assert "--continue" in KIMI.resume
    assert "-c" not in KIMI.resume


def test_gemini_gateway_uses_isolated_explicit_api_key_auth(monkeypatch, tmp_path):
    """自定义 base URL 不能再让 Gemini CLI 推断成其校验器不接受的 gateway。"""
    from harness.backends import _retry as R

    monkeypatch.setattr(R, "gateway", lambda model="": ("https://gateway.example/v1", "secret"))
    monkeypatch.delenv("GEMINI_API_KEY", raising=False)
    monkeypatch.delenv("GOOGLE_GEMINI_BASE_URL", raising=False)
    monkeypatch.setenv("HOME", str(tmp_path / "unrelated-home"))

    s = C.CliAgentSession(GEMINI, tmp_path, _Cfg())
    env = s._credential_env()
    home = Path(env["HOME"])
    settings = json.loads((home / ".gemini" / "settings.json").read_text(encoding="utf-8"))

    assert home != Path.home()
    assert settings == {"security": {"auth": {"selectedType": "gemini-api-key"}}}
    assert env["GEMINI_API_KEY"] == "secret"
    assert env["GOOGLE_GEMINI_BASE_URL"] == "https://gateway.example"

    asyncio.run(s.close())
    assert not home.exists()


def test_kimi_cli_149_wire_status_update_and_block_content(tmp_path):
    """镜像里的 kimi-cli 1.49：wire.jsonl 用量是 StatusUpdate.payload.token_usage（snake_case），
    assistant content 可能是 think/text block 列表。2026-09-04 镜像内实测格式。"""
    s = C.CliAgentSession(KIMI, tmp_path, _Cfg()); s.on_event = lambda e: None
    s._runtime_home = tmp_path / "share"
    wire = s._runtime_home / "sessions/59fa73ff/1283f374/wire.jsonl"
    wire.parent.mkdir(parents=True)
    import json as _j
    wire.write_text("\n".join([
        _j.dumps({"timestamp": 1.0, "message": {"type": "StatusUpdate", "payload": {
            "context_tokens": 10699, "token_usage": {"input_other": 10699, "output": 71,
                                                     "input_cache_read": 0, "input_cache_creation": 0}}}}),
        _j.dumps({"timestamp": 2.0, "message": {"type": "StatusUpdate", "payload": {
            "context_tokens": 10810, "token_usage": {"input_other": 6000, "output": 16,
                                                     "input_cache_read": 4810, "input_cache_creation": 0}}}}),
    ]) + "\n", encoding="utf-8")
    out = "\n".join([
        _j.dumps({"role": "assistant", "content": [{"type": "think", "think": "plan…"}],
                  "tool_calls": [{"type": "function", "id": "Shell:0",
                                  "function": {"name": "Shell", "arguments": "{\"command\": \"echo hi\"}"}}]}),
        _j.dumps({"role": "tool", "content": [{"type": "text", "text": "hi\n"}], "tool_call_id": "Shell:0"}),
        _j.dumps({"role": "assistant", "content": "DONE-42"}),
    ])
    events = []; s.on_event = events.append
    s._record_metadata(out, "", 5.0)
    assert C._clean(out, KIMI) == "DONE-42"
    assert sum(e.type == "tool_call" for e in events) == 1
    tok = s.usage_snapshot()["tokens"]
    assert tok["steps"] == 2 and tok["input"] == 16699 and tok["cache_read"] == 4810 and tok["output"] == 87


def test_cli_send_uses_remaining_run_budget(monkeypatch, tmp_path):
    """Clarification and solve sends share one deadline instead of each getting 60s."""
    calls = []

    def fake_run_pg(cmd, **kwargs):
        calls.append(kwargs["timeout"])
        return subprocess.CompletedProcess(
            cmd, 0,
            '{"type":"result","result":"done","usage":{"input_tokens":1,"output_tokens":1}}\n',
            "",
        )

    monkeypatch.setattr("harness.backends._proc.run_pg", fake_run_pg)
    s = C.CliAgentSession(CLAUDE, tmp_path, _Cfg()); s.on_event = lambda e: None
    # Simulate time already spent by an earlier clarification send.
    s._runtime_deadline = time.monotonic() + 7
    assert s._send("continue") == "done"
    assert 0 < calls[0] <= 7


def test_kimi_expired_run_harvests_wire_usage_and_pinned_model(monkeypatch, tmp_path):
    """At the shared deadline, Kimi reads its live wire before the private home is removed."""
    class Cfg:
        model = "responses/kimi-k3"
        max_runtime_s = 60

    s = C.CliAgentSession(KIMI, tmp_path, Cfg()); s.on_event = lambda e: None
    s._runtime_home = tmp_path / "share"
    wire = s._runtime_home / "sessions/workdir/session/wire.jsonl"
    wire.parent.mkdir(parents=True)
    wire.write_text(json.dumps({
        "timestamp": 1,
        "message": {"type": "StatusUpdate", "payload": {"token_usage": {
            "input_other": 10, "output": 2, "input_cache_read": 3,
            "input_cache_creation": 0,
        }}},
    }) + "\n", encoding="utf-8")
    s.set_usage_phase("solve")
    s._runtime_deadline = time.monotonic() - 1

    def must_not_run(*args, **kwargs):
        raise AssertionError("expired budget must not launch another CLI process")

    monkeypatch.setattr("harness.backends._proc.run_pg", must_not_run)
    assert s._send("continue") == ""
    snap = s.usage_snapshot()
    assert s.killed_reason == "max_runtime"
    assert snap["tokens"]["input"] == 10 and snap["phase"]["solve"]["output"] == 2
    assert s.effective_models() == ["responses/kimi-k3"]


def test_nonzero_exit_with_completed_result_is_success(monkeypatch, tmp_path):
    """claude 跑完、产物齐、result terminal_reason=completed，进程却 exit=1：按成功处理，
    并记 usage；完整 stdout/stderr 落到 run_dir/.cli_logs/。"""
    import subprocess
    out = (FIX / "claude_stream.jsonl").read_text(encoding="utf-8")
    ws = tmp_path / "run" / "workspace"; ws.mkdir(parents=True)
    monkeypatch.setattr("harness.backends._proc.run_pg",
                        lambda cmd, **kw: subprocess.CompletedProcess(cmd, 1, out, "hook failed\n"))
    s = C.CliAgentSession(CLAUDE, ws, _Cfg()); notes = []; s.on_event = notes.append
    assert s._send("hi") == "DONE-42"
    assert s.usage_snapshot()["tokens"]["output"] == 104
    assert any("exit=1" in e.content for e in notes if e.type == "info")
    logs = sorted((tmp_path / "run" / ".cli_logs").iterdir())
    assert [p.name for p in logs] == ["claude-code_send01.stderr.txt", "claude-code_send01.stdout.jsonl"]
    assert (tmp_path / "run" / ".cli_logs" / "claude-code_send01.stderr.txt").read_text().startswith("# exit=1")


def test_nonzero_exit_without_completion_still_fails(monkeypatch, tmp_path):
    import subprocess, pytest as _pt
    from harness.backends import _retry as R
    monkeypatch.setattr(R.time, "sleep", lambda s: None)
    ws = tmp_path / "run" / "workspace"; ws.mkdir(parents=True)
    monkeypatch.setattr("harness.backends._proc.run_pg",
                        lambda cmd, **kw: subprocess.CompletedProcess(cmd, 1, "", "Error: invalid model"))
    s = C.CliAgentSession(CLAUDE, ws, _Cfg()); s.on_event = lambda e: None
    with _pt.raises(RuntimeError):
        s._send("hi")


def test_kimi_overload_on_stdout_not_masked_by_stderr_hint(monkeypatch, tmp_path):
    from harness.backends import _retry as R
    monkeypatch.setattr(R.time, "sleep", lambda _: None)
    ws = tmp_path / "run" / "workspace"
    ws.mkdir(parents=True)
    partial = json.dumps({"role": "assistant", "content": "Starting implementation"})
    out = partial + "\nError: The engine is currently overloaded, please try again later\n"
    calls = []
    def failed(cmd, **kwargs):
        calls.append(cmd)
        return subprocess.CompletedProcess(cmd, 1, out, "To resume this session: kimi -r abc\n")
    monkeypatch.setattr("harness.backends._proc.run_pg", failed)
    s = C.CliAgentSession(KIMI, ws, _Cfg())
    with pytest.raises(RuntimeError, match="currently overloaded"):
        s._send("hi")
    assert len(calls) == 3
    assert "--continue" not in calls[0]
    assert all("--continue" in cmd for cmd in calls[1:])


def test_kimi_partial_message_is_not_terminal_success():
    partial = json.dumps({"role": "assistant", "content": "Starting implementation"})
    assert not C._stream_completed(partial, KIMI)


def test_gemini_created_session_resumes_after_transport_error(monkeypatch, tmp_path):
    from harness.backends import _retry as R
    monkeypatch.setattr(R.time, "sleep", lambda _: None)
    s = C.CliAgentSession(GEMINI, tmp_path, _Cfg())
    calls = []
    def run(cmd, **kwargs):
        calls.append(cmd)
        if len(calls) == 1:
            return subprocess.CompletedProcess(cmd, 1,
                json.dumps({"type": "init", "session_id": s.session_id}),
                "TypeError: terminated")
        return subprocess.CompletedProcess(cmd, 0,
            json.dumps({"type": "result", "status": "success"}), "")
    monkeypatch.setattr("harness.backends._proc.run_pg", run)
    s._send("hi")
    assert len(calls) == 2
    assert "--session-id" in calls[0] and "--resume" in calls[1]


def test_permanent_error_is_not_retried_even_with_overload():
    from harness.backends._retry import is_transient
    assert not is_transient("invalid_api_key: currently overloaded")
