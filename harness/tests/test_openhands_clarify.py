"""OpenHands 的澄清 IPC 循环。

OpenHands 与其他脚手架结构不同：run_controller 一次跑到底，澄清走它原生的
fake_user_response_fn 回调。而回调在**独立 venv 的子进程**里（rich 版本冲突），
拿不到父进程的 answerer，只能用文件做 IPC。

这条链路一旦错位就是死锁（driver 等 .ans、父进程等错文件名），且没有报错，
只能靠超时兜底——必须有测试。

    python3 -m harness.tests.test_openhands_clarify
"""

from __future__ import annotations

import sys
import tempfile
import threading
import time
from types import SimpleNamespace
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

import harness.clarify.detect as ask_detect                       # noqa: E402
from harness.backends.openhands import _OpenHandsSession  # noqa: E402
from harness.backends.openhands import driver as oh_driver  # noqa: E402


class FakeAnswerer:
    def __init__(self):
        self.asked: list[str] = []

    def answer(self, q):
        self.asked.append(q)
        return f"答:{q[:12]}"


class Cfg:
    model = "chat/glm-5.2"
    max_runtime_s = 60


def _session():
    ws = Path(tempfile.mkdtemp())
    s = _OpenHandsSession(ws, Cfg())
    s.on_event = lambda e: None
    return s, ws


def test_path_protocol_matches_driver():
    """父进程 glob 的名字必须与 driver 写的一致，否则死锁。"""
    s, ws = _session()
    s.set_answerer(FakeAnswerer())
    base = s.clarify_answers
    ask = base.with_suffix(".0.ask")                    # driver 侧写法
    assert ask.match(base.name + ".*.ask"), ask         # 父进程侧 glob
    assert ask.with_suffix(".ans") == base.with_suffix(".0.ans")
    print("✓ IPC 路径协议两侧一致")


def test_serves_question_and_records_round():
    """.ask 出现 → 调 answerer → 写 .ans，并记进 clarify_rounds。"""
    ask_detect.detect = lambda t, **kw: ["速度多少？"] if "速度" in t else []
    s, ws = _session()
    ans = FakeAnswerer()
    s.set_answerer(ans)

    stop = threading.Event()
    th = threading.Thread(target=s._serve_clarify, args=(stop,), daemon=True)
    th.start()
    try:
        ask = s.clarify_answers.with_suffix(".0.ask")
        ask.write_text("请问速度多少？", encoding="utf-8")
        reply_path = s.clarify_answers.with_suffix(".0.ans")
        for _ in range(100):
            if reply_path.exists():
                break
            time.sleep(0.05)
        assert reply_path.exists(), "未写出 .ans（driver 会一直等到超时）"
        assert "答:速度多少？" in reply_path.read_text(encoding="utf-8")
        assert ans.asked == ["速度多少？"], ans.asked
        assert len(s.clarify_rounds) == 1
    finally:
        stop.set(); th.join(timeout=2)
    print("✓ 提问→作答→回写，轮次已记录")


def test_non_question_gets_proceed():
    """agent 只是汇报进度时，不该当成提问作答。"""
    ask_detect.detect = lambda t, **kw: []
    s, ws = _session()
    ans = FakeAnswerer()
    s.set_answerer(ans)

    stop = threading.Event()
    th = threading.Thread(target=s._serve_clarify, args=(stop,), daemon=True)
    th.start()
    try:
        s.clarify_answers.with_suffix(".0.ask").write_text("我先看看数据。", encoding="utf-8")
        rp = s.clarify_answers.with_suffix(".0.ans")
        for _ in range(100):
            if rp.exists():
                break
            time.sleep(0.05)
        assert rp.exists()
        assert "carry out the work" in rp.read_text(encoding="utf-8")
        assert ans.asked == [], ans.asked
        assert s.clarify_rounds == []
    finally:
        stop.set(); th.join(timeout=2)
    print("✓ 非提问 → 推它执行，不消耗澄清轮次")


def test_round_limit():
    """撞上限后不再作答，返回固定的「无更多信息」。"""
    ask_detect.detect = lambda t, **kw: ["问题？"]
    s, ws = _session()
    ans = FakeAnswerer()
    s.set_answerer(ans, max_rounds=1)

    stop = threading.Event()
    th = threading.Thread(target=s._serve_clarify, args=(stop,), daemon=True)
    th.start()
    try:
        for i in (0, 1):
            s.clarify_answers.with_suffix(f".{i}.ask").write_text("问题？", encoding="utf-8")
            rp = s.clarify_answers.with_suffix(f".{i}.ans")
            for _ in range(100):
                if rp.exists():
                    break
                time.sleep(0.05)
        assert len(s.clarify_rounds) == 1, s.clarify_rounds
        assert s.hit_round_limit is True
        assert "No further information" in \
            s.clarify_answers.with_suffix(".1.ans").read_text(encoding="utf-8")
    finally:
        stop.set(); th.join(timeout=2)
    print("✓ 轮数上限生效")


def test_question_limit_caps_individual_answers():
    """一轮多个问题也必须受 max_questions 总条数约束。"""
    ask_detect.detect = lambda t, **kw: ["q1?", "q2?", "q3?"]
    s, ws = _session()
    ans = FakeAnswerer()
    s.set_answerer(ans, max_questions=2)

    stop = threading.Event()
    th = threading.Thread(target=s._serve_clarify, args=(stop,), daemon=True)
    th.start()
    try:
        s.clarify_answers.with_suffix(".0.ask").write_text("three", encoding="utf-8")
        rp = s.clarify_answers.with_suffix(".0.ans")
        for _ in range(100):
            if rp.exists():
                break
            time.sleep(0.05)
        assert ans.asked == ["q1?", "q2?"]
        assert s.clarify_rounds[0]["dropped_over_budget"] == 1
        assert s.hit_question_limit is True
        assert "No further information" in rp.read_text(encoding="utf-8")
    finally:
        stop.set(); th.join(timeout=2)


def test_declares_native_clarify():
    """OpenHands 走原生通路（回调），不是文本多轮。"""
    s, _ = _session()
    assert s.supports_native_clarify is True
    assert hasattr(s, "set_answerer")
    print("✓ 声明为原生澄清通路")


def test_finish_question_detection_is_conservative():
    """Only a finish that is clearly waiting for the client is intercepted."""
    class Finish:
        def __init__(self, text):
            self.final_thought = text

    assert oh_driver._finish_requests_client(Finish(
        "Before I proceed:\n\n## Questions for the client\n1. Which unit?"))
    assert oh_driver._finish_requests_client(Finish(
        "I need some clarification before implementation. Which unit should I use?"))
    assert not oh_driver._finish_requests_client(Finish(
        "Completed the report. Shall I also draw a chart?"))
    assert not oh_driver._finish_requests_client(Finish(
        "Completed successfully; files are in the workspace."))


def test_progress_message_detection_is_conservative():
    class Message:
        def __init__(self, text): self.content = text

    assert oh_driver._progress_message(Message(
        "I'll start by exploring the working directory and provided data files."))
    assert oh_driver._progress_message(Message("Let me first inspect the dataset."))
    assert not oh_driver._progress_message(Message("Which unit should I use?"))
    assert not oh_driver._progress_message(Message(
        "I need some clarification before proceeding."))


def test_final_message_never_uses_recall_or_error_observation():
    def event(name, source="agent", content="", final_thought=""):
        obj = type(name, (), {})()
        obj.source, obj.content, obj.final_thought = source, content, final_thought
        return obj

    history = [
        event("MessageAction", "agent", "real agent answer"),
        event("ErrorObservation", "agent", "Missing required parameters"),
        event("RecallObservation", "environment", "Added workspace context"),
    ]
    assert oh_driver._final_agent_message(history) == "real agent answer"

    rejected = event("AgentRejectAction", "agent")
    rejected.message = "Task is rejected by the agent. Reason: unsupported"
    assert oh_driver._final_agent_message([rejected]) == rejected.message


def test_history_diagnostics_detects_schema_validation_storm():
    def event(name, source="agent", content=""):
        obj = type(name, (), {})()
        obj.source, obj.content = source, content
        return obj

    history = [event("ErrorObservation", content=(
        "Missing required parameters for function 'execute_bash': {'security_risk'}"))
               for _ in range(12)]
    history += [event("CmdRunAction"), event("AgentFinishAction")]
    d = oh_driver._history_diagnostics(history)
    assert d["error_count"] == 12 and d["tool_schema_error_count"] == 12
    assert d["tool_action_count"] == 1 and d["finish_count"] == 1


def test_history_diagnostics_accepts_explicit_agent_rejection():
    obj = type("AgentRejectAction", (), {})()
    obj.source = "agent"
    obj.content = ""
    d = oh_driver._history_diagnostics([obj])
    assert d["reject_count"] == 1 and d["finish_count"] == 0


def test_responses_bridge_choices_are_coalesced_without_dropping_tools():
    """Responses output items must satisfy OpenHands 1.6's one-choice contract."""
    tool_a = SimpleNamespace(id="call-a", function=SimpleNamespace(
        name="execute_bash", arguments='{"command":"pwd"}'))
    tool_b = SimpleNamespace(id="call-b", function=SimpleNamespace(
        name="str_replace_editor", arguments='{"command":"view"}'))
    response = SimpleNamespace(choices=[
        SimpleNamespace(index=0, finish_reason="stop",
                        message=SimpleNamespace(role="assistant", content="first", tool_calls=None)),
        SimpleNamespace(index=1, finish_reason="tool_calls",
                        message=SimpleNamespace(role=None, content=None, tool_calls=[tool_a])),
        SimpleNamespace(index=2, finish_reason="tool_calls",
                        message=SimpleNamespace(role="assistant", content="second", tool_calls=[tool_b])),
    ])

    original_count = oh_driver._coalesce_response_choices(response)

    assert original_count == 3
    assert len(response.choices) == 1
    msg = response.choices[0].message
    assert msg.role == "assistant"
    assert msg.content == "first\nsecond"
    assert [x.id for x in msg.tool_calls] == ["call-a", "call-b"]
    assert response.choices[0].finish_reason == "tool_calls"


def test_responses_bridge_single_choice_is_unchanged():
    message = SimpleNamespace(role="assistant", content="done", tool_calls=None)
    response = SimpleNamespace(choices=[SimpleNamespace(
        index=0, finish_reason="stop", message=message)])
    assert oh_driver._coalesce_response_choices(response) == 1
    assert response.choices[0].message is message
    assert message.content == "done"


def test_build_config_forces_native_tool_calling(monkeypatch):
    """新模型名不能让 OpenHands 退回带 security_risk 的 XML 模拟协议。"""
    import sys
    import types

    class Config:
        model_fields = {}
        def set_llm_config(self, llm): self.llm = llm

    class LLMConfig:
        def __init__(self, **kwargs):
            # Mirror OpenHands 1.6: these fields cannot be None at validation
            # time, although the downstream client needs them omitted.
            assert kwargs.get("temperature", 0.0) is not None
            assert kwargs.get("top_p", 1.0) is not None
            self.kwargs = kwargs
            self.temperature = kwargs.get("temperature", 0.0)
            self.top_p = kwargs.get("top_p", 1.0)

    cfg_mod = types.ModuleType("openhands.core.config")
    cfg_mod.LLMConfig, cfg_mod.OpenHandsConfig = LLMConfig, Config
    utils_mod = types.ModuleType("openhands.core.config.utils")
    utils_mod.load_openhands_config = Config
    monkeypatch.setitem(sys.modules, "openhands", types.ModuleType("openhands"))
    monkeypatch.setitem(sys.modules, "openhands.core", types.ModuleType("openhands.core"))
    monkeypatch.setitem(sys.modules, "openhands.core.config", cfg_mod)
    monkeypatch.setitem(sys.modules, "openhands.core.config.utils", utils_mod)

    cfg = oh_driver._build_config({"workspace": "/tmp/ws", "runtime": "cli",
                                   "max_iterations": 2, "model": "openai/qwen3.8-max",
                                   "llm_timeout_s": 3600,
                                   "max_output_tokens": 131072})
    assert cfg.llm.kwargs["native_tool_calling"] is True
    assert cfg.llm.kwargs["timeout"] == 3600
    assert cfg.llm.kwargs["max_output_tokens"] == 131072
    assert "completion_kwargs" not in cfg.llm.kwargs

    glm_cfg = oh_driver._build_config({
        "workspace": "/tmp/ws", "runtime": "cli", "max_iterations": 2,
        "model": "openai/glm-5.3", "llm_timeout_s": 3600,
        "max_output_tokens": 131072, "force_max_tokens": True,
    })
    assert glm_cfg.llm.kwargs["max_output_tokens"] == 131072
    assert glm_cfg.llm.kwargs["completion_kwargs"] == {
        "max_tokens": 131072,
    }

    responses_cfg = oh_driver._build_config({
        "workspace": "/tmp/ws", "runtime": "cli", "max_iterations": 2,
        "model": "openai/responses/gpt-6-astra", "use_responses_api": True,
        "omit_sampling_params": True,
    })
    assert responses_cfg.llm.kwargs["model"] == "openai/responses/gpt-6-astra"
    assert "temperature" not in responses_cfg.llm.kwargs
    assert "top_p" not in responses_cfg.llm.kwargs
    assert responses_cfg.llm.temperature is None
    assert responses_cfg.llm.top_p is None

    # Kimi-k3's responses Responses endpoint rejects the OpenHands default
    # temperature (0.0); the driver must support the same omission path.
    kimi_cfg = oh_driver._build_config({
        "workspace": "/tmp/ws", "runtime": "cli", "max_iterations": 2,
        "model": "openai/responses/kimi-k3", "use_responses_api": True,
        "omit_sampling_params": True,
    })
    assert kimi_cfg.llm.kwargs["model"] == "openai/responses/kimi-k3"
    assert "temperature" not in kimi_cfg.llm.kwargs
    assert "top_p" not in kimi_cfg.llm.kwargs
    assert kimi_cfg.llm.temperature is None
    assert kimi_cfg.llm.top_p is None


def test_session_exposes_driver_protocol_failure(monkeypatch, tmp_path):
    """driver 的异常终态必须传到 CLI，不能再因为 exit=0 被记为 ok。"""
    import asyncio
    import json
    import subprocess
    from harness.backends.base import RunnerConfig

    ws = tmp_path / "ws"
    ws.mkdir()
    monkeypatch.setattr("harness.backends._retry.gateway", lambda model: ("https://gw/v1", "key"))
    result = {
        "final_message": "",
        "notes": ["openhands protocol failure: terminal_state_stopped"],
        "sid": "s1",
        "usage_events": [],
        "effective_models": ["deepseek-v4-pro"],
        "tool_names": [],
        "agent_state": "stopped",
        "diagnostics": {"error_count": 0, "finish_count": 0},
        "failure_reason": "terminal_state_stopped",
    }
    monkeypatch.setattr(
        "harness.backends._proc.run_pg",
        lambda *args, **kwargs: subprocess.CompletedProcess(
            args[0], 0, json.dumps(result), ""),
    )
    s = _OpenHandsSession(
        ws, RunnerConfig(runner_type="openhands", model="chat/deepseek-v4-pro",
                         max_runtime_s=60))
    s.on_event = lambda event: None
    assert asyncio.run(s.send("task")) == ""
    assert s.degraded_reason == "terminal_state_stopped"
    assert s.backend_diagnostics["finish_count"] == 0


def test_session_enables_responses_only_for_responses_routes(monkeypatch, tmp_path):
    """The parent-to-driver payload is the protocol switch's source of truth."""
    import asyncio
    import json
    import subprocess
    from harness.backends.base import RunnerConfig

    captured = []
    result = {
        "final_message": "", "notes": [], "sid": "s1", "usage_events": [],
        "effective_models": [], "tool_names": [], "agent_state": "stopped",
        "diagnostics": {}, "failure_reason": "terminal_state_stopped",
    }

    def fake_run(args, **kwargs):
        captured.append(json.loads(kwargs["input"]))
        return subprocess.CompletedProcess(args, 0, json.dumps(result), "")

    monkeypatch.setattr("harness.backends._proc.run_pg", fake_run)
    monkeypatch.setattr("harness.backends._retry.gateway",
                        lambda model: ("https://gw/v1", "key"))

    for i, model in enumerate(("responses/grok-4.6", "responses/gpt-6-astra",
                               "responses/kimi-k3", "chat/qwen3.8-max",
                               "direct/glm-5.3")):
        ws = tmp_path / str(i); ws.mkdir()
        session = _OpenHandsSession(
            ws, RunnerConfig(runner_type="openhands", model=model, max_runtime_s=60))
        session.on_event = lambda event: None
        asyncio.run(session.send("task"))

    assert [item["use_responses_api"] for item in captured] == [
        True, True, True, False, False,
    ]
    # The Grok Responses route rejects temperature altogether;
    # Astra and Kimi reject OpenHands' default sampling values as well.
    assert [item["omit_sampling_params"] for item in captured] == [
        True, True, True, False, False,
    ]
    assert [item["force_max_tokens"] for item in captured] == [
        False, False, False, False, True,
    ]
    assert [item["llm_timeout_s"] for item in captured] == [3600] * 5
    assert [item["max_output_tokens"] for item in captured] == [131072] * 5
    assert [item["model"] for item in captured] == [
        "openai/responses/grok-4.6",
        "openai/responses/gpt-6-astra",
        "openai/responses/kimi-k3",
        "openai/qwen3.8-max",
        "openai/glm-5.3",
    ]


if __name__ == "__main__":
    orig = ask_detect.detect
    try:
        for fn in [v for k, v in sorted(globals().items()) if k.startswith("test_")]:
            fn()
    finally:
        ask_detect.detect = orig
    print("\nall openhands-clarify tests passed")


def test_driver_reads_metrics_token_usages():
    """OpenHands 1.x 的用量在 state.metrics.token_usages（逐 LLM 调用），不在事件属性里。"""
    import importlib.util
    from pathlib import Path as _P
    spec = importlib.util.spec_from_file_location(
        "oh_driver", _P(__file__).resolve().parents[1] / "backends" / "openhands" / "driver.py")
    drv = importlib.util.module_from_spec(spec); spec.loader.exec_module(drv)

    class TU:
        def __init__(self, **kw): self._d = kw
        def model_dump(self): return dict(self._d)

    class Metrics:
        accumulated_cost = 0.0123
        token_usages = [TU(model="direct/glm-5.3", prompt_tokens=1000, completion_tokens=50,
                           cache_read_tokens=400, cache_write_tokens=0),
                        TU(model="direct/glm-5.3", prompt_tokens=1500, completion_tokens=20,
                           cache_read_tokens=900, cache_write_tokens=0)]

    class State:
        def get_local_metrics(self): return Metrics()

    models, notes = set(), []
    out = drv._metrics_usage(State(), models, notes)
    assert len(out) == 2 and out[0]["prompt_tokens"] == 1000 and out[1]["cache_read_tokens"] == 900
    assert models == {"direct/glm-5.3"} and any("accumulated_cost" in n for n in notes)

    # 账本按 litellm 口径归一：input 不含缓存 → (1000-400)+(1500-900)
    from harness.backends.usage import UsageLedger
    led = UsageLedger(input_includes_cache_read=True)
    for u in out: led.record(u)
    tok = led.snapshot()["tokens"]
    assert tok["input"] == 1200 and tok["cache_read"] == 1300 and tok["steps"] == 2


def test_driver_preserves_openhands_action_types():
    """不能再把所有 OpenHands Action 伪装成 shell。"""
    import importlib.util
    from pathlib import Path as _P
    spec = importlib.util.spec_from_file_location(
        "oh_driver_actions", _P(__file__).resolve().parents[1] / "backends" / "openhands" / "driver.py")
    drv = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(drv)

    def event(name, source="agent"):
        cls = type(name, (), {})
        obj = cls()
        obj.source = source
        return obj

    history = [event("SystemMessageAction"), event("CmdRunAction"), event("FileEditAction"),
               event("MessageAction"), event("CmdRunAction", "user")]
    assert drv._tool_action_names(history) == ["CmdRunAction", "FileEditAction"]


def test_single_send_phase_split_by_steps_at_last_answer(tmp_path):
    """OpenHands 一次 send 含全程：按最后一次作答时的调用数切 clarify/solve，
    秒数减掉 harness 作答时间。修前整段 token/耗时全记成澄清。"""
    from harness.backends.openhands.session import _OpenHandsSession
    from harness.backends.base import RunnerConfig
    ws = tmp_path / "ws"; ws.mkdir()
    s = _OpenHandsSession(ws, RunnerConfig(runner_type="openhands", max_runtime_s=60))
    s._send_started_ms = 1_000_000
    s.clarify_rounds = [{"turn": 1, "questions": [{"question": "q?"}], "answered_at_ms": 1_040_000,
                         "harness_secs": 10.0, "steps_before_answer": 2}]
    events = [{"prompt_tokens": 100, "completion_tokens": 10, "cache_read_tokens": 0},
              {"prompt_tokens": 100, "completion_tokens": 10, "cache_read_tokens": 50},
              {"prompt_tokens": 500, "completion_tokens": 50, "cache_read_tokens": 100},
              {"prompt_tokens": 500, "completion_tokens": 50, "cache_read_tokens": 100}]
    s._record_usage_events(events, elapsed_s=100.0)
    snap = s.usage_snapshot()
    ph = snap["phase"]
    assert ph["clarify"]["input"] == 150 and ph["solve"]["input"] == 800    # 前 2 步 vs 后 2 步（input 已减缓存）
    assert ph["clarify_secs"] == 30.0        # 40s 到最后作答 − 10s harness
    assert ph["solve_secs"] == 60.0          # 100 − 30 − 10
    assert snap["tokens"]["steps"] == 4


def test_single_send_without_sidecar_marks_phase_unknown(tmp_path):
    from harness.backends.openhands.session import _OpenHandsSession
    from harness.backends.base import RunnerConfig
    ws = tmp_path / "ws"; ws.mkdir()
    s = _OpenHandsSession(ws, RunnerConfig(runner_type="openhands", max_runtime_s=60))
    s._send_started_ms = 1_000_000
    s.clarify_rounds = [{"turn": 1, "questions": [{"question": "q?"}], "answered_at_ms": 1_040_000,
                         "steps_before_answer": None}]
    s._record_usage_events([{"prompt_tokens": 100, "completion_tokens": 10}], elapsed_s=50.0)
    ph = s.usage_snapshot()["phase"]
    assert ph["clarify_tokens"] is None and ph["clarify_secs"] is None     # 切不出来就说切不出来


def test_single_send_no_clarify_is_all_solve_with_zero_clarify(tmp_path):
    from harness.backends.openhands.session import _OpenHandsSession
    from harness.backends.base import RunnerConfig
    ws = tmp_path / "ws"; ws.mkdir()
    s = _OpenHandsSession(ws, RunnerConfig(runner_type="openhands", max_runtime_s=60))
    s._send_started_ms = 1_000_000
    s._record_usage_events([{"prompt_tokens": 100, "completion_tokens": 10}], elapsed_s=50.0)
    ph = s.usage_snapshot()["phase"]
    assert ph["clarify_secs"] == 0.0 and ph["solve_secs"] == 50.0 and ph["clarify"]["input"] == 0
    assert ph["solve"]["input"] == 100
