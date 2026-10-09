"""OpenHands 后端（副表用）。

与 opencode 后端的结构性差异，直接影响 C 条件怎么接：

opencode 是常驻进程，外部可反复 send，澄清循环由我们在外面驱动。
OpenHands 的 run_controller 是「一次跑到底」，agent 需要用户输入时进入
AWAITING_USER_INPUT，回调 fake_user_response_fn 取回复。所以澄清必须
注入到它自己的循环里，不能从外部多次 send。

由此 send() 的语义在本后端是「跑完整个任务」：首次 send 真跑，
后续 send 只在澄清回调里消费——见 _OpenHandsSession.send 的说明。

依赖隔离：本机全局 rich 13.7.1 与 fastmcp 3.2.4 冲突（RichHandler 不认
tracebacks_max_frames），而 17 个包依赖全局 rich，不能升级。故 OpenHands
跑在 .venv-openhands 里，通过子进程调用，rich 15 只存在于该 venv。
"""

from __future__ import annotations

import json
import os
import subprocess
import threading
import time
from datetime import datetime, timezone
from pathlib import Path

from ..base import AgentEvent, AgentRunner, AgentSession, RunnerConfig
from ..usage import UsageLedger

def _cfg(key: str, default):
    from ...config import get
    return get("openhands", key, default)


# 本文件在 harness/backends/openhands/ 下，仓库根要上跳四层
_REPO = Path(__file__).resolve().parents[3]
# 镜像里 venv 在 /opt/venv-openhands，由 Dockerfile 的 DELIVER_OPENHANDS_VENV 指过来；
# 本机是仓库内的 .venv-openhands。此前只认 config 相对路径，pod 里找不到。
_VENV = Path(os.environ["DELIVER_OPENHANDS_VENV"]) if os.environ.get("DELIVER_OPENHANDS_VENV") \
    else _REPO / str(_cfg("venv", ".venv-openhands"))
_DRIVER = Path(__file__).resolve().parent / "driver.py"

# CLIRuntime：本机直跑，不起容器。与 opencode 后端同机同 workspace，
# 保证两脚手架的对照不掺入运行环境差异。
_RUNTIME = "cli"


class _OpenHandsSession(AgentSession):
    """把 run_controller 的一次性执行包装成 AgentSession。

    send() 首次调用跑完整个任务并返回最终回复。澄清回答不走 send，
    而是预先通过 answers_file 交给驱动脚本，由 fake_user_response_fn 消费。
    """

    # OpenHands 的澄清走它原生的 fake_user_response_fn 回调（在 driver 子进程里），
    # 不是文本多轮——所以 send() 只调一次，clarify_loop 走原生通路。
    supports_native_clarify = True
    # 一次 send 跑完全程，不能再发第二条消息（send 会 raise）。clarify/loop 据此不做
    # 文本回落、不推 PROCEED——那两条路都要再 send。
    supports_followup = False
    reports_tool_calls = True          # driver 从 state.history 数 Action

    def __init__(self, workspace: Path, config: RunnerConfig, *, clarify_answers: Path | None = None):
        self.workspace = workspace
        self._workspace = workspace          # clarify/loop._has_deliverable 等通用代码读这个名字
        self.config = config
        self.clarify_answers = clarify_answers
        self.session_id: str | None = None
        self._done = False
        self._answerer = None
        self._max_clarify_rounds = 30
        self._max_clarify_questions = 0
        self._answered_clarify_questions = 0
        self.clarify_rounds: list[dict] = []
        self.hit_round_limit = False
        self.hit_question_limit = False
        self.killed_reason: str | None = None
        self.degraded_reason: str | None = None
        self.backend_diagnostics: dict = {}
        # OpenHands 经 litellm：prompt_tokens 含 cache_read，账本入账归一（见 base）
        self.input_includes_cache_read = True
        self._usage_ledger = UsageLedger(input_includes_cache_read=True)
        self._effective_models: set[str] = set()

    def set_answerer(self, answerer, *, max_rounds: int = 30,
                     max_questions: int = 0) -> None:
        """挂上澄清回答者（clarify_loop 的原生通路契约）。"""
        self._answerer = answerer
        self._max_clarify_rounds = max_rounds
        self._max_clarify_questions = max_questions
        # 有 answerer 时才建 IPC 通道；无则 driver 侧 exit_on_message 生效
        self.clarify_answers = self.workspace / ".oh_clarify"

    async def send(self, message: str, *, stop_on_tools: set[str] | None = None) -> str:
        if self._done:
            # 本后端一次 send 即跑完全程；多轮澄清由驱动内的回调承担。
            # 静默接受重复 send 会让 C 条件看起来能多轮、实际没有，故显式拒绝。
            raise RuntimeError(
                "OpenHands backend runs to completion in a single send(). "
                "Clarification must be routed through clarify_answers, not repeated sends."
            )
        self._note_agent_send()

        # LiteLLM uses openai/<m> for OpenAI-compatible Chat Completions.
        model = self.config.model or ""
        from .._retry import gateway
        from ...model_endpoints import bare_model, is_responses_model
        base_v1, key = gateway(model)
        use_responses_api = is_responses_model(model)
        requested_bare_model = bare_model(model)
        # OpenHands 1.6 translates LLMConfig.max_output_tokens to LiteLLM's
        # Some Chat Completions endpoints require max_tokens instead.
        force_max_tokens = model.startswith("direct/")
        llm_timeout_s = int(_cfg("llm_timeout_s", 3600))
        max_output_tokens = int(_cfg("max_output_tokens", 131072))
        stream = str(_cfg("stream", False)).strip().lower() in {"1", "true", "yes", "on"}
        if stream and use_responses_api:
            raise ValueError("OpenHands streaming currently supports Chat Completions only")
        if use_responses_api:
            # LiteLLM's documented/implemented bridge selector is the
            # responses/ model prefix.  With the provider prefix included the
            # final ID is openai/responses/<model>; completion() then sends
            # POST /responses and converts the result back to ModelResponse.
            model = "openai/responses/" + requested_bare_model
        elif model.startswith(("chat/", "direct/")):
            model = "openai/" + model.split("/", 1)[1]
        payload = {
            "task": message,
            "workspace": str(self.workspace),
            "model": model,
            "base_url": base_v1,
            "api_key": key,
            # Protocol marker retained in the driver payload/diagnostics; the
            # openai/responses/ model prefix above is the actual enforcement.
            "use_responses_api": use_responses_api,
            "stream": stream,
            "stream_sidecar": str(self.workspace.parent / ".oh_stream.json"),
            # Some Responses models reject OpenHands' default
            # sampling parameters.  Kimi-k3 is strict as well: its Responses
            # endpoint accepts only temperature=1, while Bedrock-backed Grok
            # rejects the field entirely.  Use each provider's default by
            # omitting temperature/top_p for all three models.
            "omit_sampling_params": requested_bare_model in (
                "gpt-6-astra", "grok-4.6", "kimi-k3",
            ),
            # GLM/Qwen can legitimately spend well beyond LiteLLM's default
            # timeout on a single 30k+ token reasoning/tool turn.
            "llm_timeout_s": llm_timeout_s,
            # OpenHands does not consume opencode/opencode.jsonc, where the
            # formal models already declare a 131072-token output budget.
            # Pass it explicitly; otherwise OpenHands/LiteLLM currently falls
            # back to 65536 and GLM-5.3 can exhaust the entire turn on
            # reasoning without returning content or a tool call.
            "max_output_tokens": max_output_tokens,
            "force_max_tokens": force_max_tokens,
            "max_iterations": int(_cfg("max_iterations", 200)),
            "runtime": str(_cfg("runtime", _RUNTIME)),
            "trajectory": str(self.workspace / ".openhands_trajectory.json"),
            "clarify_answers": str(self.clarify_answers) if self.clarify_answers else "",
            # 无澄清渠道时，agent 求助即退出，避免它空等到超时。
            "exit_on_message": self.clarify_answers is None,
            # 用量 sidecar 放 run 目录（workspace 的上一级），不算交付物
            "usage_sidecar": str(self.workspace.parent / ".oh_usage.json"),
        }
        self.backend_diagnostics.update({
            "wire_protocol": "responses" if use_responses_api else "chat_completions",
            "stream": stream,
            "routed_model": model,
            "llm_timeout_s": llm_timeout_s,
            "max_output_tokens": max_output_tokens,
            "output_token_param": (
                "max_tokens" if force_max_tokens else "max_completion_tokens"
            ),
        })

        self._emit(AgentEvent(type="info", content=f"openhands: {_RUNTIME} runtime, model={payload['model'] or 'default'}, stream={stream}"))

        # 澄清应答服务：driver 在子进程里把提问写成 .ask 文件，这里实时作答。
        # 必须并行——driver 会阻塞等 .ans，串行就死锁了。
        stop = threading.Event()
        server = None
        if self._answerer is not None:
            server = threading.Thread(target=self._serve_clarify, args=(stop,),
                                      daemon=True, name="oh-clarify")
            server.start()

        started = time.monotonic()
        self._send_started_ms = int(time.time() * 1000)
        try:
            from .._proc import run_pg
            proc = run_pg(
                [str(_VENV / "bin" / "python"), str(_DRIVER)],
                input=json.dumps(payload),
                timeout=self.config.max_runtime_s,
                cwd=str(self.workspace),
                env={**os.environ, "PYTHONUNBUFFERED": "1"},
            )
        except subprocess.TimeoutExpired:
            # 与其他后端同口径：记 killed_reason，run 记 timeout 而不是 harness error。
            # 用量从 driver 每 5s 写的 sidecar 读，不至于整段丢。
            self.killed_reason = "max_runtime"
            # 工具计数只在 driver 收尾时返回；被掐就没有 → 记 None 而不是 0
            self.reports_tool_calls = False
            self._done = True
            stop.set()
            if server is not None:
                server.join(timeout=5)
            self._ingest_sidecar(payload["usage_sidecar"], time.monotonic() - started)
            self._emit(AgentEvent(type="info",
                                  content=f"[watchdog] max_runtime ({self.config.max_runtime_s}s) — kill openhands driver"))
            return ""
        self._done = True
        stop.set()
        if server is not None:
            server.join(timeout=5)

        if proc.returncode != 0:
            # 只挑有信息量的行：driver error / Traceback / ERROR，跳过 openhands 的 INFO 噪音
            lines = [line for line in (proc.stderr or "").splitlines()
                     if line.strip() and ("driver error" in line or "Error" in line or "ERROR" in line
                                          or "Traceback" in line or "raise" in line)]
            tail = (lines or (proc.stderr or "").strip().splitlines())[-6:]
            # 必须发 info：cli.py 只打印 info 事件，tool_result 静默进 events 列表，
            # 实测 driver 26s 失败、run 记 degraded、日志里一个字都没有
            self._emit(AgentEvent(type="info",
                                  content="[openhands] driver failed (exit=%d): %s" % (proc.returncode, " | ".join(tail)[-900:])))
            return ""

        try:
            result = json.loads(proc.stdout.strip().splitlines()[-1])
        except (json.JSONDecodeError, IndexError):
            self._emit(AgentEvent(type="info",
                                  content="[openhands] driver returned unparseable output: "
                                          + (proc.stdout or "")[-300:]))
            return ""

        self.session_id = result.get("sid")
        self.backend_diagnostics.update(result.get("diagnostics") or {})
        failure_reason = str(result.get("failure_reason") or "").strip()
        if failure_reason:
            self.degraded_reason = failure_reason
        limit_reason = str(result.get("limit_reason") or "").strip()
        if limit_reason:
            self.killed_reason = limit_reason
        self._effective_models.update(result.get("effective_models", []) or [])
        self._record_usage_events(result.get("usage_events", []) or [], time.monotonic() - started)
        tool_names = result.get("tool_names")
        if isinstance(tool_names, list):
            for name in tool_names:
                self._emit(AgentEvent(type="tool_call", tool=str(name or "openhands-action")))
        else:
            # 兼容旧 driver 输出；未知类别时宁可保留 coarse 名，不伪造具体工具。
            for _ in range(int(result.get("tool_calls") or 0)):
                self._emit(AgentEvent(type="tool_call", tool="openhands-action"))
        for note in result.get("notes", []):
            self._emit(AgentEvent(type="info", content=note))
        return result.get("final_message", "")

    def usage_snapshot(self) -> dict | None:
        snapshot = self._usage_ledger.snapshot()
        if snapshot is not None and snapshot.get("phase") is not None:
            # 一次 send 内按「最后一次澄清作答时的 LLM step 数/时间戳」切分。
            snapshot["phase_precision"] = "event_anchor"
        return snapshot

    def effective_models(self) -> list[str]:
        return sorted(self._effective_models)

    def set_usage_phase(self, phase: str | None) -> None:
        self._usage_ledger.set_phase(phase)

    def relabel_last_usage_phase(self, phase: str | None) -> None:
        self._usage_ledger.relabel_last(phase)

    def _serve_clarify(self, stop: threading.Event) -> None:
        """轮询 driver 写出的 .ask 文件，调 answerer 后写 .ans。

        撞上 max_rounds 就回固定的「无更多信息」——继续无限作答只会空转
        （opencode 那边实测过连问 15 轮同一批问题）。
        """
        base = self.clarify_answers
        served: set[Path] = set()
        while not stop.wait(0.3):
            for ask in sorted(base.parent.glob(base.name + ".*.ask")):
                if ask in served:
                    continue
                served.add(ask)
                try:
                    question = ask.read_text(encoding="utf-8").strip()
                except OSError:
                    continue

                turn = len(self.clarify_rounds) + 1
                if turn > self._max_clarify_rounds:
                    self.hit_round_limit = True
                    reply = ("No further information is available. "
                             "Proceed with your best judgement.")
                else:
                    from ...clarify.detect import detect
                    # 与文本通路同一套语义判定：agent 可能只是在汇报进度而非提问
                    t_h = time.time()
                    questions = detect(question) if question else []
                    if questions:
                        remaining = (None if self._max_clarify_questions == 0 else
                                     max(0, self._max_clarify_questions
                                         - self._answered_clarify_questions))
                        allowed = questions if remaining is None else questions[:remaining]
                        dropped = len(questions) - len(allowed)
                        answers = [{"question": q, "answer": self._answerer.answer(q)}
                                   for q in allowed]
                        self._answered_clarify_questions += len(answers)
                        if dropped:
                            self.hit_question_limit = True
                        # 此刻已发生的模型调用数（读 driver 每 5s 落的 sidecar）：一次 send 含
                        # 全程，阶段只能按"最后一次作答时跑到第几步"切；没有它，整段 token
                        # 和耗时会全记成澄清（或全记成求解）
                        self.clarify_rounds.append({
                            "turn": turn,
                            "asked_at": datetime.now(timezone.utc).isoformat(),
                            "questions": [{"question": q} for q in questions],
                            "answers": answers,
                            "dropped_over_budget": dropped,
                            "answered_at_ms": int(time.time() * 1000),
                            "harness_secs": round(time.time() - t_h, 1),
                            "steps_before_answer": self._sidecar_steps(),
                        })
                        flush = getattr(self, "_clarify_flush", None)
                        if callable(flush):
                            try:
                                flush()
                            except Exception:          # noqa: BLE001  落盘失败不能打断作答
                                pass
                        reply = "\n".join(f"- {a['question']}\n  {a['answer']}"
                                          for a in answers)
                        if dropped or (self._max_clarify_questions > 0 and
                                       self._answered_clarify_questions >=
                                       self._max_clarify_questions):
                            reply += ("\n\nNo further information is available. "
                                      "Proceed with your best judgement.")
                        self._emit(AgentEvent(
                            type="info",
                            content=f"[clarify] turn {turn}: {len(answers)} 问已作答"))
                    else:
                        reply = ("Understood. Please carry out the work and write the "
                                 "deliverable files into the working directory.")
                ask.with_suffix(".ans").write_text(reply, encoding="utf-8")

    def _sidecar_steps(self) -> int | None:
        try:
            data = json.loads(Path(self.workspace.parent / ".oh_usage.json").read_text(encoding="utf-8"))
            return len(data.get("usage_events") or [])
        except (OSError, ValueError, json.JSONDecodeError):
            return None

    def _record_usage_events(self, events: list, elapsed_s: float) -> None:
        for i, usage in enumerate(events):
            # 本轮墙钟只记在最后一条上，phase 秒数按 send 归属
            self._usage_ledger.record(usage, elapsed_s=elapsed_s if i == len(events) - 1 else None)
        self._apply_phase_split(elapsed_s)

    def _apply_phase_split(self, elapsed_s: float) -> None:
        """一次 send 含澄清与求解：按最后一轮作答时的调用数把账本切成两段。

        clarify_secs = send 开始 → 最后一次作答（减去 harness 自己作答的时间，
        agent 只是在等）；solve_secs = 其余。没有澄清轮次 → 全部求解。
        """
        if not self.clarify_rounds:
            # 没澄清：与 opencode 口径一致，clarify 记 0（不是 None），全部归求解
            self._usage_ledger.label_calls(0)
            snap = self._usage_ledger.snapshot() or {}
            ph = dict(snap.get("phase") or {})
            ph.update({"clarify_secs": 0.0, "solve_secs": round(elapsed_s, 1),
                       "clarify": {"input": 0, "output": 0, "reasoning": 0,
                                   "cache_read": 0, "cache_write": 0, "steps": 0}})
            self._usage_ledger.set_phase_override(ph)
            return
        last = self.clarify_rounds[-1]
        k = last.get("steps_before_answer")
        harness = sum((r.get("harness_secs") or 0) for r in self.clarify_rounds)
        if k is None:
            # sidecar 读不到：token 切不出来，秒数还能切；token 阶段记 None 而不是乱归
            self._usage_ledger.set_phase_override({
                "clarify_secs": None, "solve_secs": None,
                "clarify_tokens": None, "solve_tokens": None})
            return
        self._usage_ledger.label_calls(k)
        snap = self._usage_ledger.snapshot() or {}
        ph = dict(snap.get("phase") or {})
        started_ms = getattr(self, "_send_started_ms", None)
        answered = last.get("answered_at_ms")
        if started_ms and answered:
            clarify_secs = max(0.0, (answered - started_ms) / 1000 - harness)
            ph["clarify_secs"] = round(clarify_secs, 1)
            ph["solve_secs"] = round(max(0.0, elapsed_s - clarify_secs - harness), 1)
        self._usage_ledger.set_phase_override(ph)

    def _ingest_sidecar(self, path: str, elapsed_s: float) -> None:
        try:
            data = json.loads(Path(path).read_text(encoding="utf-8"))
        except (OSError, ValueError, json.JSONDecodeError):
            return
        self._effective_models.update(data.get("effective_models") or [])
        self._record_usage_events(data.get("usage_events") or [], elapsed_s)

    async def close(self) -> None:
        return None


class OpenHandsRunner(AgentRunner):
    """OpenHands 的 AgentRunner 实现。"""

    def __init__(self, config: RunnerConfig, *, clarify_answers: Path | None = None):
        super().__init__(config)
        self.clarify_answers = clarify_answers

    async def start(self, workspace: Path) -> AgentSession:
        if not (_VENV / "bin" / "python").exists():
            raise RuntimeError(
                f"OpenHands venv 缺失：{_VENV}\n"
                "重建：python3 -m venv --system-site-packages .venv-openhands "
                "&& .venv-openhands/bin/pip install 'rich>=14'"
            )
        return _OpenHandsSession(workspace, self.config, clarify_answers=self.clarify_answers)
