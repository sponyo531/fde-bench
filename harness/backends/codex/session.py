"""Codex backend —— 第一方脚手架，走文本多轮澄清。

与 opencode serve 的结构性差异
────────────────────────────
opencode serve 有原生 `question` tool，框架可在旁路拦截 tool 调用、注入答案。
Codex（以及 Claude Code / Gemini CLI）在 headless 下**不注册任何 ask 类工具**
（实测 2026-08-13），agent 缺信息时直接在回复正文里用自然语言提问并停下。

所以澄清走的是**通用原生通道：多轮对话**——
    send(prompt)        → agent 提问，停在无产物状态
    send(答案)          → agent 继续干活
上层 clarify_loop._run_text 用 ask_detect 语义判定「这轮是不是在提问」，
不要求 agent 输出任何标记格式。

多轮靠 `codex exec resume <session_id> <prompt>` 续接。注意 resume 子命令
**不接受 `-C/--cd`**（实测 rc=2 "unexpected argument"），工作目录只能靠 cwd。
"""

from __future__ import annotations

import asyncio
import json
import os
import subprocess
import time
from pathlib import Path

from ..base import AgentEvent, AgentRunner, AgentSession
from ..usage import UsageLedger, extract_model, extract_usage


def _events(stdout: str):
    """逐行解析 codex 的 JSONL 事件流。"""
    for line in stdout.splitlines():
        line = line.strip()
        if not line.startswith("{"):
            continue
        try:
            yield json.loads(line)
        except json.JSONDecodeError:
            continue


class CodexSession(AgentSession):
    """一次 send = 一次 codex exec（首轮）或 codex exec resume（后续轮）。"""
    reports_tool_calls = True
    input_includes_cache_read = True     # OpenAI usage.input_tokens 含 cached_input_tokens

    # Codex 在 headless 下没有 ask 类工具，澄清只能走文本多轮
    supports_native_clarify = False

    def __init__(self, workspace: Path, config=None):
        self.session_id: str | None = None
        self.on_event = None
        self.killed_reason: str | None = None
        self._workspace = workspace
        self._config = config
        self._usage_ledger = UsageLedger(input_includes_cache_read=self.input_includes_cache_read)
        self._last_call_tokens = None
        self._effective_models: set[str] = set()
        self._rollout_consumed = 0          # 已入账的 rollout token_count 条数
        self._runtime_deadline: float | None = None
        # 独立 CODEX_HOME：config.toml 把 provider 指到网关（wire_api=responses，实测
        # /v1/responses 可用）。不依赖运行者机器的 ~/.codex——pod 里没有；本机有也
        # 可能指向别的网关，跨脚手架对照不能掺端点差异。rollout 也落在这里。
        self._codex_home = self._make_codex_home()

    async def send(self, message: str, *, stop_on_tools: set[str] | None = None,
                   fork: bool = False) -> str:
        self._note_agent_send()
        return await asyncio.to_thread(self._send, message)

    def set_usage_phase(self, phase: str | None) -> None:
        self._usage_ledger.set_phase(phase)

    def usage_snapshot(self) -> dict | None:
        return self._usage_ledger.snapshot()

    def relabel_last_usage_phase(self, phase: str | None) -> None:
        self._usage_ledger.relabel_last(phase)

    def usage_call_count(self) -> int:
        return self._usage_ledger.call_count

    def label_usage_from(self, start: int, phase: str) -> None:
        self._usage_ledger.label_from(start, phase)

    def effective_models(self) -> list[str]:
        return sorted(self._effective_models)

    def _send(self, message: str) -> str:
        max_runtime = int(getattr(self._config, "max_runtime_s", None) or 43200)
        last_file = self._workspace / ".codex_last_message"

        base = ["codex", "exec"]
        if self.session_id:
            # resume 不接受 -C，工作目录靠 cwd 传递
            cmd = base + ["resume", self.session_id]
        else:
            cmd = base + ["-C", str(self._workspace)]
        cmd += ["--json", "--skip-git-repo-check",
                "--dangerously-bypass-approvals-and-sandbox",
                "-o", str(last_file)]
        model = getattr(self._config, "model", None)
        if model and not self.session_id:
            cmd += ["-m", _strip_provider(model)]
        # `--` 必须有：澄清答案以 "- 问题\n  答案" 的列表形式回灌，开头的 `-`
        # 会被 clap 当成命令行参数（实测 exit=2 "unexpected argument '- '"）。
        # Keep large prompts out of argv.  Codex accepts ``-`` as a stdin
        # prompt; this avoids E2BIG when a clarification answer is large.
        cmd += ["-"]
        stdin_data = message

        # 必须先删：-o 写的是"最后一条 agent 消息"，codex 本轮若失败（网关 502、
        # 参数错误等）不会覆盖它，于是我们读到**上一轮**的内容。实测因此把同一批
        # 问题反复检测、反复作答，空转 15 轮撞上限（194 个重复问题）。
        last_file.unlink(missing_ok=True)

        started = time.monotonic()
        if self._runtime_deadline is None:
            self._runtime_deadline = started + max_runtime

        def _run_once():
            from .._proc import run_pg
            remaining = self._runtime_deadline - time.monotonic()
            if remaining <= 0:
                raise subprocess.TimeoutExpired(cmd, max_runtime)
            proc = run_pg(cmd, timeout=remaining, cwd=str(self._workspace),
                          env=self._env(), input=stdin_data)
            if proc.returncode != 0:
                tail = (proc.stderr or proc.stdout or "")[-400:].strip()
                raise RuntimeError(f"codex exit={proc.returncode}: {tail}")
            return proc

        try:
            from .._retry import retry_transient
            proc = retry_transient(
                _run_once, label="codex send",
                on_retry=lambda m: self._emit(AgentEvent(type="info", content=m)))
        except subprocess.TimeoutExpired:
            self.killed_reason = "max_runtime"
            # A continuation timeout still has authoritative per-step usage in
            # the rollout.  Harvest it before close() removes the private
            # CODEX_HOME; otherwise only clarification usage survives.
            self._record_rollout_steps(time.monotonic() - started)
            return last_file.read_text(encoding="utf-8").strip() if last_file.is_file() else ""
        except RuntimeError as exc:
            self._emit(AgentEvent(type="tool_result", is_error=True, content=str(exc)[:400]))
            raise

        texts: list[str] = []
        usage_events = []
        for ev in _events(proc.stdout):
            usage = extract_usage(ev)
            if usage is not None:
                usage_events.append(usage)
            model = extract_model(ev)
            if model:
                self._effective_models.add(model)
            etype = ev.get("type")
            if etype == "thread.started" and not self.session_id:
                self.session_id = ev.get("thread_id") or ev.get("session_id")
                self._emit(AgentEvent(type="info",
                                      content=f"session_id: {self.session_id}"))
            elif etype == "item.completed":
                item = ev.get("item") or {}
                if item.get("type") == "agent_message" and item.get("text"):
                    texts.append(item["text"])
                elif item.get("type") in ("command_execution", "file_change"):
                    self._emit(AgentEvent(type="tool_call",
                                          tool=item.get("type", "?"),
                                          tool_input=str(item)[:500]))
            elif etype == "error":
                self._emit(AgentEvent(type="info",
                                      content=f"[error] {str(ev)[:300]}", is_error=True))

        # 逐 step 用量优先从 rollout 文件取（每次 API 调用一条 token_count），
        # stdout 的 turn.completed.usage 只有整轮总量——gpt-5.6-sol 按每 step 的
        # input+cache_read 是否 >272k 分档，拿整轮聚合判档会系统性高估。
        elapsed = time.monotonic() - started
        if not self._record_rollout_steps(elapsed) and usage_events:
            self._usage_ledger.record(usage_events[-1], elapsed_s=elapsed)
            self._last_call_tokens = self._usage_ledger.last_call_tokens

        # -o 落的是本轮最后一条 agent 消息，比事件流拼接更可靠
        if last_file.is_file():
            tail = last_file.read_text(encoding="utf-8").strip()
            if tail:
                return tail
        return "\n".join(texts)

    def _make_codex_home(self) -> Path:
        import tempfile
        from .._retry import gateway
        model = _strip_provider(getattr(self._config, "model", None) or "gpt-6-astra")
        base_v1, _ = gateway(model)
        # The release package never embeds a gateway address. Configure it in
        # the environment used to run the benchmark.
        base_v1 = os.environ.get("DELIVER_RESPONSES_BASE_URL", base_v1)
        # Codex creates helper binaries below CODEX_HOME/tmp/arg0.  Recent
        # Codex CLI releases refuse to create those helpers when CODEX_HOME
        # itself lives below the process temporary directory (/tmp), which is
        # where tempfile.mkdtemp() would place it by default.  Keep the home
        # isolated per run, but anchor it in the run directory on the shared filesystem instead.
        # The directory is removed by close(), so this does not become part of
        # the submitted artifact tree.
        home_parent = self._workspace.parent
        home_parent.mkdir(parents=True, exist_ok=True)
        home = Path(tempfile.mkdtemp(prefix=".codex_home_", dir=str(home_parent)))
        (home / "config.toml").write_text(
            f'model = "{model}"\nmodel_provider = "benchmark_responses"\n'
            'approval_policy = "never"\nsandbox_mode = "danger-full-access"\n'
            'web_search = "disabled"\n\n'
            '[model_providers.benchmark_responses]\nname = "Benchmark responses"\n'
            f'base_url = "{base_v1}"\nwire_api = "responses"\nenv_key = "DELIVER_RESPONSES_API_KEY"\n',
            encoding="utf-8")
        return home

    def _env(self) -> dict:
        from .._retry import gateway
        model = _strip_provider(getattr(self._config, "model", None) or "gpt-6-astra")
        _, key = gateway(model)
        env = dict(os.environ, CODEX_HOME=str(self._codex_home))
        if key:
            env["DELIVER_RESPONSES_API_KEY"] = key
        return env

    def _rollout_path(self) -> Path | None:
        """codex 的会话日志：$CODEX_HOME/sessions/YYYY/MM/DD/rollout-<ts>-<thread_id>.jsonl。"""
        if not self.session_id:
            return None
        home = self._codex_home
        hits = sorted((home / "sessions").glob(f"*/*/*/rollout-*-{self.session_id}.jsonl"))
        return hits[-1] if hits else None

    def _record_rollout_steps(self, elapsed_s: float) -> bool:
        """把本次 send 新增的 token_count 事件逐条入账。返回是否记到了东西。

        token_count.info.last_token_usage 是该次 API 调用的用量（OpenAI 口径，
        input_tokens 含 cached_input_tokens，账本会归一）。用已消费条数做游标，
        续接轮只记新增部分。
        """
        path = self._rollout_path()
        if path is None or not path.is_file():
            return False
        steps: list[dict] = []
        try:
            with path.open(encoding="utf-8", errors="replace") as fh:
                for line in fh:
                    try:
                        d = json.loads(line)
                    except json.JSONDecodeError:
                        continue
                    if d.get("type") == "turn_context":
                        m = (d.get("payload") or {}).get("model")
                        if m:
                            self._effective_models.add(str(m))
                        continue
                    if d.get("type") != "event_msg":
                        continue
                    payload = d.get("payload") or {}
                    if payload.get("type") != "token_count":
                        continue
                    last = (payload.get("info") or {}).get("last_token_usage")
                    if isinstance(last, dict):
                        steps.append(last)
        except OSError:
            return False
        fresh = steps[self._rollout_consumed:]
        if not fresh:
            return False
        for i, u in enumerate(fresh):
            self._usage_ledger.record(u, elapsed_s=elapsed_s if i == len(fresh) - 1 else None)
        self._rollout_consumed = len(steps)
        self._last_call_tokens = self._usage_ledger.last_call_tokens
        return True

    async def close(self) -> None:
        # codex exec 每轮都是独立进程，退出即收尾，无常驻资源
        stale = self._workspace / ".codex_last_message"
        if stale.is_file():
            stale.unlink()
        import shutil
        shutil.rmtree(self._codex_home, ignore_errors=True)


def _strip_provider(model: str) -> str:
    """`responses/gpt-6-astra` → `gpt-6-astra`。

    provider 前缀是 opencode 的写法；codex 的 provider 在 ~/.codex/config.toml
    里配置，命令行只认裸模型名。
    """
    return model.split("/", 1)[-1] if "/" in model else model


class CodexRunner(AgentRunner):
    async def start(self, workspace: Path) -> CodexSession:
        workspace.mkdir(parents=True, exist_ok=True)
        return CodexSession(workspace, config=self.config)
