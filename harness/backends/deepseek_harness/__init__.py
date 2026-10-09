"""DeepSeek Harness（`deepseek-ai/deepseek-harness`，命令行叫 dsh）——走 `--profile sdk` 的 JSON-RPC stdio。

    scaffold = "deepseek-harness"     （别名 "dsh"）

为什么不用 `dsh --profile headless`
──────────────────────────────────
headless 是**一次性**的：跑一个任务、打印最终答案、退出，官方 README 明说
"one task per run, no interactive follow-up"，也没有 resume。文本澄清回路要求
同一会话多轮（答案作为下一条 user message 回灌），headless 走不了。

`--profile sdk` 起一个常驻进程，stdin/stdout 走换行分隔的 JSON-RPC 2.0：

    client→server  initialize        {cwd, provider, model, reasoningEffort?, maxTokens?}
    client→server  session/prompt    {sessionId, contentBlocks:[{type:"text", text}]}
                                     未知 sessionId 会惰性建会话；复用即多轮
    client→server  shutdown
    server→client  session.event     {sessionId, event}   会话日志逐条推送
    server→client  session.status    {sessionId, status: running|idle}

一轮结束的判据：本会话出现 `turn/end` 事件（`reason.kind == completed|aborted|error`）。
回复文本取该轮最后一条 `assistant/message` 里的 text block（reasoning block 不算）。

澄清通道
────────
dsh 自带 `ask_user_question` 工具，但协议文档明说 "Server→client requests are a
dead capability — the server never sends one"，SDK 模式下没有 answerer，模型调它
只会拿到 error。实测（2026-09-02，ds/deepseek-v4-pro）模型在正文里自然语言提问、
不调该工具。所以这里和 Claude Code / Codex 一样：supports_native_clarify=False，
走 clarify/loop.py 的文本多轮。

provider 配置
─────────────
dsh 的模型路由在 `$DSH_HOME/settings.yaml` 的 `llm-pi-ai.providers` 段声明，任意
OpenAI 兼容网关都是配置而非代码：api=openai-completions + baseURL + apiKeyEnv。
每个 session 用独立的临时 DSH_HOME，避免会话库与其他 run 互相污染，close 时删除。
模型名沿用仓库惯例 `chat/<model>`：前缀是 provider 路由名，后半是模型名。

版本：实测 @deepseek-ai/dsh 0.1.2-alpha.5（npm `alpha` tag；rc.2 的子包在国内镜像
不齐）。项目自称 developer preview，"THERE WILL BE COMPATIBILITY-BREAKING CHANGES"。
"""

from __future__ import annotations

import asyncio
import json
import os
import shutil
import subprocess
import tempfile
import threading
import time
import uuid
from pathlib import Path

from ..base import AgentEvent, AgentRunner, AgentSession
from ..usage import UsageLedger, extract_usage

_DEFAULT_BASE_URL = ""
_DEFAULT_KEY_ENV = "DELIVER_AGENT_API_KEY"


def _cfg(section: str, key: str, default=None):
    try:
        from ...config import get
        return get(section, key, default)
    except Exception:              # noqa: BLE001  config.toml 缺段时用默认
        return default


def _dsh_bin() -> str:
    """dsh 可执行文件：config [deepseek_harness].bin（存在时）> PATH 上的 dsh。

    config.toml may be copied into an isolated runtime; an absolute local path
    may not exist there, so fall back to PATH.
    """
    b = _cfg("deepseek_harness", "bin", None)
    if b and not Path(b).exists():
        b = None
    b = b or shutil.which("dsh")
    if not b:
        raise FileNotFoundError(
            "找不到 dsh。装法：npm i -g @deepseek-ai/dsh@alpha "
            "--registry=https://registry.npmmirror.com；或在 config.toml [dsh] bin 指定路径")
    return str(b)


def _split_model(model: str | None) -> tuple[str, str]:
    """`chat/deepseek-v4-pro` → (provider 路由名, 模型名)。"""
    provider = str(_cfg("deepseek_harness", "provider", "chat") or "chat")
    m = (model or _cfg("agent", "model", "") or f"{provider}/deepseek-v4-pro").strip()
    if m.startswith(provider + "/"):
        return provider, m[len(provider) + 1:]
    return provider, m


def _write_home(home: Path, provider: str, model: str) -> None:
    """生成最小 settings.yaml：一个 OpenAI 兼容路由 + 默认模型。"""
    base_url = _cfg("deepseek_harness", "base_url", _DEFAULT_BASE_URL)
    key_env = _cfg("deepseek_harness", "api_key_env", _DEFAULT_KEY_ENV)
    ctx = int(_cfg("agent", "default_context_tokens", 262144) or 262144)
    home.mkdir(parents=True, exist_ok=True)
    (home / "settings.yaml").write_text(
        "llm-pi-ai:\n"
        "  providers:\n"
        f"    {provider}:\n"
        f"      displayName: {provider}\n"
        f"      apiKeyEnv: {key_env}\n"
        "      api: openai-completions\n"
        f"      baseURL: {base_url}\n"
        "      models:\n"
        f"        - id: {model}\n"
        f"          contextWindow: {ctx}\n"
        "agent-default-model:\n"
        f"  provider: {provider}\n"
        f"  model: {model}\n",
        encoding="utf-8",
    )


class DeepSeekHarnessSession(AgentSession):
    """一个 `dsh --profile sdk` 进程 = 一个会话；send() 复用同一 sessionId 实现多轮。"""

    supports_native_clarify = False
    reports_tool_calls = True
    # pi-ai TokenUsage 的 inputTokens 与 cacheReadTokens 是独立字段（Anthropic 形态），按不含缓存处理；
    # 若实测某 provider 下 cacheReadTokens > inputTokens 之类的异常再翻转。
    input_includes_cache_read = False

    def __init__(self, workspace: Path, config=None):
        self._workspace = workspace
        self._config = config
        self.session_id: str | None = None
        self.on_event = None
        self.killed_reason: str | None = None
        self.degraded_reason: str | None = None
        self._usage_ledger = UsageLedger()
        self._last_call_tokens = None
        self._effective_models: set[str] = set()
        self._proc: subprocess.Popen | None = None
        self._home = Path(tempfile.mkdtemp(prefix="deepseek_harness_home_"))
        self._provider, self._model = _split_model(getattr(config, "model", None))
        self._rpc_id = 0
        self._lock = threading.Lock()
        self._replies: dict[int, dict] = {}
        self._events: list[dict] = []           # session.event 的 event 体（仅本会话）
        self._last_activity = time.monotonic()
        self._reader: threading.Thread | None = None
        self._stderr: list[str] = []
        self._runtime_deadline: float | None = None
        # dsh emits the same provider usage twice for a successful step: first
        # as ``assistant/chunk`` (chunk.type=usage), then as the authoritative
        # ``assistant/message.data.usage``.  Keep one pending sample per
        # turn/step and let the final message replace the streaming sample.
        # Recording both doubled tokens, calls, phase totals and cost.
        self._usage_pending: dict[tuple[object, object], dict] = {}

    # ── 进程与 RPC ─────────────────────────────────────────────────────────

    def start(self) -> None:
        _write_home(self._home, self._provider, self._model)
        env = dict(os.environ, DSH_HOME=str(self._home))
        key_env = _cfg("deepseek_harness", "api_key_env", _DEFAULT_KEY_ENV)
        if not env.get(key_env):
            # 与 _cli_spec 同一兜底：config [agent].api_key，再退到 OPENAI_API_KEY
            key = _cfg("agent", "api_key", None) or os.environ.get("OPENAI_API_KEY")
            if key:
                env[key_env] = key
        self._proc = subprocess.Popen(
            [_dsh_bin(), "--profile", "sdk"],
            cwd=str(self._workspace), env=env,
            stdin=subprocess.PIPE, stdout=subprocess.PIPE, stderr=subprocess.PIPE,
            text=True, bufsize=1, start_new_session=True,
        )
        self._reader = threading.Thread(target=self._read_loop, daemon=True, name="dsh-reader")
        self._reader.start()
        threading.Thread(target=self._drain_stderr, daemon=True, name="dsh-stderr").start()
        self.session_id = "s-" + uuid.uuid4().hex[:12]
        from .._retry import retry_transient
        # 进程起来后插件还在加载（含 pi-ai 的 provider 注册），太早发 initialize 会收到
        # "no adapter registered for provider …"——不是配置错，是没加载完。本机磁盘快
        # 常常赢下这个竞争，pod 里 1 秒就撞上。把它当瞬时错误重试，给足加载时间
        # （6 次 ≈ 5+15×4 s）。真配置错（拼错 provider 名）会在重试用尽后原样抛出。
        res = retry_transient(
            lambda: self._request("initialize", {
                "cwd": str(self._workspace),
                "provider": self._provider,
                "model": self._model,
            }, timeout=120),
            label="dsh initialize", attempts=6,
            extra_transient=("no adapter registered",),
            on_retry=lambda m: self._emit(AgentEvent(type="info", content=m)))
        info = (res or {}).get("serverInfo") or {}
        self._emit(AgentEvent(type="info",
                              content=f"dsh sdk {info.get('name')} {info.get('version')} "
                                      f"provider={self._provider} model={self._model} "
                                      f"session_id: {self.session_id}"))

    def _request(self, method: str, params: dict | None, *, timeout: float) -> dict | None:
        with self._lock:
            self._rpc_id += 1
            rid = self._rpc_id
            frame = {"jsonrpc": "2.0", "id": rid, "method": method}
            if params is not None:
                frame["params"] = params
            assert self._proc and self._proc.stdin
            self._proc.stdin.write(json.dumps(frame, ensure_ascii=False) + "\n")
            self._proc.stdin.flush()
        deadline = time.monotonic() + timeout
        while time.monotonic() < deadline:
            if rid in self._replies:
                rep = self._replies.pop(rid)
                if "error" in rep:
                    raise RuntimeError(f"dsh {method} 失败: {rep['error']}")
                return rep.get("result")
            if self._proc.poll() is not None:
                raise RuntimeError(f"dsh 进程已退出 (code={self._proc.returncode}): "
                                   f"{''.join(self._stderr)[-400:]}")
            time.sleep(0.05)
        raise TimeoutError(f"dsh {method} 在 {timeout}s 内无响应")

    def _read_loop(self) -> None:
        assert self._proc and self._proc.stdout
        for line in self._proc.stdout:
            line = line.strip()
            if not line:
                continue
            try:
                d = json.loads(line)
            except json.JSONDecodeError:
                continue                       # 协议规定：坏行忽略
            self._last_activity = time.monotonic()
            if "method" not in d and "id" in d:
                self._replies[d["id"]] = d
                continue
            m = d.get("method")
            p = d.get("params") or {}
            if m == "session.event" and p.get("sessionId") == self.session_id:
                ev = p.get("event") or {}
                self._events.append(ev)
                self._on_session_event(ev)

    def _drain_stderr(self) -> None:
        assert self._proc and self._proc.stderr
        for line in self._proc.stderr:
            self._stderr.append(line)
            if len(self._stderr) > 200:
                del self._stderr[:100]

    def _on_session_event(self, ev: dict) -> None:
        """把 dsh 会话日志转成 AgentEvent（只转有信息量的几类）。"""
        t = ev.get("type") or ""
        data = ev.get("data") or {}
        if t == "tool/call":
            self._emit(AgentEvent(type="tool_call", tool=str(data.get("name") or data.get("tool") or t),
                                  tool_input=json.dumps(data, ensure_ascii=False)[:500]))
        elif t == "tool/result":
            self._emit(AgentEvent(type="tool_result", tool=str(data.get("name") or t),
                                  content=json.dumps(data, ensure_ascii=False)[:300],
                                  is_error=bool(data.get("isError") or data.get("error"))))
        elif t == "request/header":
            model = ((data.get("header") or {}).get("config") or {}).get("model")
            if model:
                self._effective_models.add(f"{self._provider}/{model}")
        elif t == "turn/end":
            reason = (data.get("reason") or {})
            kind = reason.get("kind")
            if kind not in (None, "completed"):
                self._emit(AgentEvent(type="info",
                                      content=f"[dsh] turn/end reason={json.dumps(reason, ensure_ascii=False)[:300]}"))
            # max-tokens / blocked are agent terminal outcomes.  Provider or
            # protocol errors and unexplained interruption are operational
            # failures and must not enter score/usage averages as normal runs.
            if kind in {"error", "aborted", "interrupted"} and self.killed_reason is None:
                self.degraded_reason = f"dsh turn/end: {json.dumps(reason, ensure_ascii=False)[:500]}"

        self._record_usage_event(ev)

    def _record_usage_event(self, ev: dict) -> None:
        """Fold dsh's streaming/final usage samples into one record per step."""
        t = ev.get("type") or ""
        data = ev.get("data") or {}
        key = (data.get("turn"), data.get("step"))
        keyed = key != (None, None)

        if t == "assistant/chunk":
            chunk = data.get("chunk") or {}
            if chunk.get("type") == "usage":
                u = extract_usage(chunk)
                if u is not None and keyed:
                    self._usage_pending[key] = u
            return

        if t == "assistant/message":
            u = extract_usage(data.get("usage"))
            if u is not None and keyed:
                # Final sample replaces the earlier streaming sample.
                self._usage_pending[key] = u
            return

        if t == "step/end" and keyed:
            self._commit_pending_usage(key)
            return

        if t == "turn/end":
            turn = data.get("turn")
            for pending_key in list(self._usage_pending):
                if turn is None or pending_key[0] == turn:
                    self._commit_pending_usage(pending_key)
            return

        # Compaction and future non-step model events may carry genuine usage
        # of their own.  Preserve it rather than restricting accounting to the
        # two assistant event shapes above.
        u = extract_usage(ev)
        if u is not None:
            self._usage_ledger.record(u, elapsed_s=None)
            self._last_call_tokens = self._usage_ledger.last_call_tokens
            self._saw_usage = True

    def _commit_pending_usage(self, key: tuple[object, object]) -> None:
        u = self._usage_pending.pop(key, None)
        if u is None:
            return
        self._usage_ledger.record(u, elapsed_s=None)
        self._last_call_tokens = self._usage_ledger.last_call_tokens
        self._saw_usage = True

    # ── 一轮对话 ─────────────────────────────────────────────────────────

    async def send(self, message: str, *, stop_on_tools: set[str] | None = None,
                   fork: bool = False) -> str:
        self._note_agent_send()
        return await asyncio.to_thread(self._send, message)

    def _send(self, message: str) -> str:
        if self._proc is None or self._proc.poll() is not None:
            raise RuntimeError("dsh 进程未启动或已退出")
        max_runtime = int(getattr(self._config, "max_runtime_s", None) or 43200)
        idle_timeout = int(getattr(self._config, "idle_timeout_s", None) or max_runtime)
        n0 = len(self._events)
        self._saw_usage = False
        started = time.monotonic()
        if self._runtime_deadline is None:
            self._runtime_deadline = started + max_runtime
        self._last_activity = started
        remaining = self._runtime_deadline - started
        if remaining <= 0:
            self.killed_reason = "max_runtime"
            self._kill()
            return self._collect_text(n0, started)
        from .._retry import retry_transient
        # 只重试「入队回执」这一步（enqueue 失败 = 消息没进会话，重发不会重复）；
        # 入队成功后模型调用的失败由 dsh 自己的 dsh-llm-retry 处理
        retry_transient(
            lambda: self._request("session/prompt", {
                "sessionId": self.session_id,
                "contentBlocks": [{"type": "text", "text": message}],
            }, timeout=min(120, max(0.1, self._runtime_deadline - time.monotonic()))),
            label="dsh session/prompt",
            on_retry=lambda m: self._emit(AgentEvent(type="info", content=m)))

        # 等本轮 turn/end。看门狗：总时长 / 静默时长，超了 killpg 并记 killed_reason。
        while True:
            for ev in self._events[n0:]:
                if ev.get("type") == "turn/end":
                    return self._collect_text(n0, started)
            if self._proc.poll() is not None:
                # A spontaneous process exit is an infrastructure/backend
                # failure, not a run-budget timeout.
                self.degraded_reason = self.degraded_reason or "dsh process_exit"
                self._emit(AgentEvent(type="info",
                                      content=f"[dsh] 进程退出 code={self._proc.returncode}: "
                                              f"{''.join(self._stderr)[-300:]}"))
                return self._collect_text(n0, started)
            now = time.monotonic()
            reason = ("max_runtime" if now >= self._runtime_deadline else
                      "idle" if now - self._last_activity > idle_timeout else None)
            if reason:
                self.killed_reason = reason
                self._emit(AgentEvent(type="info",
                                      content=f"[watchdog] {reason} (elapsed={now - started:.0f}s "
                                              f"idle={now - self._last_activity:.0f}s) — kill dsh"))
                self._kill()
                return self._collect_text(n0, started)
            time.sleep(0.3)

    def _collect_text(self, n0: int, started: float) -> str:
        texts: list[str] = []
        for ev in self._events[n0:]:
            if ev.get("type") != "assistant/message":
                continue
            msg = (ev.get("data") or {}).get("message") or {}
            parts = [b.get("text", "") for b in (msg.get("content") or [])
                     if isinstance(b, dict) and b.get("type") == "text"]
            joined = "\n".join(p for p in parts if p).strip()
            if joined:
                texts.append(joined)
        if getattr(self, "_saw_usage", False) and self._usage_ledger._calls:
            # 本轮墙钟记在本轮最后一条 usage 上，phase 秒数按 send 归属
            self._usage_ledger._calls[-1]["elapsed_s"] = round(time.monotonic() - started, 1)
        # 与 codex/opencode 同口径：本轮最后一条 assistant 文本即回复
        return texts[-1] if texts else ""

    # ── 资源 ───────────────────────────────────────────────────────────────

    def _kill(self) -> None:
        if self._proc and self._proc.poll() is None:
            try:
                os.killpg(self._proc.pid, 15)
                self._proc.wait(timeout=10)
            except Exception:                  # noqa: BLE001
                try:
                    os.killpg(self._proc.pid, 9)
                except Exception:              # noqa: BLE001
                    pass

    async def close(self) -> None:
        if self._proc and self._proc.poll() is None:
            try:
                self._request("shutdown", None, timeout=15)
            except Exception:                  # noqa: BLE001
                pass
            self._kill()
        shutil.rmtree(self._home, ignore_errors=True)

    # ── usage 契约 ──────────────────────────────────────────────────────────

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


class DeepSeekHarnessRunner(AgentRunner):
    async def start(self, workspace: Path) -> DeepSeekHarnessSession:
        s = DeepSeekHarnessSession(workspace, self.config)
        await asyncio.to_thread(s.start)
        return s


# 全称为正名；"dsh" 是命令行同名别名，两个键指向同一个 Runner
AGENTS = {"deepseek-harness": DeepSeekHarnessRunner, "dsh": DeepSeekHarnessRunner}
