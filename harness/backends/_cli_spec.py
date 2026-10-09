"""headless CLI 脚手架的共用机制。

这里只放**机制**，不放具体 agent——每个 agent 一个包（见 backends/ 的目录列表），
在自己的 `__init__.py` 里声明 `_Spec` 与 `AGENTS`。这样 `ls backends/` 就是
「支持哪些 agent」的答案，加一个 agent 也只是加一个文件夹。

为什么这些 agent 能共用一套机制：headless 下它们都不注册 ask 类工具
（实测 2026-08-13），agent 缺信息时在回复正文里自然语言提问并停下，
澄清一律走文本多轮（见 clarify/loop.py 的 _run_text）。差异只在命令行，
故抽成声明式的 `_Spec`。
"""

from __future__ import annotations

import asyncio
import json
import os
import re
import urllib.request
import subprocess
import uuid
import time
from dataclasses import dataclass, field
from pathlib import Path

from .base import AgentEvent, AgentRunner, AgentSession
from .usage import UsageLedger, extract_model, extract_usage, structured_event


@dataclass(frozen=True)
class _Spec:
    name: str
    bin: str
    first: tuple[str, ...]          # 首轮参数模板
    resume: tuple[str, ...]         # 续接参数模板；{sid}/{model}/{message} 会被填充
    needs_session_id: bool = True   # 是否由我们生成 UUID
    via_stdin: bool = False         # 消息经 stdin 传（避开 `-` 开头被当 flag）
    env: dict = field(default_factory=dict)
    noise: tuple[str, ...] = ()     # 要剔除的 CLI 自身提示行
    # stdout 格式：text（纯文本，无 usage/工具事件）| claude-stream-json |
    # gemini-stream-json | kimi-stream-json。JSON 格式下回复文本、usage、
    # 工具调用都从事件流里取；实测（2026-09-02）三家 stream-json 形态各异：
    #   claude  result.usage.iterations[] 逐 API 调用（Anthropic 口径 input 不含缓存）
    #   gemini  result.stats 整轮聚合：input（不含缓存）/ input_tokens（含）/ cached
    #   kimi    完全没有 usage 字段，tokens 只能诚实记 None
    output: str = "text"



_SESSION_LINE = re.compile(r"^\s*(To resume this session:|Approval mode).*$")


class _ApiClient:
    """从 API response 提取 usage，不依赖脚手架 CLI 的 stdout 格式。

    适用于 gemini/kimi 等 CLI 吞掉 response metadata 的脚手架。
    用法：backend 在 send() 里遇到需要追踪 token 时调用
    _ApiClient.fetch(system, user) 获取文本 + usage 字段。
    """

    def __init__(self, base_url: str, api_key: str, model: str, *, protocol: str = "openai"):
        self._base_url = base_url.rstrip("/")
        self._api_key = api_key
        self._model = model
        self._protocol = protocol

    def fetch(self, messages: list[dict], *, system: str = "") -> tuple[str, dict]:
        """返回 (response_text, usage_dict)。"""
        if self._protocol == "gemini":
            contents = [{"role": "user" if m["role"] == "user" else "model",
                         "parts": [{"text": m["content"]}]} for m in messages]
            payload = {"contents": contents, "generationConfig": {}}
            if system:
                payload["systemInstruction"] = {"parts": [{"text": system}]}
            req = urllib.request.Request(
                f"{self._base_url}/v1beta/models/{self._model}:generateContent",
                data=json.dumps(payload).encode(), method="POST",
                headers={"x-goog-api-key": self._api_key, "Content-Type": "application/json"})
            with urllib.request.urlopen(req, timeout=120) as r:
                d = json.loads(r.read().decode())
            text = d.get("candidates", [{}])[0].get("content", {}).get("parts", [{}])[0].get("text", "")
            usage = d.get("usageMetadata", {})
            return text, usage
        # OpenAI 兼容（kimi / codex 通过网关调时也走这个）
        full = ([{"role": "system", "content": system}] if system else []) + messages
        req = urllib.request.Request(
            f"{self._base_url}/chat/completions",
            data=json.dumps({"model": self._model, "messages": full}).encode(),
            method="POST",
            headers={"Authorization": f"Bearer {self._api_key}",
                     "Content-Type": "application/json"})
        with urllib.request.urlopen(req, timeout=120) as r:
            d = json.loads(r.read().decode())
        text = d.get("choices", [{}])[0].get("message", {}).get("content", "")
        usage = d.get("usage", {})
        return text, usage




def _usage_sum(u: dict) -> int:
    """从 usage dict 提取总 token 数（兼容 OpenAI 和 Gemini 两种格式）。"""
    if "total_tokens" in u: return u["total_tokens"]
    if "totalTokenCount" in u: return u["totalTokenCount"]
    return (u.get("input_tokens", 0) + u.get("output_tokens", 0) +
            u.get("prompt_tokens", 0) + u.get("completion_tokens", 0))


class _ApiClient:
    """从 API response 提取 usage，不依赖 CLI 的 stdout 格式。

    gemini: response 有 usageMetadata（promptTokenCount / candidatesTokenCount）
    kimi:   response 有 usage（prompt_tokens / completion_tokens）
    两者都走华颜网关，用同一套认证。
    """

    def __init__(self, base_url, api_key, model, *, protocol="openai"):
        self._base = base_url.rstrip("/")
        self._key = api_key
        self._model = model
        self._protocol = protocol

    def fetch(self, messages, *, system=""):
        if self._protocol == "gemini":
            contents = [{"role":"user" if m["role"]=="user" else "model",
                         "parts":[{"text":m["content"]}]} for m in messages]
            payload = {"contents": contents, "generationConfig": {}}
            if system:
                payload["systemInstruction"] = {"parts": [{"text": system}]}
            req = urllib.request.Request(
                f"{self._base}/v1beta/models/{self._model}:generateContent",
                data=json.dumps(payload).encode(), method="POST",
                headers={"x-goog-api-key": self._key, "Content-Type": "application/json"})
            with urllib.request.urlopen(req, timeout=120) as r:
                d = json.loads(r.read().decode())
            text = d.get("candidates",[{}])[0].get("content",{}).get("parts",[{}])[0].get("text","")
            usage = d.get("usageMetadata", {})
            return text, usage
        full = ([{"role":"system","content":system}] if system else []) + messages
        req = urllib.request.Request(
            f"{self._base}/chat/completions",
            data=json.dumps({"model":self._model,"messages":full}).encode(),
            method="POST",
            headers={"Authorization":f"Bearer {self._key}","Content-Type":"application/json"})
        with urllib.request.urlopen(req, timeout=120) as r:
            d = json.loads(r.read().decode())
        text = d.get("choices",[{}])[0].get("message",{}).get("content","")
        usage = d.get("usage", {})
        return text, usage

class CliAgentSession(AgentSession):
    """一次 send = 一次 CLI 调用（首轮）或续接调用（后续轮）。"""

    supports_native_clarify = False     # headless 无 ask 工具，走文本多轮

    def __init__(self, spec: _Spec, workspace: Path, config=None):
        self._spec = spec
        self._workspace = workspace
        self._config = config
        self.session_id: str | None = None
        self.on_event = None
        self.killed_reason: str | None = None
        self._started = False
        self._api_client = None
        # Only structured metadata emitted by the CLI is recorded.  Older
        # versions used [pre]/[post] probe requests; those measured extra
        # requests rather than the agent conversation and are intentionally
        # no longer used.
        # 账本口径见 base.AgentSession.input_includes_cache_read。
        # claude：Anthropic input_tokens 不含缓存；gemini：我们取 stats.input（已不含缓存）；
        # kimi：stdout 无 usage，但 wire.jsonl 的 input_other 已是不含缓存口径。
        # 三家账本最终都按「不含 cache_read」入账，flag 一律 False。
        self.input_includes_cache_read = False
        self.reports_tool_calls = spec.output != "text"
        self._usage_ledger = UsageLedger(input_includes_cache_read=self.input_includes_cache_read)
        self._last_call_tokens = None
        self._effective_models: set[str] = set()
        # ``--timeout`` is a run-level budget.  Text clarification performs
        # several send() calls; giving every call a fresh full timeout lets the
        # outer asyncio watchdog fire first and discard the final call's
        # metadata.  Start one shared deadline on the first send instead.
        self._runtime_deadline: float | None = None

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
        spec = self._spec
        max_runtime = int(getattr(self._config, "max_runtime_s", None) or 43200)
        model = _strip_provider(getattr(self._config, "model", "") or "", spec)

        if spec.needs_session_id and self.session_id is None:
            self.session_id = str(uuid.uuid4())

        def _build():
            """每次尝试重新拼命令：首轮失败后重试可能已切到续接模板。"""
            tmpl = spec.resume if self._started else spec.first
            args = [a.format(sid=self.session_id or "", model=model, message=message)
                    for a in tmpl]
            cmd = [spec.bin, *args]
            # 模板没占位 message 的（claude 把 prompt 放末尾），补在最后。
            # 用 `--` 分隔：澄清答案以 "- 问题" 形式回灌，开头的 `-` 会被当成参数
            # （Codex 实测 exit=2 "unexpected argument"，其余 CLI 同理）。
            stdin_data = None
            if spec.via_stdin:
                stdin_data = message
            elif "{message}" not in "".join(tmpl):
                cmd += ["--", message]
            return cmd, stdin_data

        env = {**os.environ, **spec.env, **self._credential_env()}
        started = time.monotonic()
        if self._runtime_deadline is None:
            self._runtime_deadline = started + max_runtime

        def _run_once():
            from ._proc import run_pg
            cmd, stdin_data = _build()
            remaining = self._runtime_deadline - time.monotonic()
            if remaining <= 0:
                raise subprocess.TimeoutExpired(cmd, max_runtime)
            proc = run_pg(cmd, timeout=remaining, cwd=str(self._workspace),
                          env=env, input=stdin_data)
            self._persist_cli_logs(proc.stdout, proc.stderr, proc.returncode)
            if proc.returncode != 0:
                # A partial assistant message is not completion when the CLI
                # reports a terminal transport/provider failure.  In
                # particular kimi-cli can emit an assistant chunk and then
                # exit=1 with an overload response; accepting that chunk
                # loses the retry and leaves an unfinished workspace marked
                # successful.
                # Kimi prints its resume hint on stderr but the actual
                # overload on stdout. Never let harmless stderr hide failure.
                tail = ((proc.stderr or "")[-800:] + "\n" +
                        (proc.stdout or "")[-800:]).strip()
                failure_tail = tail.lower()
                if spec.name == "kimi" and "To resume this session:" in (proc.stderr or ""):
                    self._started = True
                if spec.name == "gemini" and any(
                        d.get("type") == "init" and d.get("session_id") == self.session_id
                        for d in _json_lines(proc.stdout)):
                    self._started = True
                if any(marker in failure_tail for marker in (
                    "currently overloaded", "terminated", "fetch failed",
                    "connection reset", "socket closed")):
                    raise RuntimeError(
                        f"{spec.name} exit={proc.returncode}: "
                        f"{tail}"
                    )
                if _stream_completed(proc.stdout, spec):
                    # 流里已有"完成"的终止事件（claude result / gemini result:success /
                    # kimi 最终 assistant），退出码却非零——实测 claude-code 跑 36 分钟、
                    # 产物齐全、result 里 terminal_reason=completed，进程仍 exit=1（原因不明，
                    # hook 或收尾请求）。这种要按成功处理，否则整 run 记 error、usage 丢光。
                    # 完整 stdout/stderr 已落盘 .cli_logs/，退出码原因留给事后查。
                    self._emit(AgentEvent(type="info",
                                          content=f"[{spec.name}] exit={proc.returncode} 但流里有完成事件，"
                                                  f"按成功处理（stderr 见 .cli_logs/）"))
                    return proc
                if (("already in use" in tail or "already exists" in tail)
                        and not self._started):
                    # 首轮因网关抖动失败但会话已建好（claude：`Session ID … is already
                    # in use`）——重试必须切到 --resume，否则永远撞这个错
                    self._started = True
                # 交给 retry_transient 判：网关 5xx / 连接中断重试，其余直接抛
                raise RuntimeError(f"{spec.name} exit={proc.returncode}: {tail}")
            return proc

        try:
            from ._retry import retry_transient
            proc = retry_transient(
                _run_once, label=f"{spec.name} send",
                extra_transient=("already in use", "already exists"),
                on_retry=lambda m: self._emit(AgentEvent(type="info", content=m)))
        except subprocess.TimeoutExpired as exc:
            # 保留已捕获的 stdout：超时不代表 agent 什么都没说。直接返回空串会让
            # response.txt 为空、澄清记录看起来是 0 轮，掩盖它实际问过什么
            # （实测 claude-code / kimi 超时后 response.txt 全空、无从判断）。
            self.killed_reason = "max_runtime"
            partial = exc.stdout or b""
            if isinstance(partial, bytes):
                partial = partial.decode(errors="replace")
            self._record_metadata(partial, "", time.monotonic() - started)
            return _clean(partial, spec)

        except RuntimeError as exc:
            # 重试用尽或永久错误
            self._emit(AgentEvent(type="tool_result", is_error=True, content=str(exc)[:400]))
            raise

        self._record_metadata(proc.stdout, proc.stderr, time.monotonic() - started)
        self._started = True
        return _clean(proc.stdout, spec)

    def _persist_cli_logs(self, stdout: str, stderr: str, returncode: int) -> None:
        """每次 send 的完整 stdout / stderr 落到 run_dir/.cli_logs/（隐藏目录，不算产物）。

        此前失败只留 400 字尾巴，事后查不出退出码的根因（如 claude exit=1 却已完成）。
        stream-json 可能有几 MB，但一个 run 只有几次 send，存得起。
        """
        try:
            logdir = Path(self._workspace).parent / ".cli_logs"
            logdir.mkdir(parents=True, exist_ok=True)
            n = getattr(self, "_send_no", 0) + 1
            self._send_no = n
            base = logdir / f"{self._spec.name}_send{n:02d}"
            (base.with_suffix(".stdout.jsonl")).write_text(stdout or "", encoding="utf-8")
            (base.with_suffix(".stderr.txt")).write_text(
                f"# exit={returncode}\n" + (stderr or ""), encoding="utf-8")
        except OSError:
            pass

    def _credential_env(self) -> dict:
        """按脚手架生成子进程的凭据与运行时配置，让四个 CLI 都指向同一网关。

        运行时凭据和用户配置不应依赖宿主机隐式状态，必须显式透传。
        实测网关同时提供 Anthropic /v1/messages 与 OpenAI /v1/responses，都能指过去。

          claude  ANTHROPIC_BASE_URL（网关根）+ ANTHROPIC_AUTH_TOKEN；环境已有则不覆盖
          gemini  独立 HOME/settings.json 固定 gemini-api-key 认证，并注入
                  GEMINI_API_KEY + GOOGLE_GEMINI_BASE_URL（网关根，不带 /v1）
          kimi    每 session 一个 KIMI_SHARE_DIR，写 config.toml 注册 provider 与模型
                  （兼容旧 KIMI_CODE_HOME；max_context_size 262144）
        """
        from ._retry import gateway
        configured_model = getattr(self._config, "model", "") or ""
        base_v1, key = gateway(configured_model)
        root = base_v1[:-3] if base_v1.endswith("/v1") else base_v1
        out: dict = {}
        name = self._spec.name
        if name == "gemini":
            # 新版 Gemini CLI 看到 GOOGLE_GEMINI_BASE_URL 时会从环境推断出 gateway
            # auth type，但它的非交互 validateAuthMethod 又不接受 gateway，启动即报
            # "Invalid auth method selected"。显式选择 gemini-api-key 可绕开该 CLI
            # 内部矛盾；底层客户端仍会读取 GOOGLE_GEMINI_BASE_URL 访问相同网关。
            # 用独立 HOME 隔离设置与会话，避免镜像或运行者 ~/.gemini 污染实验。
            out["HOME"] = str(self._gemini_home())
            if key and not os.environ.get("GEMINI_API_KEY"):
                out["GEMINI_API_KEY"] = key
            if not os.environ.get("GOOGLE_GEMINI_BASE_URL"):
                out["GOOGLE_GEMINI_BASE_URL"] = root
        elif name == "claude-code":
            if key and not os.environ.get("ANTHROPIC_AUTH_TOKEN") and not os.environ.get("ANTHROPIC_API_KEY"):
                out["ANTHROPIC_AUTH_TOKEN"] = key
                out["ANTHROPIC_BASE_URL"] = root
        elif name == "kimi":
            home = str(self._kimi_home(base_v1, key))
            # kimi-cli 1.49.0 使用 KIMI_SHARE_DIR；旧版本使用 KIMI_CODE_HOME。
            # 两者指向同一独占目录，既适配镜像锁定版本，也保留旧版本兼容性。
            out["KIMI_SHARE_DIR"] = home
            out["KIMI_CODE_HOME"] = home
        return out

    def _gemini_home(self) -> Path:
        """生成仅属于当前 session 的 Gemini CLI HOME 与认证选择。"""
        home = getattr(self, "_runtime_home", None)
        if home is None:
            import tempfile
            home = Path(tempfile.mkdtemp(prefix="gemini_home_"))
            self._runtime_home = home
            settings = home / ".gemini" / "settings.json"
            settings.parent.mkdir(parents=True)
            settings.write_text(
                json.dumps(
                    {"security": {"auth": {"selectedType": "gemini-api-key"}}},
                    indent=2,
                ) + "\n",
                encoding="utf-8",
            )
        return home

    def _kimi_home(self, base_v1: str, key: str) -> Path:
        """为本 session 生成独立的 kimi 主目录（provider + 模型注册）。"""
        home = getattr(self, "_runtime_home", None)
        if home is None:
            import tempfile
            home = Path(tempfile.mkdtemp(prefix="kimi_home_"))
            self._runtime_home = home
            model = getattr(self._config, "model", "") or ""
            from ..model_endpoints import is_responses_model
            responses = is_responses_model(model)
            provider = "responses" if responses else "chat"
            provider_type = "openai_responses" if responses else "openai_legacy"
            provider_name = "Benchmark responses" if responses else "Benchmark chat"
            alias = model if "/" in model else f"{provider}/{model}"
            bare = alias.split("/", 1)[1]
            (home / "config.toml").write_text(
                'default_permission_mode = "auto"\n\n'
                # kimi-cli 1.49.0 将旧 provider type "openai" 拆成
                # openai_legacy（/chat/completions）与 openai_responses。
                f'[providers.{provider}]\ntype = "{provider_type}"\nname = "{provider_name}"\n'
                f'base_url = "{base_v1}"\napi_key = "{key}"\n\n'
                f'[models."{alias}"]\nprovider = "{provider}"\nmodel = "{bare}"\n'
                'max_context_size = 262144\n',
                encoding="utf-8")
        return home

    def _record_metadata(self, stdout: str, stderr: str, elapsed_s: float) -> None:
        """Extract usage objects from JSONL/JSON CLI output, if present."""
        if self._spec.output == "claude-stream-json":
            return self._record_claude(stdout, elapsed_s)
        if self._spec.output == "gemini-stream-json":
            return self._record_gemini(stdout, elapsed_s)
        if self._spec.output == "kimi-stream-json":
            return self._record_kimi(stdout, elapsed_s)
        found = []
        for line in (stdout or "").splitlines() + (stderr or "").splitlines():
            line = line.strip()
            if not line.startswith(("{", "[")):
                continue
            try:
                payload = json.loads(line)
                got = extract_usage(payload)
                model = extract_model(payload)
                if model:
                    self._effective_models.add(model)
                event = structured_event(payload)
                if event:
                    self._emit(AgentEvent(type=event["type"], tool=event["tool"],
                                          is_error=event["is_error"],
                                          content=("structured CLI error" if event["is_error"] else "")))
            except (TypeError, json.JSONDecodeError):
                got = None
            if got is not None:
                found.append(got)
        # A stream can repeat cumulative usage on every event.  Prefer the
        # final metadata object; per-call ledger still accumulates calls.
        if found:
            self._usage_ledger.record(found[-1], elapsed_s=elapsed_s)
            self._last_call_tokens = self._usage_ledger.last_call_tokens

    # ── 三家 stream-json 的专用抽取 ─────────────────────────────────────
    def _record_claude(self, stdout: str, elapsed_s: float) -> None:
        """claude：usage 只取 result 事件的总量；工具调用从 assistant content 的
        tool_use block 数；模型名从 message.model 取。"""
        result = None
        for d in _json_lines(stdout):
            t = d.get("type")
            if t == "assistant":
                msg = d.get("message") or {}
                if msg.get("model"):
                    self._effective_models.add(str(msg["model"]))
                for b in msg.get("content") or []:
                    if isinstance(b, dict) and b.get("type") == "tool_use":
                        self._emit(AgentEvent(type="tool_call", tool=str(b.get("name") or "?"),
                                              tool_input=json.dumps(b.get("input") or {}, ensure_ascii=False)[:500]))
            elif t == "user":
                for b in ((d.get("message") or {}).get("content") or []):
                    if isinstance(b, dict) and b.get("type") == "tool_result" and b.get("is_error"):
                        self._emit(AgentEvent(type="tool_result", tool="", is_error=True,
                                              content=str(b.get("content"))[:300]))
            elif t == "result":
                result = d
        if not result:
            # 进程被超时掐掉：result 没落地，但已经流出来的 assistant 事件各带一份该
            # message 的 usage 快照。按 message.id 去重取最后一份求和，作为**下界估计**
            # 入账并标 partial——比整段 None 有用（费用/阶段统计至少有下界），
            # 又不冒充精确值。
            per_msg: dict[str, dict] = {}
            for d in _json_lines(stdout):
                if d.get("type") == "assistant":
                    msg = d.get("message") or {}
                    if msg.get("id") and isinstance(msg.get("usage"), dict):
                        per_msg[msg["id"]] = msg["usage"]
            if per_msg:
                agg: dict = {}
                for u in per_msg.values():
                    for k, v in u.items():
                        if isinstance(v, (int, float)) and not isinstance(v, bool):
                            agg[k] = agg.get(k, 0) + v
                self._usage_ledger.record(agg, elapsed_s=elapsed_s, partial=True)
                self._last_call_tokens = self._usage_ledger.last_call_tokens
            return
        # result.usage 是本次 send 的**总量**（权威）。它带的 iterations[] 实测只含
        # 最后一次调用（2 次调用只列 1 条），不是完整明细，不能拿来当 step 用；
        # assistant 事件里的 usage 是流式快照、同一 message 重复出现，也不可靠。
        # 所以每次 send 记一条总量，steps = send 数（与 codex 同粒度）。claude 没有
        # 分档定价，step 粒度不影响费用。
        usage = {k: v for k, v in (result.get("usage") or {}).items() if k != "iterations"}
        # result.num_turns = 本次 send 的模型调用数 → steps 仍按「模型调用」计，与
        # opencode / codex / kimi / dsh 同单位（实测 2 次调用 num_turns=2）
        n_calls = result.get("num_turns") or 1
        self._usage_ledger.record(usage, elapsed_s=elapsed_s, n_calls=int(n_calls))
        self._last_call_tokens = self._usage_ledger.last_call_tokens

    def _record_gemini(self, stdout: str, elapsed_s: float) -> None:
        """gemini：result.stats 是整轮聚合（steps=1/轮，分档定价按聚合判档会偏高，
        这是 gemini CLI 不给逐调用明细的限制，记进 precision 字段）。
        stats.input 已不含缓存，与 opencode 口径一致；cached 是命中数。"""
        for d in _json_lines(stdout):
            t = d.get("type")
            if t == "tool_use":
                self._emit(AgentEvent(type="tool_call", tool=str(d.get("tool_name") or "?"),
                                      tool_input=json.dumps(d.get("parameters") or {}, ensure_ascii=False)[:500]))
            elif t == "result":
                stats = d.get("stats") or {}
                for m in (stats.get("models") or {}):
                    self._effective_models.add(str(m))
                u = {"input": stats.get("input"), "output_tokens": stats.get("output_tokens"),
                     "cache_read": stats.get("cached")}
                if any(v is not None for v in u.values()):
                    # gemini 只给整轮聚合（telemetry 本地文件实测也不产出）：aggregate=True →
                    # steps_unit=send；分档定价按聚合判档会偏高，报表据此标注
                    self._usage_ledger.record(u, elapsed_s=elapsed_s, aggregate=True)
                    self._last_call_tokens = self._usage_ledger.last_call_tokens

    def _record_kimi(self, stdout: str, elapsed_s: float = 0.0) -> None:
        """kimi：stream-json 的 stdout 没有 usage，但会话目录里有——
        1.49.0 的 KIMI_SHARE_DIR/sessions/<hash>/<uuid>/wire.jsonl 用 StatusUpdate
        保存逐 step token；旧版 ~/.kimi-code 的 usage.record 也继续兼容。
        session_id 尽量从 stdout 的 session.resume_hint 取，超时缺 hint 时依靠当前
        session 的独占 share dir 定位。工具调用从 assistant.tool_calls[] 数。"""
        sid = None
        for d in _json_lines(stdout):
            if d.get("role") == "assistant":
                for c in d.get("tool_calls") or []:
                    fn = (c.get("function") or {}) if isinstance(c, dict) else {}
                    self._emit(AgentEvent(type="tool_call", tool=str(fn.get("name") or "?"),
                                          tool_input=str(fn.get("arguments") or "")[:500]))
            elif d.get("type") == "session.resume_hint" and d.get("session_id"):
                sid = str(d["session_id"])
        if sid:
            self._kimi_session_id = sid
        sid = sid or getattr(self, "_kimi_session_id", None)
        home = Path(getattr(self, "_runtime_home", None)
                    or os.environ.get("KIMI_SHARE_DIR")
                    or os.environ.get("KIMI_CODE_HOME")
                    or (Path.home() / ".kimi"))
        if sid:
            # 1.49.0: sessions/<workdir-hash>/<uuid>/wire.jsonl
            # 旧版本: sessions/<workdir-hash>/<session-id>/agents/main/wire.jsonl
            hits = sorted((home / "sessions").glob(f"*/{sid}/wire.jsonl"))
            hits += sorted((home / "sessions").glob(f"*/{sid}/agents/main/wire.jsonl"))
        else:
            # resume_hint 在 stdout 最后一行，被超时掐掉就拿不到 sid。本 session 的
            # share dir 是独占的，里面只有我们这一个会话，直接取最新 wire.jsonl。
            hits = list((home / "sessions").glob("*/*/wire.jsonl"))
            hits += list((home / "sessions").glob("*/session_*/agents/main/wire.jsonl"))
            hits = sorted(hits, key=lambda x: x.stat().st_mtime)
            if hits and getattr(self, "_runtime_home", None) is None:
                hits = []           # 共享的 ~/.kimi-code 里不能瞎猜别人的会话
        if not hits:
            return
        records = []
        for d in _json_lines(hits[-1].read_text(encoding="utf-8", errors="replace")):
            # Model identity may live on a request/config event rather than on
            # the StatusUpdate that carries usage.  Preserve the runtime alias
            # when present; do not depend on the final resume_hint.
            model = d.get("modelAlias") or d.get("model")
            msg = d.get("message") if isinstance(d.get("message"), dict) else None
            payload = (msg.get("payload") or {}) if msg else {}
            model = model or payload.get("modelAlias") or payload.get("model")
            if isinstance(model, str) and model:
                self._effective_models.add(model)
            # 旧版 kimi-code：{"type":"usage.record","model":…,"usage":{inputOther,…}}
            if d.get("type") == "usage.record" and isinstance(d.get("usage"), dict):
                records.append(d)
                continue
            # kimi-cli 1.49（镜像里的）：{"timestamp":…,"message":{"type":"StatusUpdate",
            #   "payload":{"token_usage":{input_other,output,input_cache_read,input_cache_creation},…}}}
            # 每个 LLM step 一条，2026-09-04 镜像内实测
            if msg and msg.get("type") == "StatusUpdate":
                tu = payload.get("token_usage")
                if isinstance(tu, dict):
                    records.append({"usage": tu, "model": payload.get("model")})
        # kimi-cli 1.49 StatusUpdate may omit the model entirely.  Once a
        # usage record proves that the isolated CLI actually made a request,
        # its explicit --model plus single-model private config is reliable
        # runtime evidence (an unknown model makes Kimi fail, it does not
        # silently fall back).  This avoids losing effective_models on timeout.
        if records and not self._effective_models:
            configured = getattr(self._config, "model", None)
            if isinstance(configured, str) and configured:
                self._effective_models.add(configured)
        consumed = getattr(self, "_kimi_consumed", 0)
        fresh = records[consumed:]
        for i, r in enumerate(fresh):
            if r.get("model"):
                self._effective_models.add(str(r["model"]))
            elif fresh:
                # kimi-cli 1.49 的 StatusUpdate 不带模型名；用我们写进 config 的别名兜底
                m = getattr(self._config, "model", "") or ""
                if m:
                    self._effective_models.add(m if "/" in m else f"responses/{m}")
            self._usage_ledger.record(r["usage"], elapsed_s=elapsed_s if i == len(fresh) - 1 else None)
        self._kimi_consumed = len(records)
        if fresh:
            self._last_call_tokens = self._usage_ledger.last_call_tokens

    def start(self):
        spec = self._spec
        if spec.name in ('gemini','kimi'):
            # 取 key：config.toml 优先，环境变量兜底
            # get() 缺字段时不抛异常（只返回 None），所以不能用 except 分支
            from ..config import get
            key = get('agent','api_key',None) or os.environ.get('OPENAI_API_KEY')
            if key:
                base = os.environ.get(
                    'DELIVER_GEMINI_BASE_URL' if spec.name == 'gemini'
                    else 'DELIVER_AGENT_BASE_URL',
                    'https://generativelanguage.googleapis.com' if spec.name == 'gemini'
                    else 'https://api.openai.com/v1',
                )
                proto = 'gemini' if spec.name=='gemini' else 'openai'
                model = _strip_provider(getattr(self._config,'model',''), spec)
                self._api_client = _ApiClient(base, key, model, protocol=proto)

    async def close(self) -> None:
        home = getattr(self, "_runtime_home", None)
        if home:
            import shutil
            shutil.rmtree(home, ignore_errors=True)


def _stream_completed(out: str, spec: _Spec) -> bool:
    """事件流里是否已有"本轮正常结束"的终止事件（与退出码无关）。"""
    if spec.output == "claude-stream-json":
        for d in _json_lines(out):
            if d.get("type") == "result":
                return (d.get("terminal_reason") == "completed"
                        or d.get("subtype") == "success" or d.get("is_error") is False)
        return False
    if spec.output == "gemini-stream-json":
        return any(d.get("type") == "result" and d.get("status") == "success"
                   for d in _json_lines(out))
    if spec.output == "kimi-stream-json":
        # Kimi has no explicit success event here. Assistant chunks (often
        # followed by tools) do not override a nonzero process exit.
        return False
    return False


def _json_lines(out: str):
    for line in (out or "").splitlines():
        line = line.strip()
        if not line.startswith("{"):
            continue
        try:
            yield json.loads(line)
        except json.JSONDecodeError:
            continue


def _reply_from_stream(out: str, spec: _Spec) -> str:
    """从 stream-json 事件流里取本轮 agent 的最终回复文本。

    与 codex/opencode 同口径：本轮**最后一条** assistant 文本。gemini 是 delta 流，
    要把连续的 assistant delta 拼起来再取最后一段。
    """
    if spec.output == "claude-stream-json":
        final = None
        texts: list[str] = []
        for d in _json_lines(out):
            if d.get("type") == "result" and isinstance(d.get("result"), str):
                final = d["result"]
            elif d.get("type") == "assistant":
                blocks = ((d.get("message") or {}).get("content") or [])
                t = "\n".join(b.get("text", "") for b in blocks
                              if isinstance(b, dict) and b.get("type") == "text").strip()
                if t:
                    texts.append(t)
        return (final or (texts[-1] if texts else "")).strip()
    if spec.output == "gemini-stream-json":
        segments: list[str] = []
        cur: list[str] = []
        for d in _json_lines(out):
            if d.get("type") == "message" and d.get("role") == "assistant":
                cur.append(str(d.get("content") or ""))
            elif cur:
                segments.append("".join(cur)); cur = []
        if cur:
            segments.append("".join(cur))
        segments = [x.strip() for x in segments if x.strip()]
        return segments[-1] if segments else ""
    if spec.output == "kimi-stream-json":
        texts: list[str] = []
        for d in _json_lines(out):
            if d.get("role") != "assistant" or not d.get("content"):
                continue
            c = d["content"]
            # kimi-cli 1.49 的 content 可能是 block 列表（think / text）；只取 text 块，
            # 别把 think 块或整个列表 str() 进回复
            if isinstance(c, list):
                c = "\n".join(str(b.get("text", "")) for b in c
                              if isinstance(b, dict) and b.get("type") == "text").strip()
            if isinstance(c, str) and c.strip():
                texts.append(c.strip())
        return texts[-1] if texts else ""
    return _clean_text(out, spec)


def _clean(out: str, spec: _Spec) -> str:
    return _reply_from_stream(out, spec) if spec.output != "text" else _clean_text(out, spec)


def _clean_text(out: str, spec: _Spec) -> str:
    """剔除 CLI 自身的提示行——它们不是 agent 的话，会污染 ask_detect 判定。"""
    lines = []
    for line in (out or "").splitlines():
        if any(n in line for n in spec.noise):
            continue
        if _SESSION_LINE.match(line):
            continue
        lines.append(line.rstrip())
    return "\n".join(lines).strip()


def _strip_provider(model: str, spec: _Spec) -> str:
    """provider 前缀的处理各家不同。

    kimi 的模型键包含 provider 前缀（例如 `responses/kimi-k3`），
    要保留前缀；claude / gemini 只认裸模型名。
    """
    if spec.name == "kimi":
        return model
    return model.split("/", 1)[-1] if "/" in model else model



def make_runner(spec: _Spec):
    """由 _Spec 造一个 AgentRunner 子类。各 agent 包在自己的 __init__ 里调用。"""
    class _Runner(AgentRunner):
        async def start(self, workspace: Path) -> CliAgentSession:
            workspace.mkdir(parents=True, exist_ok=True)
            return CliAgentSession(spec, workspace, config=self.config)
    _Runner.__name__ = spec.name.replace("-", "_").title() + "Runner"
    return _Runner
