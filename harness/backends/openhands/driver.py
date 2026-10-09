"""在 .venv-openhands 内执行的 OpenHands 驱动。

从 stdin 读 JSON 配置，跑 run_controller，把结果以单行 JSON 写到 stdout
最后一行。由同目录的 session.py 以子进程方式调用——这层子进程的唯一目的是
隔离 rich 版本冲突（详见 openhands/__init__.py）。
"""

from __future__ import annotations

import asyncio
import json
import re
import time
import sys
import traceback
from pathlib import Path


def _build_config(payload: dict):
    from openhands.core.config import LLMConfig, OpenHandsConfig
    from openhands.core.config.utils import load_openhands_config

    try:
        config = load_openhands_config()
    except Exception:
        config = OpenHandsConfig()

    config.workspace_base = payload["workspace"]
    config.runtime = payload["runtime"]
    config.max_iterations = payload["max_iterations"]
    if payload.get("trajectory"):
        # 字段名随版本变：0.x 是 trajectories_path，1.x 是 save_trajectory_path。
        # 直接赋一个不存在的字段，pydantic 抛 ValueError，driver 在建配置这一步就崩，
        # 整个后端每个 run 都 degraded（2026-09-02 实测 1.6.0 就是这样，且错误被吞）。
        for name in ("save_trajectory_path", "trajectories_path"):
            if name in getattr(type(config), "model_fields", {}):
                setattr(config, name, payload["trajectory"])
                break

    model = payload.get("model")
    if model:
        # 沿用 opencode 那套凭证，保证两后端打同一个模型端点。
        import os
        max_output_tokens = int(payload.get("max_output_tokens") or 131072)
        llm_kwargs = dict(
            model=model,
            api_key=(payload.get("api_key") or os.environ.get("DELIVER_ANSWERER_TOKEN")
                     or os.environ.get("OPENHANDS_API_KEY", "")),
            base_url=payload.get("base_url") or os.environ.get("DELIVER_ANSWERER_BASE_URL") or None,
            # FDE-bench 的模型都走支持原生 tool_calls 的 OpenAI 兼容网关。
            # OpenHands 1.6 不认识 deepseek-v4-pro/qwen3.8-max/kimi-k3 等新名称，
            # 会错误退回 XML 工具模拟，并因 synthetic security_risk 必填项反复失败。
            native_tool_calling=True,
            timeout=payload.get("llm_timeout_s") or 3600,
            # This is independent of opencode's model registry.  Without an
            # explicit value OpenHands/LiteLLM has used a 65536-token fallback
            # even though the formal model configuration declares 131072.
            max_output_tokens=max_output_tokens,
        )
        if payload.get("force_max_tokens"):
            # OpenHands 1.6 builds max_completion_tokens from the field above.
            # The configured direct endpoint may ignore that alias.
            # OpenHands merges completion_kwargs last, so adding max_tokens here
            # reaches LiteLLM and the provider unchanged. The endpoint accepts both
            # fields together and gives max_tokens precedence.
            llm_kwargs["completion_kwargs"] = {
                "max_tokens": max_output_tokens,
            }
        llm = LLMConfig(**llm_kwargs)
        if payload.get("omit_sampling_params"):
            # Some Responses backends reject both fields, even when a client
            # sends the framework defaults rather than an explicit user value.
            # OpenHands 1.6's Pydantic schema rejects None in the constructor,
            # so validate the config first and then bypass assignment validation.
            # LiteLLM treats None as "argument not supplied" when it builds the
            # Responses request.
            object.__setattr__(llm, "temperature", None)
            object.__setattr__(llm, "top_p", None)
        config.set_llm_config(llm)

    return config


_ASK_SUFFIX = ".ask"
_ANS_SUFFIX = ".ans"
_ASK_TIMEOUT_S = 900


def _message_text(content) -> list[str]:
    """Extract text from a LiteLLM/OpenAI message without assuming one schema.

    The Responses bridge has returned both strings and content-part lists across
    LiteLLM releases.  OpenHands' legacy parser accepts a string or a list of
    ``{"type": "text", "text": ...}`` parts, so normalising to plain text is
    the least surprising representation here.
    """
    if isinstance(content, str):
        return [content] if content else []
    if isinstance(content, list):
        out: list[str] = []
        for part in content:
            if isinstance(part, str):
                if part:
                    out.append(part)
                continue
            if isinstance(part, dict):
                text = part.get("text")
            else:
                text = getattr(part, "text", None)
            if isinstance(text, str) and text:
                out.append(text)
        return out
    return []


def _coalesce_response_choices(response) -> int:
    """Collapse a Responses-bridge response to OpenHands' one-choice contract.

    LiteLLM 1.77.x converts every Responses output item/content part into a
    separate ChatCompletion choice.  OpenHands 1.6's function-calling parser
    asserts ``len(response.choices) == 1``.  Preserve every text part and tool
    call while exposing one assistant message to that parser.  Return the
    original choice count for diagnostics (0/1 means no normalisation).
    """
    choices = getattr(response, "choices", None)
    if not choices or len(choices) <= 1:
        return len(choices or [])

    first = choices[0]
    first_message = getattr(first, "message", None)
    if first_message is None:
        return len(choices)

    texts: list[str] = []
    tool_calls = []
    finish_reasons: list[str] = []
    for choice in choices:
        message = getattr(choice, "message", None)
        if message is None:
            continue
        texts.extend(_message_text(getattr(message, "content", None)))
        calls = getattr(message, "tool_calls", None)
        if calls:
            tool_calls.extend(list(calls))
        reason = getattr(choice, "finish_reason", None)
        if reason:
            finish_reasons.append(str(reason))

    # LiteLLM's Message/Choices are mutable Pydantic models in the supported
    # release.  Mutating the first object avoids depending on private model
    # constructors and also preserves provider-specific fields.
    first_message.content = "\n".join(texts) if texts else None
    first_message.tool_calls = tool_calls or None
    if getattr(first_message, "role", None) in (None, ""):
        first_message.role = "assistant"
    first.finish_reason = "tool_calls" if tool_calls else (finish_reasons[-1] if finish_reasons else "stop")
    first.index = 0
    response.choices = [first]
    return len(choices)


def _install_responses_bridge_compat(stats: dict):
    """Patch LiteLLM's Responses bridge for OpenHands 1.6 in this subprocess.

    Returns an uninstall callback.  The import is intentionally optional so
    tests and non-Responses backends can import this driver without LiteLLM.
    """
    try:
        from litellm.completion_extras.litellm_responses_transformation.transformation import (
            LiteLLMResponsesTransformationHandler,
        )
    except Exception:  # pragma: no cover - exercised only in the OpenHands venv
        return lambda: None

    original = LiteLLMResponsesTransformationHandler.transform_response

    def transform_response(handler, *args, **kwargs):
        response = original(handler, *args, **kwargs)
        count = _coalesce_response_choices(response)
        if count > 1:
            stats["responses_choices_normalized"] = stats.get("responses_choices_normalized", 0) + 1
            stats["responses_max_choices"] = max(stats.get("responses_max_choices", 1), count)
        return response

    LiteLLMResponsesTransformationHandler.transform_response = transform_response

    def restore() -> None:
        LiteLLMResponsesTransformationHandler.transform_response = original

    return restore


def _install_chat_streaming_compat(stats: dict, sidecar: str = ""):
    """Use SSE on the wire while preserving OpenHands 1.6's ModelResponse API.

    Patch only this driver's OpenHands completion alias, before LLM instances
    are created. Assembly stays inside OpenHands' existing retry boundary, so
    a broken stream cannot execute a partial tool call. No sampling, token,
    timeout, or retry budgets are changed. The sidecar contains counters only,
    never prompts, response text, tool arguments, or credentials.
    """
    import importlib
    import litellm

    llm_module = importlib.import_module("openhands.llm.llm")
    original = llm_module.litellm_completion
    stats.update({"enabled": True, "started": 0, "completed": 0, "failed": 0})
    last_write = 0.0

    def publish(force: bool = False) -> None:
        nonlocal last_write
        now = time.monotonic()
        if not sidecar or (not force and now - last_write < 5):
            return
        try:
            path = Path(sidecar)
            tmp = path.with_name(path.name + ".tmp")
            tmp.write_text(json.dumps(stats), encoding="utf-8")
            tmp.replace(path)
            last_write = now
        except OSError:
            pass  # diagnostics must not interrupt a valid completion

    def completion(*args, **kwargs):
        request = dict(kwargs)
        request["stream"] = True
        request["stream_options"] = {
            **(request.get("stream_options") or {}), "include_usage": True,
        }
        stats["started"] += 1
        stats.update({"state": "connecting", "request_started_at": time.time(),
                      "first_chunk_at": None, "last_chunk_at": None,
                      "chunks": 0, "finish_reason": None, "last_error_type": None})
        publish(True)
        stream = None
        chunks = []
        terminal = False
        try:
            stream = original(*args, **request)
            for chunk in stream:
                chunks.append(chunk)
                now = time.time()
                stats["first_chunk_at"] = stats["first_chunk_at"] or now
                stats.update({"state": "receiving", "last_chunk_at": now,
                              "chunks": len(chunks)})
                for choice in chunk.get("choices", []) or []:
                    reason = choice.get("finish_reason")
                    if reason is not None:
                        terminal = True
                        stats["finish_reason"] = reason
                publish(len(chunks) == 1)
            if not terminal:
                raise litellm.APIConnectionError(
                    message="OpenHands stream ended without a finish_reason",
                    llm_provider="openai", model=str(request.get("model", "")),
                )
            response = litellm.stream_chunk_builder(chunks, messages=request.get("messages"))
            if response is None or not getattr(response, "choices", None):
                raise litellm.APIConnectionError(
                    message="OpenHands stream could not be assembled into a response",
                    llm_provider="openai", model=str(request.get("model", "")),
                )
            # Prefer authoritative provider usage, including cache/reasoning
            # details, over the builder's fallback token estimate.
            usage = next((chunk.get("usage") for chunk in reversed(chunks)
                          if chunk.get("usage") is not None), None)
            if usage is not None:
                response.usage = litellm.Usage(**usage) if isinstance(usage, dict) else usage
            stats["completed"] += 1
            stats.update({"state": "completed", "request_finished_at": time.time(),
                          "provider_usage_received": usage is not None})
            publish(True)
            return response
        except Exception as exc:
            stats["failed"] += 1
            stats.update({"state": "failed", "last_error_type": type(exc).__name__,
                          "request_finished_at": time.time()})
            publish(True)
            raise
        finally:
            close = getattr(stream, "close", None)
            if callable(close):
                try:
                    close()
                except Exception:
                    pass

    llm_module.litellm_completion = completion

    def restore() -> None:
        llm_module.litellm_completion = original

    return restore


def _finish_requests_client(action) -> bool:
    """Whether an OpenHands ``finish`` is really a clarification request.

    OpenHands 1.6 documents ``finish`` as valid both for completed work and for
    follow-up questions.  Its controller nevertheless maps every finish to
    ``FINISHED``, so ``fake_user_response_fn`` is never called in the latter
    case.  Interact prompts give agents a soft, optional heading; accept that
    high-precision signal and a conservative natural-language fallback.
    """
    text = str(getattr(action, "final_thought", "") or "")
    if not text.strip():
        return False
    if re.search(r"(?im)^\s*#{1,6}\s*questions?\s+for\s+the\s+client\s*[:：]?\s*$",
                 text):
        return True
    if not re.search(r"[?？]", text):
        return False
    return bool(re.search(
        r"(?i)\b(?:need|require|request)(?:\s+some)?\s+clarification\b|"
        r"\b(?:questions?|details?)\s+for\s+the\s+client\b|"
        r"(?:需要|想要|请)(?:向您)?(?:确认|澄清|询问).{0,24}(?:问题|细节|口径|信息)",
        text,
    ))


def _progress_message(action) -> bool:
    """识别不应进入等待用户状态的高置信度进度旁白。"""
    text = str(getattr(action, "content", "") or "").strip()
    if not text or re.search(r"[?？]", text):
        return False
    if re.search(r"(?i)\b(?:need|require|request)\s+(?:some\s+)?clarification\b", text):
        return False
    return bool(re.search(
        r"(?i)^(?:okay[,!. ]*|sure[,!. ]*|all right[,!. ]*)?\s*"
        r"(?:i(?:'ll| will| am going to)|let me|i(?:'m| am) (?:starting|beginning))\s+"
        r"(?:start\s+by\s+|begin\s+by\s+|first\s+)?"
        r"(?:explor|inspect|examin|check|look|read|list|analy[sz]|review|work|proceed)",
        text,
    ))


_TOOL_SCHEMA_ERROR_PATTERNS = (
    "Missing required parameters for function",
    "Missing required argument",
    "is expected to be one of",
    "is not allowed for function",
    "Unexpected argument",
    "Failed to parse tool call arguments",
)


def _history_diagnostics(history) -> dict:
    """跨 OpenHands 小版本提取终态诊断，不依赖序列化格式。"""
    schema_errors: list[str] = []
    error_count = 0
    finish_count = 0
    reject_count = 0
    agent_messages = 0
    for event in history or []:
        name = type(event).__name__
        source = str(getattr(event, "source", "") or "").lower()
        is_agent = source.endswith("agent") or source in {"", "none"}
        content = str(getattr(event, "content", "") or getattr(event, "message", "") or "")
        if name == "AgentFinishAction" and is_agent:
            finish_count += 1
        elif name == "AgentRejectAction" and is_agent:
            reject_count += 1
        elif name == "MessageAction" and is_agent:
            agent_messages += 1
        if name == "ErrorObservation" or "errorobservation" in name.lower():
            error_count += 1
            if any(pattern in content for pattern in _TOOL_SCHEMA_ERROR_PATTERNS):
                schema_errors.append(content[:300])
    tool_names = _tool_action_names(history)
    return {
        "error_count": error_count,
        "tool_schema_error_count": len(schema_errors),
        "tool_schema_error_samples": schema_errors[:3],
        "finish_count": finish_count,
        "reject_count": reject_count,
        "agent_message_count": agent_messages,
        "tool_action_count": len(tool_names),
    }


def _final_agent_message(history) -> str:
    """只取真正的 Agent 回复，不能把 recall/error observation 当答案。"""
    for event in reversed(history or []):
        name = type(event).__name__
        source = str(getattr(event, "source", "") or "").lower()
        if ((source and not source.endswith("agent"))
                or name not in {"MessageAction", "AgentFinishAction", "AgentRejectAction"}):
            continue
        content = (getattr(event, "content", None)
                   or getattr(event, "final_thought", None)
                   or getattr(event, "message", None))
        if content:
            return str(content)
    return ""


def _make_answer_fn(channel: Path, notes: list[str]):
    """实时应答：把 agent 的提问写给父进程，等它回。

    为什么不能用预置答案列表：answerer 扮演业务方，必须看到 agent **实际问了
    什么**才能作答——预置意味着提前知道它会问什么，那就不是澄清了。

    走文件而非管道：driver 跑在独立 venv 的子进程里（OpenHands 需 rich>=14，
    与本机全局 rich 13.7.1 冲突），stdin/stdout 已被 payload 与结果 JSON 占用。

    协议：第 i 轮把提问写进 <channel>.<i>.ask，然后轮询 <channel>.<i>.ans。
    父进程（session.py 的 _serve_clarify）看到 .ask 就调 answerer 并写 .ans。
    """
    state = {"i": 0, "exhausted": False}

    def fn(controller_state) -> str:
        i = state["i"]
        state["i"] = i + 1
        question = _last_agent_message(controller_state)

        ask = channel.with_suffix(f".{i}{_ASK_SUFFIX}")
        ans = channel.with_suffix(f".{i}{_ANS_SUFFIX}")
        ask.write_text(question or "", encoding="utf-8")
        notes.append(f"clarify turn {i + 1}: asked ({len(question or '')} chars)")

        deadline = time.time() + _ASK_TIMEOUT_S
        while time.time() < deadline:
            if ans.exists():
                reply = ans.read_text(encoding="utf-8")
                if "No further information is available" in reply:
                    state["exhausted"] = True
                notes.append(f"clarify turn {i + 1}: answer received")
                return reply
            time.sleep(0.3)

        notes.append(f"clarify turn {i + 1}: timed out waiting for answer")
        state["exhausted"] = True
        return "No further information is available. Proceed with your best judgement."

    fn.can_intercept_finish = lambda: not state["exhausted"]
    return fn


def _last_agent_message(controller_state) -> str:
    """从 controller state 里取 agent 最后说的话（即它的提问）。"""
    for event in reversed(getattr(controller_state, "history", []) or []):
        if getattr(event, "source", None) == "user":
            continue
        text = getattr(event, "content", None) or getattr(event, "final_thought", None)
        if text and str(text).strip():
            return str(text)
    return ""


def _event_usage(event):
    """Best-effort extraction from OpenHands event objects.

    OpenHands has changed event classes across releases; usage may be a dict
    attribute or nested under ``metadata``/``llm_response``.  Keep this
    optional and return None when the installed version exposes no metadata.
    """
    candidates = [event]
    for attr in ("usage", "usage_metadata", "metadata", "llm_response", "response"):
        try:
            value = getattr(event, attr, None)
        except Exception:
            value = None
        if value is not None:
            candidates.append(value)
    keys = {
        "input": ("input_tokens", "prompt_tokens", "promptTokenCount"),
        "output": ("output_tokens", "completion_tokens", "candidatesTokenCount"),
        "reasoning": ("reasoning_tokens", "reasoningTokenCount"),
        "cache_read": ("cached_tokens", "cache_read_input_tokens", "cache_read"),
        "cache_write": ("cache_creation_input_tokens", "cache_write"),
    }
    for candidate in candidates:
        if not isinstance(candidate, dict):
            try:
                candidate = vars(candidate)
            except TypeError:
                continue
        out = {}
        for field, names in keys.items():
            val = next((candidate.get(k) for k in names if candidate.get(k) is not None), None)
            try:
                out[field] = int(val) if val is not None else None
            except (TypeError, ValueError):
                out[field] = None
        if any(v is not None for v in out.values()):
            return out
    return None


def _event_model(event):
    for attr in ("model", "model_name", "model_id", "provider_model"):
        value = getattr(event, attr, None)
        if isinstance(value, str) and value.strip():
            provider = getattr(event, "provider", None) or getattr(event, "provider_id", None)
            return f"{provider}/{value}" if provider and "/" not in value else value
    return None


async def _run(payload: dict) -> dict:
    from openhands.core.main import run_controller
    from openhands.events.action import MessageAction

    notes: list[str] = []
    config = _build_config(payload)
    stream_stats: dict = {"enabled": bool(payload.get("stream"))}
    if payload.get("stream") and payload.get("use_responses_api"):
        raise ValueError("OpenHands streaming currently supports Chat Completions only")

    answers_file = payload.get("clarify_answers") or ""
    answer_fn = _make_answer_fn(Path(answers_file), notes) if answers_file else None

    # 截获 run_controller 内部建的 ConversationStats：OpenHands 1.x 的逐 LLM 调用用量
    # 攒在它的 service_to_metrics 里（每个 LLM 服务一个 Metrics，token_usages 逐条），
    # run_controller 只返回 state，state.metrics 在 CLI runtime 下实测为空。
    captured: dict = {}
    import openhands.core.main as _ohmain
    from openhands.controller.agent_controller import AgentController
    from openhands.core.schema import AgentState
    from openhands.events.action import AgentFinishAction, MessageAction
    _orig_create = _ohmain.create_registry_and_conversation_stats
    _orig_handle_action = AgentController._handle_action

    def _capture(*a, **kw):
        out = _orig_create(*a, **kw)
        try:
            captured["stats"] = out[1]
        except Exception:                       # noqa: BLE001
            pass
        return out

    _ohmain.create_registry_and_conversation_stats = _capture

    protocol = {"progress_repeats": {}, "failure_reason": ""}
    bridge_stats: dict = {"responses_choices_normalized": 0, "responses_max_choices": 1}
    restore_bridge = _install_responses_bridge_compat(bridge_stats)
    restore_stream = (_install_chat_streaming_compat(
        stream_stats, payload.get("stream_sidecar") or "")
        if payload.get("stream") else lambda: None)
    raw_exception: dict = {}
    _orig_step = AgentController._step

    async def _step_with_trace(controller, *args, **kwargs):
        try:
            return await _orig_step(controller, *args, **kwargs)
        except Exception as exc:  # noqa: BLE001 - preserve the hidden root cause
            raw_exception.update({
                "type": type(exc).__name__,
                "message": str(exc)[:2000],
                "traceback": traceback.format_exc(limit=40)[-8000:],
            })
            raise

    async def _handle_action_with_finish_clarification(controller, action):
        # In OpenHands 1.6 a finish carrying questions bypasses
        # AWAITING_USER_INPUT entirely.  Convert only high-confidence question
        # finishes; ordinary completion finishes retain native semantics.
        can_intercept = (answer_fn is not None
                         and answer_fn.can_intercept_finish())
        if (can_intercept and isinstance(action, AgentFinishAction)
                and _finish_requests_client(action)):
            notes.append("clarification request intercepted from finish action")
            await controller.set_agent_state_to(AgentState.AWAITING_USER_INPUT)
            return
        if (isinstance(action, MessageAction)
                and bool(getattr(action, "wait_for_response", False))
                and _progress_message(action)):
            # Kimi 等模型会先说“我先检查目录”，OpenHands 1.6 却把所有纯消息
            # 都当成向用户提问。进度旁白直接续跑，重复三次则结束协议空转。
            action.wait_for_response = False
            normalized = re.sub(r"\s+", " ", str(action.content or "").strip().lower())
            repeats = protocol["progress_repeats"]
            repeats[normalized] = repeats.get(normalized, 0) + 1
            notes.append("progress narration continued without user round-trip")
            if repeats[normalized] >= 3:
                protocol["failure_reason"] = "repeated_progress_message"
                notes.append("repeated progress narration detected; stopping protocol loop")
                await controller.set_agent_state_to(AgentState.ERROR)
                return
        await _orig_handle_action(controller, action)

    AgentController._handle_action = _handle_action_with_finish_clarification
    AgentController._step = _step_with_trace

    # 用量 sidecar：driver 只在结束时打印结果，被 max_runtime 掐掉就全丢（实测 1200s
    # 超时 → tokens None）。每 5s 把当前 token_usages 落到 run 目录，session 超时时读它。
    sidecar = payload.get("usage_sidecar") or ""

    def _dump_sidecar() -> None:
        stats = captured.get("stats")
        if not sidecar or stats is None:
            return
        models: set = set()
        events = _metrics_usage(stats, models, [])
        tmp = Path(sidecar + ".tmp")
        tmp.write_text(json.dumps({"usage_events": events, "effective_models": sorted(models)}),
                       encoding="utf-8")
        tmp.replace(Path(sidecar))

    async def _sidecar_loop():
        while True:
            await asyncio.sleep(5)
            try:
                _dump_sidecar()
            except Exception:                   # noqa: BLE001  落盘失败不影响 agent
                pass

    dumper = asyncio.create_task(_sidecar_loop()) if sidecar else None
    try:
        state = await run_controller(
            config=config,
            initial_user_action=MessageAction(content=payload["task"]),
            exit_on_message=bool(payload.get("exit_on_message")) and answer_fn is None,
            fake_user_response_fn=answer_fn,
            headless_mode=True,
        )
    finally:
        _ohmain.create_registry_and_conversation_stats = _orig_create
        AgentController._handle_action = _orig_handle_action
        AgentController._step = _orig_step
        restore_bridge()
        restore_stream()
        if dumper is not None:
            dumper.cancel()
        try:
            _dump_sidecar()
        except Exception:                       # noqa: BLE001
            pass

    final = ""
    usage_events = []
    effective_models = set()
    agent_state = "missing"
    diagnostics = _history_diagnostics([])
    diagnostics["streaming"] = stream_stats
    if state is not None:
        agent_state = str(getattr(state.agent_state, "value", state.agent_state)).lower()
        notes.append(f"openhands final state: {agent_state}")
        history = getattr(state, "history", []) or []
        diagnostics = _history_diagnostics(history)
        diagnostics["streaming"] = stream_stats
        diagnostics["last_error"] = str(getattr(state, "last_error", "") or "")[:1000]
        diagnostics.update(bridge_stats)
        if raw_exception:
            diagnostics["raw_exception"] = raw_exception
        final = _final_agent_message(history)
        # 用量：权威来源是 state.metrics.token_usages —— OpenHands 1.x 每次 LLM 调用
        # 追加一条 TokenUsage(model, prompt_tokens, completion_tokens, cache_read_tokens,
        # cache_write_tokens, …)。litellm 口径：prompt_tokens **含** cache_read，
        # session.py 的账本按 input_includes_cache_read=True 归一。
        # 修前只从事件属性里猜 usage/metadata 字段（1.x 不这么放），而且和"找最终
        # 回复"共用一个循环、在第一条回复处就 break，等于从没扫过 → OpenHands 的
        # tokens 恒 None。
        usage_events = _metrics_usage(captured.get("stats") or state, effective_models, notes)
        if not usage_events:
            for event in getattr(state, "history", []) or []:
                usage = _event_usage(event)
                if usage is not None:
                    usage_events.append(usage)
                model = _event_model(event)
                if model:
                    effective_models.add(model)

    # 工具调用：保留 Action 的真实类名。旧实现只返回一个整数，父进程只能制造
    # N 个 ``openhands-action`` 并全部归为 shell，文件编辑/浏览因此无法与其他
    # 脚手架比较。仍保留 tool_calls 整数供旧消费者兼容。
    tool_names = _tool_action_names(getattr(state, "history", []) if state is not None else [])

    failure_reason = str(protocol.get("failure_reason") or "")
    last_error = diagnostics.get("last_error", "").lower()
    limit_reason = ""
    if ("maximum iteration" in last_error or "maximum budget" in last_error
            or "max iteration" in last_error):
        limit_reason = "max_turns"
    valid_terminal = (
        (agent_state == "finished" and diagnostics["finish_count"] > 0)
        or (agent_state == "rejected" and diagnostics["reject_count"] > 0)
    )
    if not failure_reason and not limit_reason and not valid_terminal:
        failure_reason = f"terminal_state_{agent_state}"
    schema_errors = diagnostics["tool_schema_error_count"]
    if (not failure_reason and schema_errors >= 10
            and schema_errors >= 5 * max(1, diagnostics["tool_action_count"])):
        failure_reason = "tool_schema_validation_loop"
    if failure_reason:
        notes.append(f"openhands protocol failure: {failure_reason}")

    return {"final_message": final, "notes": notes, "sid": getattr(config, "sid", None),
            "usage_events": usage_events,
            "effective_models": sorted(effective_models),
            "tool_calls": len(tool_names), "tool_names": tool_names,
            "agent_state": agent_state, "diagnostics": diagnostics,
            "failure_reason": failure_reason, "limit_reason": limit_reason}


def _tool_action_names(history) -> list[str]:
    """提取 agent 真正执行的 OpenHands Action 类名，不计消息/思考/结束动作。"""
    ignored = {"MessageAction", "SystemMessageAction", "AgentFinishAction",
               "AgentRejectAction", "AgentThinkAction", "ThinkAction", "NullAction",
               "AgentDelegateAction", "ChangeAgentStateAction", "LoopRecoveryAction",
               "CondensationRequestAction"}
    out: list[str] = []
    for event in history or []:
        name = type(event).__name__
        if (name.endswith("Action")
                and str(getattr(event, "source", "")).lower().endswith("agent")
                and name not in ignored):
            out.append(name)
    return out


def _metrics_usage(source, effective_models: set, notes: list[str]) -> list[dict]:
    """ConversationStats / State → 逐调用 usage 列表（键名沿用 litellm，账本认识）。

    source 优先是截获的 ConversationStats（get_combined_metrics 合并所有 LLM 服务），
    退化为 State（get_local_metrics / metrics）。
    """
    metrics = None
    for getter in ("get_combined_metrics", "get_local_metrics", "metrics"):
        try:
            m = getattr(source, getter, None)
            metrics = m() if callable(m) else m
        except Exception:                       # noqa: BLE001
            metrics = None
        if metrics is not None and (getattr(metrics, "token_usages", None) or getter == "metrics"):
            break
    if metrics is None:
        return []
    usages = getattr(metrics, "token_usages", None) or []
    out: list[dict] = []
    for tu in usages:
        d = tu.model_dump() if hasattr(tu, "model_dump") else dict(getattr(tu, "__dict__", {}) or {})
        rec = {"prompt_tokens": d.get("prompt_tokens"),
               "completion_tokens": d.get("completion_tokens"),
               "cache_read_tokens": d.get("cache_read_tokens"),
               "cache_write_tokens": d.get("cache_write_tokens")}
        if any(v is not None for v in rec.values()):
            out.append(rec)
        if d.get("model"):
            effective_models.add(str(d["model"]))
    cost = getattr(metrics, "accumulated_cost", None)
    if cost:
        notes.append(f"openhands accumulated_cost={cost}")   # litellm 自估，仅留痕，不进账本
    return out


def main() -> int:
    payload = json.loads(sys.stdin.read())
    try:
        result = asyncio.run(_run(payload))
    except Exception as exc:
        print(f"driver error: {type(exc).__name__}: {exc}", file=sys.stderr)
        return 1
    print(json.dumps(result, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    sys.exit(main())
