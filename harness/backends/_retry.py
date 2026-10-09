"""脚手架子进程/本地 RPC 层的瞬时故障重试。

LLM 直连（answerer / judge / detect）的重试在 llm_client.ChatLLM.complete。这里覆盖
**agent 本身的 send**：CLI 进程因网关 5xx / 连接中断退出非零、dsh SDK 握手超时、
opencode serve 本地 HTTP 抖动。修前这些都直接 raise → 整个 run 记 error、零产物，
与"agent 能力不足"混在一起；批量跑几百个 run 时网关抖动是常态。

判据与 ChatLLM 共用同一份签名表：瞬时错误重试，永久错误（模型不存在 / 鉴权 /
配额）立刻放弃。默认 3 次尝试，退避 5s / 15s + 抖动。

**不重试的边界**：超时（max_runtime / idle）不是瞬时故障，是 agent 跑太久，重试只会
再跑 12 小时；opencode 的 POST /message 已经把 prompt 送进会话，盲重会重复用户消息。
"""

from __future__ import annotations

import random
import subprocess
import time
from typing import Callable, TypeVar

from ..llm_client import ChatLLM
from ..model_endpoints import endpoint

T = TypeVar("T")

ATTEMPTS = 3
_BACKOFF_S = (5, 15)


def is_transient(msg: str) -> bool:
    """网关/连接类瞬时错误 → True；永久错误或不认识的错误 → False。"""
    m = (msg or "").lower()
    if any(sig.lower() in m for sig in ChatLLM._NEVER_RETRY):
        return False
    return any(sig.lower() in m for sig in ChatLLM._RETRY_ON) or any(
        sig in m for sig in ("currently overloaded", "typeerror: terminated",
                             "fetch failed", "connection reset", "socket closed",
                             "other side closed", "und_err_socket"))


# 永不重试的异常类型：子进程超时 = agent 跑太久，不是网关抖动。它的消息
# "Command ... timed out after N seconds" 会命中签名表里的 "timed out"，
# 不按类型拦下来就会把一个 12 小时的 run 重跑 3 次。
NEVER_RETRY_TYPES: tuple[type[BaseException], ...] = (subprocess.TimeoutExpired,)


def retry_transient(fn: Callable[[], T], *, label: str, attempts: int = ATTEMPTS,
                    on_retry: Callable[[str], None] | None = None,
                    never: tuple[type[BaseException], ...] = NEVER_RETRY_TYPES,
                    extra_transient: tuple[str, ...] = ()) -> T:
    """跑 fn；抛出的异常文本命中瞬时签名就退避重试，最多 attempts 次。

    never 里的异常类型原样抛出、不重试。on_retry(消息) 在每次重试前调用，
    供后端发 info 事件留痕——重试本身要可见，否则"跑成功了"和"第三次才成功"
    在结果里分不开。
    """
    last: Exception | None = None
    for i in range(attempts):
        try:
            return fn()
        except never:
            raise
        except Exception as exc:                          # noqa: BLE001
            msg = f"{type(exc).__name__}: {exc}"
            transient = is_transient(msg) or any(sig in msg for sig in extra_transient)
            if i == attempts - 1 or not transient:
                raise
            last = exc
            wait = _BACKOFF_S[min(i, len(_BACKOFF_S) - 1)] + random.uniform(0, 3)
            if on_retry:
                on_retry(f"[retry] {label} 第 {i + 1}/{attempts} 次失败（{msg[:160]}），{wait:.0f}s 后重试")
            time.sleep(wait)
    raise last  # pragma: no cover  上面 raise 已覆盖


# ── 网关凭据（各 CLI 后端生成运行时配置时共用）────────────────────────────
def gateway(model: str = "") -> tuple[str, str]:
    """返回 (OpenAI 兼容 base_url 含 /v1, api_key)。

    优先 config.toml [agent].base_url / api_key，其次环境变量中的通用端点与密钥。
    所有脚手架都指向同一网关同一 key，跨脚手架的对照才不掺入端点差异。
    """
    import os
    if model.startswith(("responses/", "direct/")):
        return endpoint(model)
    try:
        from ..config import get
        base = str(get("agent", "base_url", None) or
                   os.environ.get("DELIVER_AGENT_BASE_URL", "https://api.openai.com/v1"))
        key = get("agent", "api_key", None)
    except Exception:                          # noqa: BLE001
        base, key = os.environ.get("DELIVER_AGENT_BASE_URL", "https://api.openai.com/v1"), None
    key = key or os.environ.get("DELIVER_AGENT_API_KEY") or os.environ.get("OPENAI_API_KEY") or ""
    return base.rstrip("/"), key
