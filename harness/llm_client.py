"""
harness/llm_client.py — 辅助 LLM 客户端，兼容 Anthropic、OpenAI Chat 与 Responses。

answerer / simulator 共用：把"建 client + 发请求 + 取文本"按 protocol 收在一处。
- anthropic: anthropic.Anthropic(auth_token, base_url).messages.create(system=, messages=)
- openai:    通常走 chat.completions；指定的 Responses 模型走 responses.create

history(user/assistant)形状两协议通用；仅 system 的位置与取文本方式不同，由本类内部吸收。
"""

from __future__ import annotations

import os

from .model_endpoints import bare_model, endpoint, is_responses_model


# 单次 LLM 调用的 HTTP 超时。不设的话 SDK 默认无限等待——实测 answerer 的一次
# 调用挂在模型端点上 12 分钟不返回（连接 ESTAB、主线程 poll），而整个 run 只能
# 干等到外层超时，白烧预算且看不出原因。
#
# max_retries=0：SDK 自带的重试不带退避抖动，且吞掉错误类型，
# 重试逻辑统一由 ChatLLM.complete 负责（见 _RETRY_ON）。
_HTTP_TIMEOUT_S = 300


# 关思维链的模型前缀。辅助角色（answerer/judge/ask_detect）只需要简短结论，
# 开着思维链有两个实际危害：把 max_tokens 全耗在 reasoning 上导致 content 为空
# （实测 kimi-k3、glm-5.2 均如此），以及思维链长度不可控让有效输出预算漂移。
# 关思维链的模型前缀。辅助角色（answerer/judge/ask_detect）只需要简短结论，
# 开着思维链有两个实际危害：把 max_tokens 全耗在 reasoning 上导致 content 为空
# （实测 kimi-k3、glm-5.2 均如此），以及思维链长度不可控让有效输出预算漂移。
# DeepSeek-V4-Flash 在澄清 Judge 中曾把整段推理写进 content，耗尽 4096
# token 后没有 JSON。优先尝试关闭思维输出；若网关明确拒绝该参数，
# ``_complete_once`` 会自动去掉它重试，兼容不支持此开关的端点。
_THINKING_OFF_PREFIXES = ("glm-5", "kimi-k", "deepseek-v4-flash")

# 但有些模型**不允许**关思考，传了就 400。这类必须排除在上面的名单之外。
#
# 实测（2026-08-25）：
#   glm-5.3  thinking=disabled → HTTP 400 code 1210
#            「该模型始终思考，不支持关闭思考；请使用 low、high 或 max。」
#   glm-5.2  thinking=disabled → 正常，content='好的' 只花 1 个 output token
#
# 而 reasoning_effort=low **不是**替代品：实测 glm-5.3/glm-5.2 在 low 下
# 64 token 预算里 62 个都进了思维链、content 仍为空。真正的解法是不传这个参数
# 并给足 max_tokens（answerer 用 10000，实测思维链占 380~760，正文照样出来）。
#
# ⚠️ 这个 400 的后果远比"少一次回答"严重：answerer 抛异常 → question 永远不被
# reply → agent 卡在 question 工具上直到 max_runtime。实测一个 run 白烧 2270 秒、
# solve_secs=0、零产物，而分数上看像"模型不会做题"。
_ALWAYS_THINK_PREFIXES = ("glm-5.3",)


def _thinking_off(model: str | None) -> dict:
    name = (model or "").split("/")[-1].lower()
    if any(name.startswith(p) for p in _ALWAYS_THINK_PREFIXES):
        return {}
    if any(name.startswith(p) for p in _THINKING_OFF_PREFIXES):
        return {"thinking": {"type": "disabled"}}
    return {}


class ChatLLM:
    """按 protocol 分派的最小 chat 封装。complete(system, messages, max_tokens) -> str。"""

    def __init__(self, *, protocol: str, base_url: str, auth_token: str, model: str):
        protocol = (protocol or "anthropic").lower()
        if protocol not in ("anthropic", "openai"):
            raise ValueError(
                f"unknown protocol: {protocol!r}（应为 'anthropic' 或 'openai'）"
            )
        if not auth_token:
            raise ValueError("ChatLLM auth_token 未配置")
        self.protocol = protocol
        self._use_responses = protocol == "openai" and is_responses_model(model)
        route_url, route_key = endpoint(model) if self._use_responses else ("", "")
        self.base_url = route_url if self._use_responses else base_url
        self.auth_token = route_key if self._use_responses else auth_token
        self.model = bare_model(model) if model.startswith(("chat/", "responses/", "direct/")) or self._use_responses else model
        self._client = None

    def _build_client(self):
        if self.protocol == "anthropic":
            import anthropic
            # 临时清除 ANTHROPIC_API_KEY，避免 SDK 同时发 X-Api-Key + Authorization → 401
            saved = os.environ.pop("ANTHROPIC_API_KEY", None)
            try:
                return anthropic.Anthropic(auth_token=self.auth_token,
                                           base_url=self.base_url,
                                           timeout=_HTTP_TIMEOUT_S, max_retries=0)
            finally:
                if saved is not None:
                    os.environ["ANTHROPIC_API_KEY"] = saved
        # openai（base_url 需带版本段，一般 /v1）
        import openai
        return openai.OpenAI(api_key=self.auth_token, base_url=self.base_url,
                             timeout=_HTTP_TIMEOUT_S, max_retries=0)

    # 网关瞬时故障的重试。实测一次 502 Bad Gateway 就让整个 run 报废
    # （341s 白跑、status=error、零产物）——批量跑几千个 run 时这会是常态，
    # 且 answerer/judge 的失败与被测 agent 的能力无关，不该计入结果。
    _RETRY_ON = ("502", "503", "504", "Bad Gateway", "Service Unavailable",
                 "Gateway Time-out", "InternalServerError", "Connection",
                 "Timeout", "timed out",
                 # 限流：并发跑几十个 run 时最常见的瞬时错误，退避后基本都能过
                 "429", "RateLimit", "Too Many Requests", "rate limit")
    # 命中即**不**重试的永久性错误，优先级高于 _RETRY_ON。有些端点把
    # model_not_found 包在 503 里返回，按 503 重试 6 次纯属白等（实测 detect
    # 五个 judge 因模型 ID 带前缀全撞这个，每次调用空耗 3 分钟才失败）。
    _NEVER_RETRY = ("model_not_found", "does not exist", "invalid_api_key",
                    "Incorrect API key", "insufficient_quota")
    # 6 次 + 最长 60s 退避 ≈ 累计 3 分钟。4 次（累计 21s）扛不住实测遇到的
    # 间歇性 502——网关抖动窗口常在分钟级，退避太短会让落在窗口里的 run 全废。
    _MAX_ATTEMPTS = 6
    _TOTAL_BUDGET_S = 900        # 全部重试的累计上限（15 分钟）

    def complete(self, system: str, messages: list[dict], max_tokens: int = 4096,
                 extra_body: dict | None = None,
                 temperature: float | None = None) -> str:
        """带重试 + 墙钟硬截止的 complete。仅对瞬时网关错误重试。

        为什么必须自己加硬截止：httpx/openai 的 `timeout=` 是**每次读操作**的
        上限，不是整个请求的上限。服务端只要涓流发数据（每次 read 都在超时内
        返回），总时长就能无限长——实测一次 answerer 调用挂了 24 分钟仍未触发
        300s 超时，py-spy 显示卡在 ssl.read，而 run 只能干等到外层超时。

        实现用守护线程 + Future.result(timeout)：超时后主线程立即抛出，
        被抛弃的线程随连接关闭自行退出（daemon=True 保证不阻塞进程退出）。
        """
        import random
        import threading
        import time as _time

        last = None
        deadline = _time.monotonic() + self._TOTAL_BUDGET_S
        for attempt in range(self._MAX_ATTEMPTS):
            # 总预算封顶：单次 300s × 6 次 + 退避 ≈ 30 分钟，已超过单个 run 的
            # 预算。辅助 LLM（answerer/judge）不该吃掉被测 agent 的时间。
            left = deadline - _time.monotonic()
            if left <= 0:
                raise last or TimeoutError(
                    f"LLM 调用累计超过 {self._TOTAL_BUDGET_S}s 总预算")
            try:
                box: dict = {}

                def _work():
                    try:
                        box["ok"] = self._complete_once(
                            system, messages, max_tokens, extra_body, temperature)
                    except BaseException as exc:      # noqa: BLE001
                        box["err"] = exc

                # 必须是裸 daemon 线程，不能用 ThreadPoolExecutor：后者的工作线程
                # 非 daemon，解释器退出时 atexit 会 join 它们——被抛弃的挂起线程
                # 会把整个进程卡在退出阶段（实测合成用例 120s 无法退出）。
                th = threading.Thread(target=_work, daemon=True,
                                      name="llm-call")
                th.start()
                th.join(timeout=min(_HTTP_TIMEOUT_S, left))
                if th.is_alive():
                    # 连接可能仍挂着——丢弃客户端，下次重建，避免复用坏连接
                    self._client = None
                    raise TimeoutError(
                        f"LLM 调用超过 {_HTTP_TIMEOUT_S}s 墙钟上限（server 涓流或挂起）")
                if "err" in box:
                    raise box["err"]
                return box["ok"]
            except Exception as exc:
                msg = f"{type(exc).__name__}: {exc}"
                if any(sig in msg for sig in self._NEVER_RETRY):
                    raise            # 永久性错误（模型不存在/鉴权），重试只会白等
                if not any(sig in msg for sig in self._RETRY_ON):
                    raise            # 参数错误等，重试无意义
                last = exc
                if attempt < self._MAX_ATTEMPTS - 1:
                    # 指数退避 + 抖动：多个并发 run 同时撞上故障时避免齐步重试
                    _time.sleep(min(60, 2 ** attempt * 3) + random.uniform(0, 3))
        raise last

    def _complete_once(
        self,
        system: str,
        messages: list[dict],
        max_tokens: int = 4096,
        extra_body: dict | None = None,
        temperature: float | None = None,
    ) -> str:
        """messages 为 user/assistant 轮次列表（[{role, content}]）；system 单独传，内部按协议放对位置。

        extra_body: openai 协议下透传给底层 chat.completions.create 的额外参数（如 glm 的 thinking 开关）。
        """
        if self._client is None:
            self._client = self._build_client()

        if self.protocol == "anthropic":
            _kw = {}
            if temperature is not None:
                _kw["temperature"] = temperature
            msg = self._client.messages.create(
                model=self.model, max_tokens=max_tokens,
                system=system, messages=messages, **_kw,
            )
            for block in msg.content:
                if hasattr(block, "text"):
                    return block.text.strip()
            raise ValueError(f"Anthropic 响应无文本内容: {msg.content}")

        if self._use_responses:
            kwargs = {
                "model": self.model,
                "instructions": system,
                # Always carry complete history: this gateway's gpt-5.6-sol
                # route hangs when previous_response_id is used.
                "input": messages,
                "max_output_tokens": max(256, max_tokens),
            }
            if temperature is not None and self.model != "gpt-6-astra":
                kwargs["temperature"] = temperature
            if extra_body:
                kwargs["extra_body"] = extra_body
            resp = self._client.responses.create(**kwargs)
            text = (getattr(resp, "output_text", None) or "").strip()
            if not text:
                for item in getattr(resp, "output", []) or []:
                    if getattr(item, "type", None) != "message":
                        continue
                    for part in getattr(item, "content", []) or []:
                        if getattr(part, "type", None) == "output_text":
                            text += getattr(part, "text", "") or ""
            if text.strip():
                return text.strip()
            raise ValueError("Responses API 响应无 output_text")

        # openai chat-completions：system 作为首条 message
        full = [{"role": "system", "content": system}, *messages]
        kwargs = dict(model=self.model, max_tokens=max_tokens, messages=full)
        if temperature is not None:
            kwargs["temperature"] = temperature
        merged = {**_thinking_off(self.model), **(extra_body or {})}
        if merged:
            kwargs["extra_body"] = merged
        try:
            resp = self._client.chat.completions.create(**kwargs)
        except Exception as exc:                          # noqa: BLE001
            # 兜底：网关拒绝"关思考"时，去掉该参数重试一次。
            #
            # _ALWAYS_THINK_PREFIXES 是硬编码名单，只覆盖已知的型号；一个新模型
            # 上线并拒绝关思考时，名单来不及更新，而失败方式是灾难性的（提问永远
            # 得不到回答 → agent 卡到超时 → 整格记 0 分且看不出原因）。所以这里
            # 按错误文本自愈，而不是等着人去改名单。
            txt = str(exc)
            thinking_rejected = ("thinking" in txt.lower() or "思考" in txt) and (
                "400" in txt or "invalid" in txt.lower() or "不支持" in txt)
            if not (thinking_rejected and "thinking" in merged):
                raise
            merged.pop("thinking", None)
            if merged:
                kwargs["extra_body"] = merged
            else:
                kwargs.pop("extra_body", None)
            resp = self._client.chat.completions.create(**kwargs)
        msg = resp.choices[0].message
        text = (msg.content or "").strip()
        if not text:
            # 推理模型可能把全部输出写进 reasoning_content 而让 content 为空
            # （实测 kimi-k3 作 answerer 时如此，直接判失败会让整个 run 报废）。
            # 已按 _THINKING_OFF 关过思维链仍出现时，退而取 reasoning_content。
            text = (getattr(msg, "reasoning_content", None) or "").strip()
        if not text:
            raise ValueError(f"OpenAI 响应无文本内容: {resp}")
        return text
