"""
Runner 层基础抽象

AgentSession  — 统一的消息发送接口，隐藏协议差异
AgentRunner   — 启动 agent session 的工厂
RunnerConfig  — runner 可配置项（与 case 位置无关）
AgentEvent    — 统一的结构化事件
"""

from abc import ABC, abstractmethod
from collections.abc import Callable
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any


# ── 结构化事件 ──

@dataclass
class AgentEvent:
    """Runner 产出的统一事件，各 backend 统一格式。"""
    type: str           # "text" | "tool_call" | "tool_result" | "info"
    content: str = ""   # 主要文本内容
    tool: str = ""      # 工具名（仅 tool_call / tool_result）
    tool_input: str = ""  # 工具输入摘要（仅 tool_call）
    is_error: bool = False  # 仅 tool_result

EventCallback = Callable[[AgentEvent], Any] | None


def default_event_printer(event: AgentEvent) -> None:
    """默认的事件处理：打印到 stdout（CLI 场景）。"""
    if event.type == "text":
        print(event.content)
    elif event.type == "text_chunk":
        print(event.content, end="", flush=True)
    elif event.type == "tool_call":
        print(f"  [tool] {event.tool}: {event.tool_input}")
    elif event.type == "tool_result" and event.is_error:
        print(f"  [error] {event.tool}: {event.content[:300]}")
    elif event.type == "info":
        print(f"  {event.content}")


def _idle_default() -> int:
    # 兜底值与 config.toml 的 idle_timeout_s 保持一致（43200 = 与 timeout_s 相等，
    # 即 idle 不再单独构成约束）。两处写同一个数是为了：config 读不到时不会
    # 静默退回一个更严的 12 小时上限，把正在长跑求解的 run 误杀成"卡死"。
    try:
        from ..config import get
        return int(get("agent", "idle_timeout_s", 43200))
    except Exception:
        return 43200


@dataclass
class RunnerConfig:
    """Runner 的运行配置：平台 + 整装 agent 配置目录 + 运行时参数。

    跑什么 agent 由 agent_config_dir 指向的整装目录决定（installer 整体装入
    workspace），不在此处做任何行为配置。
    """
    runner_type: str                         # 脚手架名，见 registry.available()
    agent_config_dir: Path | None = None     # 整装 agent 配置目录（如 <repo>/opencode/）；None = native 不注入
    model: str | None = None                 # 覆盖默认模型（None 表示使用 runner 默认值）
    sandbox: Any = None                      # SandboxConfig | None，避免循环 import
    container: Any = None                    # ContainerSpec | None；与 sandbox 互斥，容器自带隔离
    # 仅供评分抽取器：其 session cwd 是 normalized 输出目录，但必须读取 cwd 外的
    # agent workspace / evaluator / data。主 Agent 默认 False，避免放宽实验隔离。
    allow_external_directory: bool = False
    # ── 时长防护（两平台统一；死循环检测归脚手架自身：CC 内置 / opencode doom_loop+step-limiter）──
    max_runtime_s: int = 43200               # 整个 run 总时长硬上限——文本多轮共享预算（默认 12h，标定见 docs/FRAMEWORK.md §执行预算）
    # 无 stdout 事件看门狗——防"卡死"。默认读 config.toml [agent].idle_timeout_s
    idle_timeout_s: int = field(default_factory=lambda: _idle_default())
    # ── 轮数上限：工具调用次数硬上限，防"永远在打转"。只在能精确数到每次工具
    #    调用的后端生效（opencode serve）；CLI subprocess 后端拿不到中间状态，
    #    仍只靠 max_runtime_s 计时约束。None = 不限制。
    max_turns: int | None = None


class AgentSession(ABC):
    """统一的 agent 交互接口。"""

    session_id: str | None
    on_event: EventCallback = None
    # 该后端能否逐条上报工具调用（AgentEvent type="tool_call"）。False 时
    # usage.json 的 tool_calls 记 None 而非 0——纯文本 stdout 的 CLI 发不出
    # 结构化事件，"数到 0 次"和"数不到"必须分开，否则 avg_tools 把假零和
    # opencode 的真实计数平均在一起。
    reports_tool_calls: bool = False
    # 该后端上报的 usage.input 是否已包含 cache_read。OpenAI / Gemini 的
    # prompt_tokens 含缓存命中；Anthropic 的 input_tokens 与 opencode 的 input
    # 不含。UsageLedger 据此把 input 归一成「不含缓存」，全框架只用一个口径：
    #   cache_hit_rate = cache_read / (input + cache_read)
    input_includes_cache_read: bool = False

    @abstractmethod
    async def send(self, message: str, *, stop_on_tools: set[str] | None = None) -> str:
        """发送一条消息，阻塞到本轮结束，返回完整回复文本。

        stop_on_tools: 若非 None，在检测到这些工具被调用时立即终止本轮，
                       返回已收集的文本。用于 B2 在执行开始前提前退出。
        """

    @abstractmethod
    async def close(self) -> None:
        """关闭 session，释放资源。"""

    def _emit(self, event: AgentEvent) -> None:
        """发送事件给回调，若无回调则使用默认 printer。"""
        handler = self.on_event or default_event_printer
        handler(event)

    def usage_snapshot(self) -> dict | None:
        """Optional native usage metadata collected by this session.

        Backends that do not expose trustworthy token metadata return ``None``.
        The OpenCode backend is intentionally unchanged: its authoritative
        SQLite accounting remains in :mod:`harness.scoring.usage`.
        """
        return None

    def set_usage_phase(self, phase: str | None) -> None:
        """Hint the optional usage ledger about the current phase."""
        return None

    def relabel_last_usage_phase(self, phase: str | None) -> None:
        return None

    def usage_call_count(self) -> int:
        """账本里已记的模型调用数；clarify/loop 在每次 send 前取一次作区间起点。"""
        return 0

    def _note_agent_send(self) -> None:
        """记录一次 harness 发给脚手架的逻辑消息。

        ``send`` 是所有 backend 都具备的共同边界；底层 LLM 调用次数则不一定
        暴露（Gemini CLI 只给整轮聚合）。因此把二者分开记录，避免再拿
        ``tokens.steps`` 同时表示 send 和 LLM call。
        """
        self._agent_sends = getattr(self, "_agent_sends", 0) + 1

    def agent_send_count(self) -> int:
        """本 run 已提交给脚手架的逻辑消息数。"""
        return int(getattr(self, "_agent_sends", 0))

    def label_usage_from(self, start: int, phase: str) -> None:
        """把第 start 条起的调用标成 phase。无账本的后端为 no-op。"""
        return None

    def effective_models(self) -> list[str]:
        """Return provider/model identities observed from backend metadata."""
        return []

    async def __aenter__(self):
        return self

    async def __aexit__(self, *_):
        await self.close()


class AgentRunner(ABC):
    """启动 agent session 的工厂。"""

    def __init__(self, config: RunnerConfig):
        self.config = config

    @abstractmethod
    async def start(self, workspace: Path) -> AgentSession:
        """
        在 workspace 准备 agent 运行环境（复制配置等），返回可交互的 session。
        workspace 内的业务数据由调用方（benchmark 层）在调用前准备好。
        """
