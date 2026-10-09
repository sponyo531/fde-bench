"""脚手架适配层。

    base.py       抽象契约（AgentSession / AgentRunner / RunnerConfig / AgentEvent）
    registry.py   自动发现（加 agent 无需登记，契约见该文件）
    _cli_spec.py  headless CLI 的共用机制
    <agent>/      一个 agent 一个包，各自带实测的坑
"""

from .base import (AgentEvent, AgentRunner, AgentSession,   # noqa: F401
                   RunnerConfig, default_event_printer)
from .registry import available, load_errors, runner_from_config  # noqa: F401
