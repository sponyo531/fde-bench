"""脚手架注册表：**自动发现**，加 agent 无需登记。

契约只有一条：`backends/<你的agent>/__init__.py` 里导出

    AGENTS = {"你的名字": 某个 AgentRunner 子类}

于是：
  - `ls backends/` 就是「支持哪些 agent」的答案
  - 加一个 agent = 加一个文件夹，不用改这里，也不用改任何调用方

一个包可以导出多个名字（如 opencode 同时提供 serve 与 run 两种形态）。

为什么不用 import 全部包来发现：某个 agent 的依赖坏掉时（OpenHands 的独立 venv
最容易出问题），不该把其他 agent 的运行一起拖崩。故逐包 import、失败只记账，
`available()` 仍返回其余可用的，`--list` 会把失败原因显示出来。
"""

from __future__ import annotations

from functools import lru_cache
from importlib import import_module
from pathlib import Path

from .base import AgentRunner, RunnerConfig

_HERE = Path(__file__).resolve().parent
# 下划线开头的是共用机制（_cli_spec）而非 agent；base/registry 也不是包
_SKIP = {"tests", "__pycache__"}


def _packages() -> list[str]:
    return sorted(
        d.name for d in _HERE.iterdir()
        if d.is_dir() and not d.name.startswith(("_", "."))
        and d.name not in _SKIP and (d / "__init__.py").is_file()
    )


@lru_cache(maxsize=1)
def _discover() -> tuple[dict[str, type[AgentRunner]], dict[str, str]]:
    """返回 (名字 → Runner 类, 加载失败的包 → 原因)。"""
    found: dict[str, type[AgentRunner]] = {}
    failed: dict[str, str] = {}
    for pkg in _packages():
        try:
            mod = import_module(f".{pkg}", __package__)
            agents = getattr(mod, "AGENTS", None)
            if not agents:
                failed[pkg] = "缺少 AGENTS 声明（见 registry.py 的契约）"
                continue
            found.update(agents)
        except Exception as exc:                      # noqa: BLE001
            failed[pkg] = f"{type(exc).__name__}: {exc}"
    return found, failed


def available() -> list[str]:
    """可用的脚手架名。"""
    return sorted(_discover()[0])


def load_errors() -> dict[str, str]:
    """加载失败的包及原因，供 --list 显示。"""
    return dict(_discover()[1])


def runner_from_config(config: RunnerConfig) -> AgentRunner:
    found, failed = _discover()
    cls = found.get(config.runner_type)
    if cls is None:
        hint = ""
        # 名字对得上但包没加载成功时，报真正的原因，而不是含混的"不认识"
        for pkg, why in failed.items():
            if pkg.replace("_", "-") == config.runner_type:
                hint = f"（该包加载失败：{why}）"
        raise ValueError(
            f"unknown scaffold {config.runner_type!r}{hint}; "
            f"available: {available()}"
        )
    return cls(config)
