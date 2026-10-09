"""沙箱：把 case 里的答案挡在 agent 视线外。

case 目录下有 gt.json（考点）与 tests/（评估器、真值、baseline），
agent 若能读到，澄清与求解两项评测同时失效。这里默认开启隔离：

    deny_read  = case 目录、results 目录
    allow_read = workspace（内含 data/ 副本）

注意：与产品 harness 不同，本模块不依赖任何配置文件——deny 路径由
case/results 目录直接推导，避免"配置缺失导致沙箱静默不启用"。
"""

from __future__ import annotations

import importlib.util
from pathlib import Path
from typing import Any

# ``isolation`` 与 ``vendor`` 都是 ``harness`` 的直接子目录。这里不能从
# ``harness/isolation`` 下继续拼 ``vendor``，否则会错误地寻找
# ``harness/isolation/vendor/sandbox.py``，使所有宿主机 sandbox 探测在真正
# 调用 bwrap 之前就因 FileNotFoundError 失败。
_VENDOR = Path(__file__).resolve().parent.parent / "vendor" / "sandbox.py"


def _mod():
    spec = importlib.util.spec_from_file_location("_db_sandbox", _VENDOR)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"cannot load {_VENDOR}")
    m = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(m)
    return m


def build_config(
    case_dir: Path, workspace: Path, results_root: Path, agent_home: Path | None = None
) -> Any:
    """构造沙箱配置。

    deny 整个 results_root 是为了防止 agent 偷看同批次其他 run 的产物；
    但本 run 自己的 workspace 与 agent 配置目录必须放行——两者都在
    results_root 之下，遗漏任一个都会导致 agent 无法工作（配置读不到时
    opencode 会静默回落到官方默认 provider，评测即失真）。
    """
    m = _mod()

    # opencode 会向上遍历祖先目录寻找 .claude/skills、AGENTS.md 等项目级配置。
    # 若不屏蔽，agent 会自动加载运行者本机的业务 skill——实测曾加载 8 个，
    # 并切换到 agent=build 模式。那样跑出的分数反映的是"这台机器装了什么"，
    # 不同机器之间不可比。
    ancestor_cfg: list[str] = []
    for parent in [workspace.resolve(), *workspace.resolve().parents]:
        for name in (".claude", ".opencode", ".agents", "AGENTS.md", "CLAUDE.md"):
            cand = parent / name
            if cand.exists():
                ancestor_cfg.append(str(cand))

    allow = [str(workspace.resolve())]
    if agent_home is not None:
        allow.append(str(agent_home.resolve()))
    return m.SandboxConfig(
        deny_read=[
            str(case_dir.resolve()),
            str(results_root.resolve()),
            *ancestor_cfg,
        ],
        allow_read=allow,
    )


def wrap_command(cmd: list[str], config: Any, workspace: Path) -> list[str]:
    """把命令包进沙箱。backends 层调用。"""
    return _mod().wrap_command(cmd, config, workspace)


def available(workspace: Path) -> bool:
    """探测当前环境能否建立隔离（Linux 需 bwrap + user namespace）。"""
    m = _mod()
    try:
        return bool(m._bwrap_works(workspace))
    except Exception:
        return False
