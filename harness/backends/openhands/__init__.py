"""OpenHands（副表用）。

与其他脚手架的结构性差异：`run_controller` **一次跑到底**，不能像 opencode 那样
反复 send()。所以澄清必须注入它自己的循环——走原生 `fake_user_response_fn` 回调，
而回调在 driver.py（独立 venv 的子进程）里，拿不到父进程的 answerer，
故用文件 IPC：driver 写 `.ask` → 父进程调 answerer → 写 `.ans`。

为什么要独立 venv：OpenHands 需 rich>=14，而本机全局 rich 13.7.1 被 17 个包
（aider / angr / keras / cudf 等）依赖不能升级。venv 位置见 config.toml
的 [openhands].venv。
"""

from .session import OpenHandsRunner, _OpenHandsSession   # noqa: F401

AGENTS = {"openhands": OpenHandsRunner}
