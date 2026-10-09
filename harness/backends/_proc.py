"""带进程组超时的子进程执行。

`subprocess.run(timeout=...)` 超时只 kill **直接**子进程；CLI 脚手架（claude / codex /
gemini / kimi / OpenHands driver）都会再起 bash、python 等子孙进程，它们握着继承的
stdout/stderr 管道不放，`communicate()` 等不到 EOF 就一直挂——表面上是"超时了也停不
下来"，最后由 cli.py 的外层 wait_for 兜底并把原因记成 outer_wait_for，而不是真实的
max_runtime。实测 codex per-send 300s 上限，拖到 420s 才由外层收尾。

修法：`start_new_session=True` 让子进程自成进程组，超时时 killpg 整组（先 TERM 后 KILL），
并把已收到的 stdout 塞进 TimeoutExpired 返回——超时不代表 agent 什么都没说。
"""

from __future__ import annotations

import os
import signal
import subprocess
import time


def run_pg(cmd: list[str], *, input: str | None = None, timeout: float | None = None,
           cwd: str | None = None, env: dict | None = None,
           grace_s: float = 5.0) -> subprocess.CompletedProcess:
    """同 subprocess.run(capture_output=True, text=True)，但超时时杀整个进程组。"""
    proc = subprocess.Popen(
        cmd, cwd=cwd, env=env, text=True,
        stdin=subprocess.PIPE if input is not None else subprocess.DEVNULL,
        stdout=subprocess.PIPE, stderr=subprocess.PIPE,
        start_new_session=True,
    )
    try:
        out, err = proc.communicate(input=input, timeout=timeout)
    except subprocess.TimeoutExpired:
        _killpg(proc, grace_s)
        # 收尾读残余：进程组已死，管道会到 EOF，不再挂
        try:
            out, err = proc.communicate(timeout=grace_s)
        except Exception:                                    # noqa: BLE001
            out, err = "", ""
        raise subprocess.TimeoutExpired(cmd, timeout or 0, output=out, stderr=err)
    return subprocess.CompletedProcess(cmd, proc.returncode, out, err)


def _killpg(proc: subprocess.Popen, grace_s: float) -> None:
    try:
        pgid = os.getpgid(proc.pid)
    except ProcessLookupError:
        return
    for sig in (signal.SIGTERM, signal.SIGKILL):
        try:
            os.killpg(pgid, sig)
        except ProcessLookupError:
            return
        deadline = time.monotonic() + grace_s
        while time.monotonic() < deadline:
            if proc.poll() is not None:
                return
            time.sleep(0.1)
