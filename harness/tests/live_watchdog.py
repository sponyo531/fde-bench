"""看门狗超时路径的联调脚本（起真 serve、真跑一条会卡住的命令）。

单测用假 HTTP 层验证逻辑；这个脚本验证的是「真实 opencode 下超时到底会不会
收手」——两者缺一不可：真实验里看门狗没按时开火，正是单测覆盖不到的那一半
（abort 的实际语义、bash 子进程挂在谁的进程组下）。

    python3 harness/tests/live_watchdog.py
"""

import asyncio
import pathlib
import sys
import time

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[2]))

from harness.backends.opencode.serve import OpenCodeServeSession  # noqa: E402


class Cfg:
    max_runtime_s = 25
    idle_timeout_s = 3600
    container = None
    agent_config_dir = None


def _is_stuck_proc(proc: pathlib.Path) -> bool:
    try:
        return b"time.sleep(600)" in (proc / "cmdline").read_bytes()
    except OSError:
        return False


async def main() -> None:
    ws = pathlib.Path("/tmp/wdtest")
    ws.mkdir(exist_ok=True)
    sess = OpenCodeServeSession(ws, config=Cfg())
    sess._ABORT_GRACE_S = 15
    sess.on_event = lambda e: print(f"  [{e.type}] {e.tool} {e.content[:140]}", flush=True)

    t0 = time.time()
    text = await sess.send(
        '请运行这条命令：python3 -c "import time; time.sleep(600)"。直接执行，不要问我。'
    )
    print(f"\nsend returned after {time.time() - t0:.0f}s "
          f"killed_reason={sess.killed_reason!r} text={text[:80]!r}", flush=True)
    await sess.close()

    leftover = [p.name for p in pathlib.Path("/proc").iterdir()
                if p.name.isdigit() and _is_stuck_proc(p)]
    assert sess.killed_reason == "max_runtime", sess.killed_reason
    # 这条是本脚本存在的理由：abort 只中断 agent 的推理循环，**不杀**它派生的
    # 子进程，而那些子进程也不在 serve 的进程组里（killpg 够不着）。实测一个
    # time.sleep(600) 在 abort + killpg 之后依然活着，必须整棵进程树杀。
    assert not leftover, f"agent 派生的子进程没被回收: {leftover}"
    print("✓ 超时收手，且未残留子进程")


asyncio.run(main())
