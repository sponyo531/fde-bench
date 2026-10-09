"""容器资源限制的离线单测。

覆盖 `wrap_command` 生成 docker 命令的资源 flag。不真起 docker——校验的是
**生成的命令**。

背景（两轮踩坑，都写在这里防回退）：

1. cgroup 走不通：宿主机容器的 cgroup v2 是 threaded，`--cpus` / `--memory` /
   `--pids-limit` 一律拉不起容器（runc 报 "cannot enter cgroupv2 ... threaded
   mode"）。所以只能走进程级 RLIMIT。

2. 三项 RLIMIT 里只有 CPU 能用：
   - RLIMIT_NPROC 限的是「该 uid 在全内核范围」的进程数，不是本容器的。容器与
     宿主共享 uid=0，宿主 root 侧常驻上千线程，于是 nproc=64 让容器内 fork 立刻
     EAGAIN——实测 agent 跑完头两个工具后静默停摆，38s 判 degraded，看着像
     provider 故障，排查方向全错。
   - RLIMIT_AS 限虚拟地址空间，而 opencode 是 Node 应用，V8 保留量远大于实际
     用量，容易误杀。
   故 wrap_command 只发 `--ulimit cpu=`，另两项即使配了也不生效。
"""

from __future__ import annotations

import re
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from harness.isolation.container import (ContainerSpec, wrap_command,  # noqa: E402
                                         resource_defaults)


def _cmd(spec, *, env=None):
    ws = Path("/tmp/_ut_ws")
    return " ".join(wrap_command(["opencode", "run"], spec, ws, env or {}))


def test_cpu_limit_emitted():
    """配了 cpu_time_s 就该出现 --ulimit cpu=，且为整数字面量。"""
    s = _cmd(ContainerSpec(image="x", cpu_time_s=57600))
    assert "--ulimit cpu=57600:57600" in s, s


def test_cpu_float_is_coerced_to_int():
    """float 必须取整再拼。

    docker 的 --ulimit 走 strconv.ParseInt，收到 "cpu=57600.0:57600.0" 直接拒绝，
    容器起不来。而这在 harness 里只表现为 status=error / 0s / 无产物——看着像
    脚手架挂了（实测 e2e 首跑即栽在这里）。config 允许 container_cpus 写小数，
    所以取整必须在拼命令这一层兜住。
    """
    s = _cmd(ContainerSpec(image="x", cpu_time_s=57600.0))
    assert "--ulimit cpu=57600:57600" in s, s
    assert not re.search(r"--ulimit \w+=\d+\.\d+", s), s


def test_nproc_never_emitted():
    """nproc 即使配了也不能进命令——它会掐死容器内的 fork。

    共享 uid 下 RLIMIT_NPROC 是全内核语义，宿主 root 的上千线程已经把额度吃满。
    这条断言就是那次 degraded 事故的回归防线。
    """
    s = _cmd(ContainerSpec(image="x", cpu_time_s=3600, nproc=64))
    assert "nproc" not in s, s


def test_memory_never_emitted():
    """memory 即使配了也不能进命令——RLIMIT_AS 会误杀 Node 应用。"""
    s = _cmd(ContainerSpec(image="x", cpu_time_s=3600, memory_bytes=8 * 1024**3))
    assert "ulimit -v" not in s, s
    assert "--memory" not in s, s
    # 不该再有 sh -c 包装，命令应直接是 agent 本身
    assert "/bin/sh" not in s, s
    assert s.rstrip().endswith("opencode run"), s


def test_no_limits_no_flags():
    """全空 → 干净 docker 命令，没有任何资源 flag。"""
    s = _cmd(ContainerSpec(image="x"))
    assert "--ulimit" not in s, s
    assert "ulimit -v" not in s, s


def test_resource_defaults_reads_config():
    """resource_defaults 按 config.toml 解析；类型必须是 int 或 None。

    cpu 由 container_cpus 折算成 RLIMIT_CPU 秒数（核数 × 1h）；
    memory / nproc 仍解析（换到 cgroup 可用的宿主机可直接接上），但当前不生效。
    """
    ct, mb, np = resource_defaults()
    for v in (ct, mb, np):
        assert v is None or isinstance(v, int), (type(v), v)
    if ct is not None:
        assert ct % 3600 == 0, ct        # 折算自整数核数


if __name__ == "__main__":
    for fn in [v for k, v in sorted(globals().items()) if k.startswith("test_")]:
        fn()
    print("\nall container-limit tests passed")
