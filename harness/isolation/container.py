"""容器执行层：把 agent 命令包进 `docker run`。

与 sandbox.py 并列，共用 opencode_run 里同一个插入点，二者互斥——容器已提供
文件与进程隔离，无需再叠 bwrap。

设计要点
────────
1. workspace 挂到容器内**同一绝对路径**。prompt 里写的是宿主机绝对路径，
   若换成 /workspace，agent 写出的路径引用会与宿主机不一致。
2. case 目录（gt.json / tests/）不挂载即天然不可见，比 bwrap 的 deny 列表更可靠。
3. 必须 --network=host：默认 bridge 网络解析不到内网镜像源，agent 运行期
   pip install 会超时，最终表现为「不会用某个库」，实则是环境问题。
4. 镜像不存在时**直接报错**，绝不回落到宿主机执行——否则测的是宿主机环境，
   而基础设施故障会被误读成模型能力不足。
"""

from __future__ import annotations

import shutil
import subprocess
from dataclasses import dataclass, field
from pathlib import Path


class ContainerError(RuntimeError):
    """镜像缺失、docker 不可用等环境问题。区别于 agent 自身的失败。"""


# 容器里 agent / 辅助 LLM 需要的全部凭据与端点。docker 不继承宿主机 env，只有列在
# 这里的才进容器；私有 OpenCode 配置引用的变量也必须列入。
DEFAULT_ENV_PASSTHROUGH: tuple[str, ...] = (
    "OPENAI_API_KEY", "OPENAI_BASE_URL",
    "DELIVER_AGENT_API_KEY", "DELIVER_AGENT_BASE_URL",
    "DELIVER_RESPONSES_API_KEY", "DELIVER_RESPONSES_BASE_URL",
    "DELIVER_DIRECT_API_KEY", "DELIVER_DIRECT_BASE_URL",
    "FDE_OPENCODE_CONFIG", "FDE_EXTRACTOR_OPENCODE_CONFIG",
    "ANTHROPIC_API_KEY", "ANTHROPIC_BASE_URL", "ANTHROPIC_AUTH_TOKEN",
    "DELIVER_ANSWERER_TOKEN", "DELIVER_JUDGE_TOKEN", "DELIVER_ANSWERER_BASE_URL",
    "GEMINI_API_KEY", "GOOGLE_GEMINI_BASE_URL",
)

# 脚手架名 → 可执行文件名。which(scaffold) 对 claude-code / deepseek-harness /
# opencode-run 全部找不到（二进制分别叫 claude / dsh / opencode）。
AGENT_BINARY: dict[str, str] = {
    "opencode": "opencode", "opencode-run": "opencode",
    "claude-code": "claude", "codex": "codex", "gemini": "gemini", "kimi": "kimi",
    "deepseek-harness": "dsh", "dsh": "dsh",
}


def agent_binary(scaffold: str, explicit: str | None = None) -> Path | None:
    """定位要挂进容器的脚手架二进制；找不到返回 None。"""
    if explicit:
        return Path(explicit).resolve()
    name = AGENT_BINARY.get(scaffold, scaffold)
    found = shutil.which(name)
    if not found and scaffold in ("deepseek-harness", "dsh"):
        try:
            from ..config import get
            found = get("deepseek_harness", "bin", None) or None
        except Exception:                                  # noqa: BLE001
            found = None
    return Path(found).resolve() if found else None


def build_spec(image: str, scaffold: str, *, agent_bin: str | None = None) -> "ContainerSpec":
    """cli.py 与 matrix.py 共用的 ContainerSpec 构造：同一份 bind、资源上限、凭据透传。"""
    real = agent_binary(scaffold, agent_bin)
    ct, mb, np_ = resource_defaults()
    return ContainerSpec(
        image=image,
        # 镜像只装 case 的科学计算栈，脚手架二进制由外部挂入，同一份镜像可供多脚手架共用
        binds=((str(real), f"/usr/local/bin/{real.name}"),) if real else (),
        cpu_time_s=ct, memory_bytes=mb, nproc=np_,
        env_passthrough=DEFAULT_ENV_PASSTHROUGH,
    )


def resource_defaults() -> tuple[int | None, int | None, int | None]:
    """从 config.toml 读容器资源上限。返回 (cpu_time_s, memory_bytes, nproc)。

    0 / 空 / 缺省 → None（不限制）。两处构造点（cli.py 单 case、matrix.py worker）共用。

    ⚠️ 目前只有 cpu_time_s 会真正作用到容器。memory_bytes / nproc 仍解析并返回，
    但 wrap_command 不再据此加限制——共享 uid 下 RLIMIT_NPROC 是全局语义、
    RLIMIT_AS 对 Node 应用会误杀（详见 wrap_command 注释）。保留读取，是为了
    换到 cgroup 可用的宿主机时能直接接上 --memory / --pids-limit。
    """
    def _v(sec, key):
        try:
            from ..config import get
            return get(sec, key)
        except Exception:
            return None

    # CPU/内存本命走进程级 RLIMIT（cgroup v2 在容器里不可用，见 wrap_command）。
    # 从 vCPU 数折算 RLIMIT_CPU 累计秒数：多核下 CPU 秒数 ≈ 墙钟 × 核数。
    # 必须取整：docker --ulimit 只收整数，传 57600.0 会被 strconv.ParseInt 拒掉，
    # 容器根本起不来（实测 status=error 0s，排查时看着像脚手架挂了）。
    cpus = _v("agent", "container_cpus")
    cpu_time_s = int(float(cpus) * 3600) if cpus else None   # 每 run 最多用 核数×1h 的 CPU 秒

    mem = _v("agent", "container_memory")               # 如 "8g" / "8192m"
    memory_bytes = None
    if mem:
        mem = str(mem).strip().lower()
        mult = 1
        for suf, m in (("g", 1024**3), ("m", 1024**2), ("k", 1024), ("t", 1024**4)):
            if mem.endswith(suf):
                try: mult = m; mem = mem[:-1]; break
                except Exception: pass
        try: memory_bytes = int(float(mem) * mult)
        except Exception: memory_bytes = None

    nproc = _v("agent", "container_nproc")
    nproc = int(nproc) if nproc else None
    return cpu_time_s, memory_bytes, nproc


@dataclass(frozen=True)
class ContainerSpec:
    image: str
    network: str = "host"
    # agent 可执行文件：宿主机路径 -> 容器内路径，只读挂载。
    # 镜像只装 case 所需的科学计算栈，与脚手架无关，这样一份镜像可跑多个脚手架。
    binds: tuple[tuple[str, str], ...] = ()
    extra_args: tuple[str, ...] = ()
    # ── 资源上限（宽松隔离，非严格对等）──
    # 目的：防单个 run 的越界行为（OOM、求解器吃满多核）拖垮并发的邻居，
    # 不为追求跨 run 公平——公平由「被测能力差异」决定，不靠限资源制造。
    # None = 不限制该项。
    #
    # 本环境（宿主机容器 cgroup v2 = threaded）下**不能用 cgroup 限制**
    #（--cpus/--memory 会触发 runc 的 "cannot enter cgroupv2 ... threaded mode"，
    # 实测拉起即失败）。故 CPU/内存走了进程级 RLIMIT——Docker 的 --ulimit
    # 是 per-process 限制，不依赖 cgroup 层级，已验证可用。
    cpu_time_s: int | None = None    # RLIMIT_CPU：累计 CPU 秒数上限（多核≈墙钟×核数）
    memory_bytes: int | None = None  # RLIMIT_AS：虚拟地址空间上限（字节）
    nproc: int | None = None         # RLIMIT_NPROC：进程数上限
    env_passthrough: tuple[str, ...] = field(default=())


def docker_available() -> bool:
    if shutil.which("docker") is None:
        return False
    return subprocess.run(
        ["docker", "info"], capture_output=True, timeout=30
    ).returncode == 0


def image_exists(image: str) -> bool:
    return subprocess.run(
        ["docker", "image", "inspect", image], capture_output=True, timeout=30
    ).returncode == 0


def require(spec: ContainerSpec) -> None:
    """前置校验。失败即抛错，让上层把这次 run 标成 harness error 而非 agent failure。"""
    if not docker_available():
        raise ContainerError("docker 不可用（未安装或 daemon 未运行）")
    if not image_exists(spec.image):
        raise ContainerError(
            f"镜像 {spec.image} 不存在。先构建：./build_images.sh <case_name>"
        )


def wrap_command(
    cmd: list[str],
    spec: ContainerSpec,
    workspace: Path,
    env: dict[str, str] | None = None,
) -> list[str]:
    """把 cmd 包成 docker run。返回可交给 Popen 的新命令。"""
    ws = str(workspace.resolve())
    docker_cmd = [
        "docker", "run", "--rm", "-i",
        "--network", spec.network,
        "-v", f"{ws}:{ws}",
        "-w", ws,
    ]

    # 以宿主机 uid 运行，避免容器写出 root 属主的产物导致宿主机后续读写失败
    import os
    docker_cmd += ["--user", f"{os.getuid()}:{os.getgid()}"]

    # environment.py 把私有 XDG 目录建在 run_dir/.agent_home（在 workspace 之外），
    # 并只通过环境变量传递。docker 既不继承宿主机 env、也看不到未挂载的路径，
    # 二者缺一 opencode 就找不到 provider 配置，报 ProviderModelNotFoundError。
    xdg_dirs: list[str] = []
    for key in ("XDG_CONFIG_HOME", "XDG_DATA_HOME", "XDG_CACHE_HOME"):
        val = (env or {}).get(key)
        if not val:
            continue
        docker_cmd += ["-e", f"{key}={val}"]
        xdg_dirs.append(val)

    if xdg_dirs:
        # 挂公共父目录（即 .agent_home）而非三个子目录，少三个 mount 且保持结构完整。
        # 必须可写：opencode 要在 data/ 下建 SQLite、在 cache/ 下写缓存。
        home = str(Path(os.path.commonpath(xdg_dirs)))
        docker_cmd += ["-v", f"{home}:{home}", "-e", f"HOME={home}"]
    else:
        docker_cmd += ["-e", f"HOME={ws}/.agent_home"]

    for host_path, container_path in spec.binds:
        docker_cmd += ["-v", f"{host_path}:{container_path}:ro"]

    # 资源上限：宽松隔离，走**进程级 RLIMIT**（见 ContainerSpec 注释——cgroup
    # 在本环境不可用，--cpus/--memory 会拉起即失败）。
    # 一律 int()：docker 的 --ulimit 走 strconv.ParseInt，收到 "57600.0" 直接拒，
    # 容器起不来而错误只体现为 status=error 0s，排查时极易误判成脚手架故障。
    #
    # ⚠️ nproc 与内存两项已停用，原因是它们在**共享 uid 的容器**里语义不对：
    #
    #   RLIMIT_NPROC 限的是「该 uid 在整个内核范围的进程/线程总数」，不是本容器
    #   内的。容器以 --user 0:0 运行、与宿主共享 uid=0，而宿主 root 侧常驻上千个
    #   线程（实测 1054），于是 nproc=64 让容器里 fork 任何新进程立刻 EAGAIN。
    #   实测现象极具迷惑性：agent 跑完头两个工具后静默停摆，无报错、无产物，
    #   38s 就被判 degraded，看着像 provider 故障。
    #
    #   RLIMIT_AS 限的是虚拟地址空间，而 opencode 是 Node 应用——V8 保留的地址
    #   空间远大于实际用量，且 bash 工具 fork 出的子进程共同受限，容易误杀。
    #
    # 结论：这两项要限，得靠 cgroup（--memory / --pids-limit）；本环境 cgroup v2
    # 是 threaded 用不了，就不限——宁可不限，也不能悄悄掐死 agent 让分数变脏。
    if spec.cpu_time_s is not None:
        _cpu = int(spec.cpu_time_s)
        docker_cmd += ["--ulimit", f"cpu={_cpu}:{_cpu}"]

    for key in spec.env_passthrough:
        if env and key in env:
            docker_cmd += ["-e", f"{key}={env[key]}"]

    docker_cmd += list(spec.extra_args)
    docker_cmd.append(spec.image)
    full = docker_cmd + list(cmd)
    if os.environ.get("DELIVER_DEBUG_CMD"):
        import shlex
        print("[container] " + shlex.join(full), flush=True)
    return full
