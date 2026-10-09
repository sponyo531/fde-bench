"""
harness/vendor/sandbox.py — 跨平台文件系统沙箱（vendored，无外部配置依赖）。

macOS: sandbox-exec + Seatbelt profile（deny 指定路径，不限网络）
Linux: bubblewrap (bwrap)，不可用时抛出 RuntimeError

用法：
    from .sandbox import SandboxConfig, wrap_command

    cfg = SandboxConfig(
        deny_read=["/path/to/cases", "/path/to/batch"],
        allow_read=["/path/to/workspace"],
    )
    cmd = wrap_command(["claude", "--print", ...], cfg, workspace)
"""

import os
import platform
import shutil
import sys
from dataclasses import dataclass, field
from pathlib import Path


@dataclass
class SandboxConfig:
    """沙箱配置：deny_read 的路径禁止读取，allow_read 在 deny 范围内例外放开。"""
    deny_read: list[str] = field(default_factory=list)
    allow_read: list[str] = field(default_factory=list)

def _opencode_install_dir() -> str:
    """返回 opencode 的安装目录（bin/opencode 所在目录的上一级）。
    优先通过 which 探测实际位置，再检查常见位置，回退到 ~/.opencode。"""
    binary = shutil.which("opencode")
    if binary:
        return str(Path(binary).resolve().parent.parent)
    # opencode 可能安装在 root 家目录（多用户环境常见）
    for candidate in [Path("/root/.opencode"), Path.home() / ".opencode"]:
        if (candidate / "bin" / "opencode").exists():
            return str(candidate)
    return str(Path.home() / ".opencode")


# agent 运行时必须可写的目录（内置默认 + bench_config.toml [sandbox] 段扩展）
_RUNTIME_WRITE_DIRS = [
    str(Path.home() / ".claude"),
    str(Path.home() / ".local"),       # opencode db、pip、各类运行时状态
    str(Path.home() / ".config"),      # opencode 配置、node 配置等
    str(Path.home() / ".cache"),       # pip cache、node cache、bun cache
    str(Path.home() / ".npm"),         # npm 全局缓存
    str(Path.home() / ".bun"),         # bun 运行时缓存
    _opencode_install_dir(),
    "/tmp",
]

# macOS 额外需要可写的目录（claude OAuth 缓存等）
_MACOS_EXTRA_WRITE_DIRS = [
    "/private/var",
]

# agent 运行时必须可写的单个文件（用 literal 而非 subpath）
_RUNTIME_WRITE_FILES = [
    str(Path.home() / ".claude.json"),  # claude 全局配置，Skill 执行时需要写
]

def wrap_command(cmd: list[str], config: SandboxConfig, workspace: Path) -> list[str]:
    """根据平台包裹命令，返回新的 cmd 列表。无沙箱工具时 warning 并返回原命令。"""
    system = platform.system()
    if system == "Darwin":
        return _wrap_macos(cmd, config, workspace)
    elif system == "Linux":
        return _wrap_linux(cmd, config, workspace)
    else:
        print(f"[sandbox] 不支持的平台: {system}，跳过沙箱", file=sys.stderr)
        return cmd


# ── macOS: sandbox-exec ──────────────────────────────────────────────────────

def _wrap_macos(cmd: list[str], config: SandboxConfig, workspace: Path) -> list[str]:
    """用 sandbox-exec 包裹命令。"""
    profile_path = workspace / ".sandbox-profile.sb"
    _write_macos_profile(profile_path, config, workspace)
    return ["sandbox-exec", "-f", str(profile_path), "--"] + cmd


def _write_macos_profile(path: Path, config: SandboxConfig, workspace: Path) -> None:
    """
    生成 Seatbelt profile。

    规则优先级：后出现的规则覆盖先出现的（last-match-wins）。
    必须用 (deny default) 而非 (allow default)，否则 deny 规则不生效。
    路径必须用 os.path.realpath()（macOS /tmp → /private/tmp）。
    """
    ws_real = os.path.realpath(str(workspace))

    # 写允许列表：workspace + 运行时目录 + macOS 额外目录
    write_dirs = [ws_real] + [os.path.realpath(d) for d in _RUNTIME_WRITE_DIRS + _MACOS_EXTRA_WRITE_DIRS]

    lines = [
        "(version 1)",
        '(deny default)',
        "",
        "; 进程基础权限",
        "(allow process-exec)",
        "(allow process-fork)",
        "(allow process-info* (target same-sandbox))",
        "(allow mach-priv-task-port (target same-sandbox))",  # Node/Bun spawn 子进程拿 task port
        "(allow signal (target same-sandbox))",
        "",
        "; 系统服务",
        "(allow sysctl-read)",
        "(allow mach-lookup)",
        "(allow ipc-posix-shm)",
        "(allow ipc-posix-sem)",
        "(allow iokit-get-properties)",
        "(allow user-preference-read)",
        "(allow pseudo-tty)",
        "(allow distributed-notification-post)",
        "",
        "; 网络：完全放开",
        "(allow network*)",
        "(allow system-socket)",
        "",
        "; 文件 ioctl（终端/设备）",
        "(allow file-ioctl)",
        "",
        "; /dev 设备文件：Node.js/Bun spawn 子进程时 stdio 重定向需要读写 /dev/null、/dev/tty、/dev/ptmx 等",
        '(allow file-read* (subpath "/dev"))',
        '(allow file-write* (subpath "/dev"))',
        "",
        "; 文件读：默认全放开，然后 deny 指定路径",
        "(allow file-read*)",
    ]

    # deny 指定路径的读取
    for p in config.deny_read:
        real_p = os.path.realpath(p)
        lines.append(f'(deny file-read* (subpath "{real_p}"))')

    # allow 例外（last-match-wins，覆盖 deny）
    for p in config.allow_read:
        real_p = os.path.realpath(p)
        lines.append(f'(allow file-read* (subpath "{real_p}"))')

    # 允许对目录做 stat（realpath 遍历需要）
    if config.deny_read:
        lines.append("(allow file-read-metadata (vnode-type DIRECTORY))")

    # 文件写：只允许 workspace + 运行时目录 + 特定文件
    lines.append("")
    lines.append("; 文件写：仅允许 workspace + 运行时目录")
    for d in write_dirs:
        lines.append(f'(allow file-write* (subpath "{d}"))')
    for f in _RUNTIME_WRITE_FILES:
        real_f = os.path.realpath(f)
        lines.append(f'(allow file-write* (literal "{real_f}"))')

    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


# ── Linux ────────────────────────────────────────────────────────────────────

def _wrap_linux(cmd: list[str], config: SandboxConfig, workspace: Path) -> list[str]:
    """Linux 文件隔离（bwrap）。不可用时抛出 RuntimeError。"""
    if shutil.which("bwrap") and _bwrap_works(workspace):
        return _wrap_linux_bwrap(cmd, config, workspace)
    raise RuntimeError(
        "[sandbox] bwrap 不可用，无法建立文件隔离。"
        "请确认 bwrap 已安装且 user namespace 可用（unshare --user --mount）。"
    )


_bwrap_ok: bool | None = None  # 缓存探测结果，每进程只探测一次


def _bwrap_works(workspace: Path) -> bool:
    """探测 bwrap 是否可用（用 --ro-bind / / 兼容 overlay/共享文件系统）。
    结果缓存，每进程只探测一次。"""
    global _bwrap_ok
    if _bwrap_ok is not None:
        return _bwrap_ok

    import subprocess
    ws_real = os.path.realpath(str(workspace))
    try:
        result = subprocess.run(
            [
                "bwrap",
                "--ro-bind", "/", "/",
                "--bind", ws_real, ws_real,
                "--dev", "/dev",
                "--proc", "/proc",
                "--die-with-parent",
                "--",
                "sh", "-c", f"echo ok > {ws_real}/.bwrap_test && cat {ws_real}/.bwrap_test",
            ],
            capture_output=True, text=True, timeout=10,
        )
        _bwrap_ok = result.returncode == 0 and "ok" in result.stdout
    except Exception:
        _bwrap_ok = False

    # 清理探测文件
    test_file = Path(ws_real) / ".bwrap_test"
    test_file.unlink(missing_ok=True)

    if not _bwrap_ok:
        print("[sandbox] bwrap 探测失败", file=sys.stderr)

    return _bwrap_ok


# ── Linux: bwrap ─────────────────────────────────────────────────────────────

def _wrap_linux_bwrap(cmd: list[str], config: SandboxConfig, workspace: Path) -> list[str]:
    """用 bwrap 包裹命令。

    挂载顺序（顺序决定覆盖关系）：
      1. --ro-bind / /          整棵 root 只读（overlay/共享文件系统兼容）
      2. runtime dir/file bind  运行时目录/文件可写（在 deny tmpfs 之前）
      3. --bind workspace        workspace 可写
      4. --tmpfs deny_path       屏蔽禁止路径（覆盖步骤 2 的父目录绑定）
         --dir / --bind          在 tmpfs 内恢复 allow/workspace 子路径

    关键：步骤 4 的 --tmpfs 必须在步骤 2 的父目录 bind 之后执行，
    否则父目录 bind 会覆盖已生效的 --tmpfs。
    """
    ws_real = os.path.realpath(str(workspace))
    bwrap_cmd = ["bwrap"]

    # 1. 整棵 root 只读（overlay/共享文件系统兼容）
    bwrap_cmd.extend(["--ro-bind", "/", "/"])
    bwrap_cmd.extend(["--proc", "/proc"])
    bwrap_cmd.extend(["--dev", "/dev"])

    # 2. 运行时目录/文件可写（必须在 deny --tmpfs 之前，否则父目录 bind 会覆盖 tmpfs）
    for d in _RUNTIME_WRITE_DIRS:
        real_d = os.path.realpath(d)
        if os.path.isdir(real_d):
            bwrap_cmd.extend(["--bind", real_d, real_d])
    for f in _RUNTIME_WRITE_FILES:
        real_f = os.path.realpath(f)
        if os.path.isfile(real_f):
            bwrap_cmd.extend(["--bind", real_f, real_f])

    # 3. workspace 可写
    bwrap_cmd.extend(["--bind", ws_real, ws_real])

    # 4. deny 路径用 --tmpfs 覆盖（内核级，对 Go 直接 syscall 也有效）
    #    若 allow_read 或 workspace 落在 deny 目录内，先 --dir 建挂载点，再 --bind 恢复
    paths_to_restore: set[str] = {ws_real}
    for p in config.allow_read:
        paths_to_restore.add(os.path.realpath(p))

    for deny_p in config.deny_read:
        real_deny = os.path.realpath(deny_p)
        if not os.path.isdir(real_deny):
            continue
        bwrap_cmd.extend(["--tmpfs", real_deny])

        for restore_p in paths_to_restore:
            if not restore_p.startswith(real_deny + "/"):
                continue
            rel = os.path.relpath(restore_p, real_deny)
            current = real_deny
            for part in rel.split(os.sep):
                current = os.path.join(current, part)
                bwrap_cmd.extend(["--dir", current])
            if os.path.exists(restore_p):
                bwrap_cmd.extend(["--bind", restore_p, restore_p])

    bwrap_cmd.extend(["--die-with-parent", "--"])
    return bwrap_cmd + cmd
