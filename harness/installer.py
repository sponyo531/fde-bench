"""
harness/installer.py — 将整装 agent 配置目录安装到 workspace。

设计原则：配置目录（如 <repo>/opencode/）是自包含的"整装"形态，
installer 只做拷贝 + 本地化渲染，不做任何拼装（无 frontmatter 合并、
无 prompt variant 继承、无 skill 注入）。想改 agent 行为，改配置目录本身。

opencode 平台的本地化渲染（opencode.jsonc.template → opencode.jsonc）：
  1. provider 置空 —— 本地评测的 provider/model 由运行者的全局
     ~/.config/opencode 配置提供（或 --model 别名注入），仓库不管理
  2. 删除 model / small_model —— 同上
  3. plugin 绝对路径（/root/.config/opencode/plugins/x.ts）→ 相对路径（./plugins/x.ts）
  4. 其余（default_agent、agent 权限、title prompt）原样保留，
     与部署行为自动对齐
"""

import json
import re
import shutil
import sys
from pathlib import Path

# 拷贝时排除的部署/构建专用文件（workspace 运行不需要）
_COPY_EXCLUDES = (
    "node_modules",            # 单独处理（见 _install_node_modules）
    "opencode.jsonc.template", # 单独渲染（见 _render_local_jsonc）
    "Dockerfile",
    "entrypoint.sh",
    "tests",
    "README.md",
    "LOGGING.md",
    "bun.lock",
    ".git",
    ".DS_Store",
)


def install_agent_config(config_dir: Path, workspace: Path, platform: str) -> None:
    """
    把整装配置目录安装到 workspace 的平台目录（.opencode/ 或 .claude/）。

    config_dir : 整装配置目录（如 <repo>/opencode/、agent_configs/claude/）
    workspace  : agent 工作目录
    platform   : ".opencode" | ".claude"
    """
    dst = workspace / platform
    if dst.exists():
        shutil.rmtree(dst)
    shutil.copytree(
        config_dir, dst,
        ignore=shutil.ignore_patterns(*_COPY_EXCLUDES),
    )

    if platform == ".opencode":
        _install_node_modules(config_dir, dst)
        _render_local_jsonc(config_dir, dst)


def _install_node_modules(config_dir: Path, dst: Path) -> None:
    """
    拷贝预装的 node_modules，保证 opencode 运行时零 npm install
    （并发评测下运行时 install 是 flaky 源头）。

    仅当配置目录带 TS 插件/工具（plugins/ 或 tools/）时才强制要求；
    缺失时直接报错，不静默降级回运行时 install。
    """
    src = config_dir / "node_modules"
    if src.is_dir():
        shutil.copytree(src, dst / "node_modules", symlinks=True)
        return

    needs_ts = (config_dir / "plugins").is_dir() or (config_dir / "tools").is_dir()
    if needs_ts:
        raise RuntimeError(
            f"配置目录含 plugins/tools 但缺少 node_modules/: {config_dir}\n"
            f"请先执行一次性预装:  cd {config_dir} && bun install"
        )


def _load_jsonc(text: str) -> dict:
    """宽松解析 jsonc：去掉 // 注释与尾逗号。"""
    stripped = re.sub(r"^\s*//.*$", "", text, flags=re.MULTILINE)
    stripped = re.sub(r",\s*([}\]])", r"\1", stripped)
    return json.loads(stripped)


def _render_local_jsonc(config_dir: Path, dst: Path) -> None:
    """
    opencode.jsonc.template → workspace 的 opencode.jsonc（本地化渲染）。

    优先级：配置目录自带的 opencode.jsonc（本地终态，已随 copytree 拷入）
    > opencode.jsonc.template（需渲染）。两者都存在时终态文件生效，跳过渲染。
    """
    template = config_dir / "opencode.jsonc.template"
    if not template.is_file():
        return
    if (config_dir / "opencode.jsonc").is_file():
        print(
            f"[installer] {config_dir.name}: 检测到 opencode.jsonc（本地终态），"
            f"跳过 template 渲染",
            file=sys.stderr,
        )
        return

    data = _load_jsonc(template.read_text(encoding="utf-8"))

    # 本地 provider/model 由运行者全局配置或 --model 注入，剥掉部署侧的
    data["provider"] = {}
    data.pop("model", None)
    data.pop("small_model", None)

    # 部署绝对路径 → workspace 相对路径
    plugins = data.get("plugin")
    if isinstance(plugins, list):
        data["plugin"] = [
            f"./plugins/{p.rsplit('/', 1)[-1]}" if p.startswith("/") else p
            for p in plugins
        ]

    (dst / "opencode.jsonc").write_text(
        json.dumps(data, indent=2, ensure_ascii=False), encoding="utf-8"
    )
