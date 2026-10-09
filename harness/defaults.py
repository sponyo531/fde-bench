"""
harness/defaults.py — 共享常量

平台只有两个：cc（Claude Code CLI）和 opencode（OpenCode CLI）。
跑什么 agent 由 --agent-config 指定的整装配置目录决定；
不注入配置（--agent-config none）即为平台原生 agent。
"""

from pathlib import Path

from .model_endpoints import bare_model, is_responses_model

# 仓库根（harness/ 的上一级）
_PKG_ROOT = Path(__file__).parent.parent

# 主真源：自包含、可部署的 OpenCode 配置目录
OPENCODE_CONFIG_DIR = _PKG_ROOT / "opencode"
# 非部署配置（CC 降级 + 实验变体），各自整装
AGENT_CONFIGS_DIR = _PKG_ROOT / "agent_configs"

PLATFORMS = ("opencode", "cc")

# 各平台不指定 --agent-config 时的默认整装配置目录
DEFAULT_AGENT_CONFIG: dict[str, Path] = {
    "opencode": OPENCODE_CONFIG_DIR,
    "cc":       AGENT_CONFIGS_DIR / "claude",
}


def detect_platform(config_dir: Path) -> str:
    """从整装配置目录推断平台：含 settings.json → cc，否则 opencode。"""
    if (config_dir / "settings.json").is_file():
        return "cc"
    return "opencode"


# ── 模型别名 ─────────────────────────────────────────────────────────────────
# CLI --model 接受别名或完整 model ID。别名由各 backend 翻译为平台格式。
# 完整 ID（不在别名表中的值）原样透传给对应平台。
# 默认不传 --model：跟随运行者的全局配置（opencode: ~/.config/opencode；cc: 全局 settings）。

MODEL_ALIASES: dict[str, dict[str, str]] = {
    "sonnet":   {"cc": "claude-sonnet-4-6",  "opencode": "anthropic/claude-sonnet-4-6"},
    "opus":     {"cc": "claude-opus-4-6",    "opencode": "anthropic/claude-opus-4-6"},
    "haiku":    {"cc": "claude-haiku-4-5",   "opencode": "anthropic/claude-haiku-4-5"},
}

def resolve_model(alias: str | None, platform: str) -> str | None:
    """将 model 别名解析为平台特定的 model ID。

    - None → None（不注入，跟随运行者全局配置）
    - 已知别名 → 对应平台 ID
    - 其他值 → 原样透传
    """
    if alias is None:
        return None
    key = "opencode" if platform.startswith("opencode") else "cc"
    if key == "opencode" and "/" not in alias and is_responses_model(alias):
        return f"responses/{bare_model(alias)}"
    entry = MODEL_ALIASES.get(alias)
    if entry:
        return entry[key]
    return alias
