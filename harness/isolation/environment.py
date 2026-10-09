"""中立运行环境。

opencode 会从多处加载配置与 skill：
  - $XDG_CONFIG_HOME/opencode/     用户全局配置（provider、model、agent、权限）
  - 项目目录下的 .claude/skills/   项目级 skill

评测必须屏蔽 agent/skill 那部分——否则分数反映的是运行者本机装了哪些
skill，而不是模型能力。实测中曾观察到 agent 自动加载本机 8 个业务 skill
并以 agent=build 模式运行，那样的结果不具可比性。

但 provider 凭证通常也写在同一份全局配置里，不能一并丢弃，否则模型无法
调用。因此这里的做法是：**只继承 provider 与 model，其余一律不继承**，
并把 XDG 各目录指向 run 目录内的私有位置。
"""

from __future__ import annotations

import json
import os
from pathlib import Path

# 允许从宿主全局配置继承的键——仅连接必需项
_INHERIT_KEYS = ("provider", "model", "small_model")

# 模型输出上限兜底（token）。
#
# opencode 对不在其模型注册表里的模型套用内置默认 4096。推理型模型的思维链
# 一轮就能吃满 4096，导致 finish_reason=length、正式回答 0 字符——表面看像
# "agent 什么都没做"，实际是预算不够它开口。实测 kimi-k3 在 4096 下产出
# 16852 字符思维链、0 字符回答；提到 65536 后恢复正常。
#
# 该兜底写在仓库内而非宿主配置里，确保换机器、换运行者都不会静默退回 4096。
def _limits() -> tuple[int, int]:
    from ..config import get
    return (int(get("agent", "min_output_tokens", 65536)),
            int(get("agent", "default_context_tokens", 262144)))


_MIN_OUTPUT_TOKENS, _DEFAULT_CONTEXT_TOKENS = _limits()

# 被测 agent **不改任何思考配置**，一律用厂商默认。
#
# 曾按前缀关掉 glm 系的思维链，理由是实测 max_tokens=64 时思维链吃光预算、
# content 为空。但那个病根是输出预算太小，已由 min_output_tokens=65536 兜住；
# 而"只关一家"在 7 模型主表里站不住——等于阉割了推理型模型的一项能力，
# 跨模型的 F−R 比较因此不干净。
#
# 与本文件开头的原则一致：只继承 provider/model，不注入任何行为配置，
# 保持平台原生形态。辅助角色（answerer/judge/ask_detect）另有口径——
# 它们只需简短结论且不是被测对象，仍关思考，见 harness/llm_client.py。


def _ensure_output_limit(cfg: dict, model: str | None = None) -> list[str]:
    """为所有模型补齐输出上限。返回被调整的模型名，供 manifest 记录。

    被测模型若不在宿主配置的 models 表里（如宿主只登记了 kimi-k3，本次却跑
    glm-5.2），必须先补一个条目——否则 opencode 找不到注册信息，套用内置默认
    4096，推理型模型的思维链一轮吃满，正式回答 0 字符。实测 glm-5.2 在
    max_tokens=64 时 finish_reason=length 且 content 为空，512 才正常。
    """
    adjusted: list[str] = []
    if model and "/" in model:
        pname, mname = model.split("/", 1)
        prov = (cfg.setdefault("provider", {})).get(pname)
        if prov is not None:
            entry = prov.setdefault("models", {}).setdefault(mname, {"name": mname})
    for pname, provider in (cfg.get("provider") or {}).items():
        for mname, m in (provider.get("models") or {}).items():
            limit = m.setdefault("limit", {})
            if limit.get("output", 0) < _MIN_OUTPUT_TOKENS:
                limit["output"] = _MIN_OUTPUT_TOKENS
                limit.setdefault("context", _DEFAULT_CONTEXT_TOKENS)
                adjusted.append(f"{pname}/{mname}")
    return adjusted


def _host_config() -> dict:
    """读取宿主的 opencode 全局配置（若存在）。"""
    explicit = os.environ.get("FDE_OPENCODE_CONFIG")
    if explicit:
        from ..installer import _load_jsonc
        path = Path(explicit).expanduser()
        if not path.is_file():
            raise FileNotFoundError(f"FDE_OPENCODE_CONFIG does not exist: {path}")
        return _load_jsonc(path.read_text(encoding="utf-8"))
    candidates = [
        Path(os.environ.get("XDG_CONFIG_HOME", Path.home() / ".config")) / "opencode" / "opencode.json",
        Path.home() / ".config" / "opencode" / "opencode.json",
    ]
    for p in candidates:
        if p.is_file():
            try:
                return json.loads(p.read_text(encoding="utf-8"))
            except json.JSONDecodeError:
                continue
    return {}


def _seed_models_cache(cache_home: Path) -> None:
    """把宿主的 models.json 复制进隔离 cache。

    opencode 启动时会拉取 models.dev 的模型清单，失败即退出。本机
    /etc/hosts 将 models.dev 指向 127.0.0.1（离线环境常见做法），
    因此必须继承这份缓存——它是公共模型元数据，不含任何 skill 或配置。
    """
    import shutil
    src = Path(os.environ.get("XDG_CACHE_HOME", Path.home() / ".cache")) / "opencode" / "models.json"
    if not src.is_file():
        src = Path.home() / ".cache" / "opencode" / "models.json"
    if src.is_file():
        dst = cache_home / "opencode" / "models.json"
        dst.parent.mkdir(parents=True, exist_ok=True)
        if not dst.exists():
            shutil.copy2(src, dst)


def _seed_provider_sdk(cfg_home: Path) -> None:
    """把宿主已安装的 provider SDK 带进隔离配置目录。

    provider 的 "npm" 字段（如 @ai-sdk/openai-compatible）需要真实安装包。
    隔离目录里没有 node_modules 时 opencode 会尝试联网安装，离线环境下
    provider 加载失败，而 CLI 只报一句含混的 "Session not found"。

    这里只搬 SDK 包本身——它是运行依赖，不是 agent 行为设定，
    不影响 agent / skill / permission 的中立性。
    """
    import shutil
    src_dir = Path.home() / ".config" / "opencode"
    dst_dir = cfg_home / "opencode"
    for name in ("node_modules", "package.json", "package-lock.json"):
        src, dst = src_dir / name, dst_dir / name
        if not src.exists() or dst.exists():
            continue
        if src.is_dir():
            shutil.copytree(src, dst, symlinks=True)
        else:
            shutil.copy2(src, dst)


def _seed_responses_provider_sdk(cfg_home: Path) -> None:
    """Expose the pinned official OpenAI AI SDK from the staged code snapshot.

    Older production images only bake ``@ai-sdk/openai-compatible``.  The code
    snapshot already carries the official provider and its dependencies for the
    extractor, so symlink those packages into the run-private OpenCode config.
    This makes the Responses migration effective immediately, without mutating
    the image or copying thousands of shared-filesystem files per run.
    """
    source = Path(__file__).resolve().parents[2] / "harness" / "scoring" / "extractor_opencode" / "node_modules"
    if not (source / "@ai-sdk" / "openai").is_dir():
        return
    target = cfg_home / "opencode" / "node_modules"
    target.mkdir(parents=True, exist_ok=True)
    for src in source.iterdir():
        if src.name == "@ai-sdk":
            scoped = target / src.name
            scoped.mkdir(exist_ok=True)
            for package in src.iterdir():
                dst = scoped / package.name
                if not dst.exists():
                    dst.symlink_to(package, target_is_directory=True)
            continue
        dst = target / src.name
        if not dst.exists():
            dst.symlink_to(src, target_is_directory=src.is_dir())


def build_env(run_dir: Path, model: str | None = None) -> dict[str, str]:
    """构造隔离环境；在 run_dir 下建立私有 XDG 目录树与最小配置。"""
    private = run_dir / ".agent_home"
    cfg_home, data_home, cache_home = private / "config", private / "data", private / "cache"
    (cfg_home / "opencode").mkdir(parents=True, exist_ok=True)
    data_home.mkdir(parents=True, exist_ok=True)
    cache_home.mkdir(parents=True, exist_ok=True)

    _seed_models_cache(cache_home)

    host = _host_config()
    cfg: dict = {"$schema": "https://opencode.ai/config.json"}
    for k in _INHERIT_KEYS:
        if k in host:
            cfg[k] = host[k]
    if model:
        cfg["model"] = model
    # 不继承 agent / skill / permission / mcp —— 保持平台原生形态

    # 显式禁用联网取数工具。三条理由：
    #  1. 污染：59 个 case 中 6 个源自公开数据集（Solomon VRPTW 等已发表最优解），
    #     agent 检索到题面即可直接抄答案；
    #  2. 无收益：考点是客户脑中的隐式业务约定（哪行是仓库、速度怎么分段），
    #     公网检索不到。InteractComp 实测 search-only 仅 6.7–9.5%，
    #     而拿到真实上下文可达 40.9–71.5%；
    #  3. 可复现：检索结果随时间漂移，跨批次分数不可比。
    # 实测该工具本就零调用（86 次 bash / 33 次 edit / 0 次 webfetch），
    # 故显式关闭不改变既有行为，只是消除不确定性。
    cfg["permission"] = {**(cfg.get("permission") or {}), "webfetch": "deny"}

    # opencode 按 schema 校验 opencode.json，未知顶层字段会让整份配置解析失败，
    # 而 CLI 只报一句含混的 "Session not found"。所以调整记录只写 manifest，
    # 绝不写进配置文件本身。
    adjusted = _ensure_output_limit(cfg, model)
    if adjusted:
        (run_dir / ".agent_home" / "output_limit_note.txt").write_text(
            f"output limit raised to {_MIN_OUTPUT_TOKENS} for: {', '.join(adjusted)}\n",
            encoding="utf-8",
        )

    (cfg_home / "opencode" / "opencode.json").write_text(
        json.dumps(cfg, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    _seed_provider_sdk(cfg_home)
    _seed_responses_provider_sdk(cfg_home)

    env = dict(os.environ)
    env.update({
        "XDG_CONFIG_HOME": str(cfg_home),
        "XDG_DATA_HOME": str(data_home),
        "XDG_CACHE_HOME": str(cache_home),
    })
    return env


def describe(run_dir: Path) -> dict:
    """回报本次实际生效的配置，写进 manifest 供复现核对。"""
    cfg_file = run_dir / ".agent_home" / "config" / "opencode" / "opencode.json"
    if not cfg_file.is_file():
        return {}
    cfg = json.loads(cfg_file.read_text(encoding="utf-8"))
    limits = {
        f"{p}/{m}": (mc.get("limit") or {}).get("output")
        for p, pc in (cfg.get("provider") or {}).items()
        for m, mc in (pc.get("models") or {}).items()
    }
    note_file = run_dir / ".agent_home" / "output_limit_note.txt"
    out = {
        "model": cfg.get("model"),
        "providers": sorted((cfg.get("provider") or {}).keys()),
        "provider_sdks": {
            p: pc.get("npm") for p, pc in (cfg.get("provider") or {}).items()
        },
        "output_limits": limits,
        "inherited_keys": [k for k in _INHERIT_KEYS if k in cfg],
    }
    if note_file.is_file():
        out["output_limit_note"] = note_file.read_text(encoding="utf-8").strip()
    return out
