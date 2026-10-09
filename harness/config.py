"""统一配置：读 config.toml。

一份文件覆盖 agent / judge / answerer / clarify / sandbox / pricing。
token 例外——只在配置里写环境变量名，实际密钥由环境变量提供，避免入库。

环境变量可覆盖配置文件（DELIVER_{SECTION}_{KEY}），便于临时切换而不改仓库文件。

为什么 judge 与 answerer 必须锁定模型版本：两者的输出都直接进入评分口径。
judge 一换，Ask-F1 / CE-A / CE-B 的基准就变；answerer 一换，C 条件的信息上限
就变，C−R 落差的含义随之改变。故实际生效的配置会写进每次 run 的结果供核对。
"""

from __future__ import annotations

import os
import tomllib
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path

CONFIG_PATH = Path(__file__).resolve().parent.parent / "config.toml"
EXAMPLE_CONFIG_PATH = CONFIG_PATH.with_name("config.toml.example")


@lru_cache(maxsize=1)
def load() -> dict:
    path = CONFIG_PATH if CONFIG_PATH.is_file() else EXAMPLE_CONFIG_PATH
    if not path.is_file():
        raise SystemExit(f"缺少配置文件 {CONFIG_PATH.name} 和示例配置。")
    try:
        with path.open("rb") as fh:
            return tomllib.load(fh)
    except tomllib.TOMLDecodeError as e:
        raise SystemExit(f"{path.name} 解析失败: {e}") from e


def section(name: str) -> dict:
    return load().get(name, {})


def get(sec: str, key: str, default=None):
    """取配置项，环境变量 DELIVER_{SEC}_{KEY} 优先。"""
    env = os.environ.get(f"DELIVER_{sec.upper()}_{key.upper()}")
    if env is not None and env != "":
        return env
    return section(sec).get(key, default)


@dataclass(frozen=True)
class LLMRole:
    """judge / answerer 的完整配置。"""
    role: str
    model: str
    protocol: str
    base_url: str
    token: str
    max_tokens: int = 4096
    style: str = "default"

    def describe(self) -> dict:
        """写入结果供复现核对——不含 token。"""
        return {
            "role": self.role,
            "model": self.model,
            "protocol": self.protocol,
            "base_url": self.base_url,
        }


def load_role(role: str) -> LLMRole:
    cfg = section(role)
    token_env = cfg.get("token_env", f"DELIVER_{role.upper()}_TOKEN")
    token = os.environ.get(token_env, "") or os.environ.get(f"DELIVER_{role.upper()}_TOKEN", "")

    model = get(role, "model", "")
    base_url = get(role, "base_url", "")
    if model.startswith(("direct/", "responses/")):
        from .model_endpoints import endpoint
        base_url, token = endpoint(model)
    if not model or not base_url:
        raise SystemExit(
            f"[{role}] 配置缺失 model / base_url。"
            f"请在 {CONFIG_PATH.name} 的 [{role}] 段填写。"
        )
    if not token:
        raise SystemExit(
            f"[{role}] 缺少 API token。请设环境变量 {token_env}（密钥不写入仓库）。"
        )

    return LLMRole(
        role=role,
        model=model,
        protocol=get(role, "protocol", "openai"),
        base_url=base_url,
        token=token,
        max_tokens=int(cfg.get("max_tokens", 4096)),
        style=get(role, "style", "default"),
    )


def make_client(role: LLMRole):
    from .llm_client import ChatLLM
    from .model_endpoints import endpoint
    model = role.model
    base_url = role.base_url
    token = role.token
    if model.startswith(("direct/", "responses/")):
        base_url, token = endpoint(model)
    return ChatLLM(
        protocol=role.protocol,
        base_url=base_url,
        auth_token=token,
        model=model,
    )


@lru_cache(maxsize=1)
def pricing() -> dict:
    """模型单价，读自 pricing.toml。

    与 config.toml 分开：单价是公共数据、应进版本库共享；config.toml 含
    API 端点被 gitignore，两者生命周期不同。
    """
    path = CONFIG_PATH.parent / "pricing.toml"
    if not path.is_file():
        return {}
    try:
        with path.open("rb") as fh:
            return tomllib.load(fh)
    except tomllib.TOMLDecodeError as e:
        raise SystemExit(f"pricing.toml 解析失败: {e}") from e
