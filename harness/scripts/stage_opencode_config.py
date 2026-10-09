#!/usr/bin/env python3
"""把 FDE-bench 的 opencode.jsonc stage 成 harness 能读的严格 JSON。

为什么需要这一步（三个不匹配，缺一个链路就断）：

1. **文件名**：`harness/isolation/environment.py::_host_config()` 只找
   `$XDG_CONFIG_HOME/opencode/opencode.json`，而我们的源文件叫 `.jsonc`。
2. **注释**：它用 `json.loads` 读，`//` 注释会直接 JSONDecodeError，
   而那个 except 分支是 `continue` —— 于是静默退回 `{}`，provider 为空。
3. **只取三个键**：environment.py 的 `_INHERIT_KEYS` 只继承
   provider / model / small_model。这里也只写这三个，多写的会被丢弃，
   留着只是让 stage 出来的文件看着像"全量配置"，误导排查。

失败模式长什么样（都踩过）：provider 为空 → opencode 只加载内置 provider →
`ProviderModelNotFoundError` 写进它自己的日志，而 CLI 那头 **exit 0**，
harness 记一条 3 秒、0 tool_call、空 response.txt 的 run，报 timeout。
表面像"模型不干活"，实则配置根本没进去。所以本脚本宁可 exit≠0 也不静默降级。

用法：
    python3 stage_opencode_config.py --src /path/to/private/opencode.jsonc \
                                     --dst <run>/hostcfg --require-model direct/glm-5.3
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

_INHERIT_KEYS = ("provider", "model", "small_model")   # 与 environment.py 保持一致


def strip_jsonc(raw: str) -> str:
    """去掉 `//` 行注释与行尾注释，保留字符串内的 `//`（URL 里就有）。

    逐字符扫而不是正则：`"baseURL": "https://example.invalid/v1"` 里的
    `//` 在引号内，正则版会把整行后半截切掉，于是 baseURL 变成 `"https:`，
    provider 加载失败——又是一个静默降级。
    """
    out = []
    for line in raw.splitlines():
        in_str = False
        escaped = False
        cut = None
        for i, ch in enumerate(line):
            if escaped:
                escaped = False
                continue
            if ch == "\\":
                escaped = True
                continue
            if ch == '"':
                in_str = not in_str
                continue
            if not in_str and ch == "/" and line[i + 1:i + 2] == "/":
                cut = i
                break
        out.append(line if cut is None else line[:cut])
    return "\n".join(out)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--src", required=True, help="opencode.jsonc 源文件")
    ap.add_argument("--dst", required=True, help="目标 XDG_CONFIG_HOME 目录")
    ap.add_argument("--require-model", default="", help="断言这个模型在白名单里")
    a = ap.parse_args()

    src = Path(a.src)
    if not src.is_file():
        print(f"FATAL: 找不到 {src}", file=sys.stderr)
        return 66

    try:
        cfg = json.loads(strip_jsonc(src.read_text(encoding="utf-8")))
    except json.JSONDecodeError as e:
        print(f"FATAL: {src} 去注释后仍不是合法 JSON：{e}", file=sys.stderr)
        return 65

    out = {"$schema": "https://opencode.ai/config.json"}
    for k in _INHERIT_KEYS:
        if k in cfg:
            out[k] = cfg[k]

    providers = out.get("provider") or {}
    if not providers:
        print("FATAL: provider 为空。这正是「CLI 秒退且 exit 0」的根因，不能放过。",
              file=sys.stderr)
        return 65

    # 白名单为空等于没配 —— opencode 会 UnknownError，所以在这里拦住。
    for pid, pc in providers.items():
        if not (pc.get("models") or {}):
            print(f"FATAL: provider {pid} 的 models 白名单为空", file=sys.stderr)
            return 65
        # Absolute local npm paths are not portable into an isolated runtime.
        npm = str(pc.get("npm", ""))
        if npm.startswith("file:") and ("/" in npm[7:] or "\\" in npm[7:]):
            print(f"FATAL: provider {pid} 的 npm 指向本地绝对路径，"
                  f"隔离运行环境中不可用：{npm}", file=sys.stderr)
            return 65

    if a.require_model:
        if "/" not in a.require_model:
            print(f"FATAL: --require-model 要写成 provider/model：{a.require_model}",
                  file=sys.stderr)
            return 64
        pid, mid = a.require_model.split("/", 1)
        if pid not in providers:
            print(f"FATAL: 没有 provider «{pid}»，现有：{', '.join(providers)}",
                  file=sys.stderr)
            return 65
        if mid not in (providers[pid].get("models") or {}):
            have = ", ".join(providers[pid]["models"])
            print(f"FATAL: «{mid}» 不在 {pid} 的白名单里。opencode 不会报错，"
                  f"只会 exit 0 交白卷。\n  白名单：{have}\n"
                  f"  补的办法：在 {src} 的 provider.{pid}.models 里加一条。",
                  file=sys.stderr)
            return 65

    dst = Path(a.dst) / "opencode"
    dst.mkdir(parents=True, exist_ok=True)
    (dst / "opencode.json").write_text(
        json.dumps(out, indent=2, ensure_ascii=False), encoding="utf-8")

    # 写完立刻回读 —— 共享挂载点可能有视图延迟，写成功不等于读得到。
    back = json.loads((dst / "opencode.json").read_text(encoding="utf-8"))

    n = sum(len(p.get("models") or {}) for p in back["provider"].values())
    print(f"[stage-cfg] {dst/'opencode.json'} ← {src}  "
          f"provider={len(back['provider'])} models={n} model={back.get('model')}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
