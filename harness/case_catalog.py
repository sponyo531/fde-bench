"""统一的 benchmark case 发现和发布版 case 解析。"""

from __future__ import annotations

from pathlib import Path
from typing import Iterable


def discover_cases(root: Path, *, clean_only: bool = True) -> list[str]:
    """返回可用于实验的 case 名。

    ``gt.json`` 是评分和澄清指标的必要文件；instruction/information/data 的完整性
    在 ``validate_case`` 中单独检查，避免把一个半成品目录静默纳入矩阵。

    发布包中的 case 已经完成去敏并按编号重命名，因此不要求 ``_clean`` 后缀。
    ``clean_only`` 仍保留用于兼容旧目录，并排除显式标记为 ``_raw`` 的目录。
    """
    if not root.is_dir():
        raise FileNotFoundError(f"case 根目录不存在: {root}")
    out = []
    for path in root.iterdir():
        if not path.is_dir() or not (path / "gt.json").is_file():
            continue
        if clean_only and path.name.endswith("_raw"):
            continue
        out.append(path.name)
    # In a mixed legacy tree, an explicit ``foo_clean`` is authoritative over
    # its raw sibling ``foo``. The release tree itself contains only the
    # already-clean, numbered names and is unaffected by this rule.
    if clean_only:
        clean_names = {name[:-6] for name in out if name.endswith("_clean")}
        out = [name for name in out if not (name in clean_names)]
    return sorted(out)


def validate_case(root: Path, case: str) -> list[str]:
    """检查 case 的运行必需文件，返回缺失项。"""
    path = root / case
    required = ("gt.json", "instruction.md", "information.md", "data")
    return [name for name in required if not (path / name).exists()]


def resolve_case_spec(spec, root: Path, *, clean_only: bool = True,
                      named_subsets: dict | None = None) -> list[str]:
    """把 ``all``、列表或命名子集解析成唯一的 clean case 名。

    子集历史上常写 raw 名（如 ``009_foo``），这里优先映射到
    ``009_foo_clean``，避免 raw/clean 双份同时进入实验。
    """
    available = discover_cases(root, clean_only=clean_only)
    avail = set(available)
    if spec == "all":
        selected = available
    elif isinstance(spec, list):
        selected = _resolve_names(spec, available, avail)
    elif named_subsets and str(spec) in named_subsets:
        selected = _resolve_names(named_subsets[str(spec)], available, avail)
    else:
        raise ValueError(f"cases={spec!r} 不是 all、列表或已定义子集")
    # 保持配置顺序，同时去重。
    return list(dict.fromkeys(selected))


def _resolve_names(names: Iterable[str], available: list[str], avail: set[str]) -> list[str]:
    out: list[str] = []
    for raw in names:
        name = str(raw)
        clean = name if name.endswith("_clean") else f"{name}_clean"
        if clean in avail:
            out.append(clean)
            continue
        if name in avail:
            out.append(name)
            continue
        prefix = f"{name}_"
        matches = [c for c in available if c.startswith(prefix)]
        if len(matches) == 1:
            out.extend(matches)
            continue
        raise ValueError(f"case {name!r} 在 clean case 根目录中不存在")
    return out
