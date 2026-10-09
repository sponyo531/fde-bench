"""发现实验结果目录，并消除同一 run 的重试副本。"""

from __future__ import annotations

import json
import math
from datetime import datetime, timezone
from pathlib import Path
from typing import Iterable


def _manifest_identity(run_dir: Path) -> tuple[str, int, str]:
    """返回 run_id 和用于重试取舍的稳定排序键。"""
    manifest = run_dir / "manifest.json"
    try:
        data = json.loads(manifest.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        data = {}
    run_id = str(data.get("run_id") or run_dir.name)
    try:
        mtime_ns = manifest.stat().st_mtime_ns
    except OSError:
        mtime_ns = 0
    try:
        raw = str(data.get("started_at") or "").replace("Z", "+00:00")
        started = datetime.fromisoformat(raw)
        if started.tzinfo is None:
            started = started.replace(tzinfo=timezone.utc)
        order_ns = int(started.timestamp() * 1_000_000_000)
    except (ValueError, OverflowError):
        order_ns = mtime_ns
    return run_id, order_ns, str(run_dir)


def discover_run_dirs(roots: Iterable[str | Path], *, deduplicate: bool = True) -> list[Path]:
    """支持直接与 batch 布局；同一 run_id 默认保留最新一次尝试。

    只扫描固定的 ``run`` 和 ``batch/results/run`` 层级，避免进入 workspace 或
    ``.agent_home/node_modules``。重试按 started_at/mtime 取最新，不按分数择优。
    """
    found: dict[str, Path] = {}
    for root_arg in roots:
        root = Path(root_arg)
        if not root.is_dir():
            raise FileNotFoundError(f"results 目录不存在: {root}")

        candidates: list[Path] = []
        if (root / "manifest.json").is_file():
            candidates.append(root)
        candidates.extend(
            d for d in root.iterdir()
            if d.is_dir() and (d / "manifest.json").is_file()
        )
        candidates.extend(mf.parent for mf in root.glob("results/*/manifest.json"))
        candidates.extend(mf.parent for mf in root.glob("*/results/*/manifest.json"))
        # canonical layout: model/case/condition/tryN/{manifest,usage,result}.json
        candidates.extend(mf.parent for mf in root.glob("*/*/*/try*/manifest.json"))
        candidates.extend(mf.parent for mf in root.glob("*/*/*/try*/results/*/manifest.json"))
        for run_dir in candidates:
            found.setdefault(str(run_dir.resolve()), run_dir)

    dirs = list(found.values())
    if not deduplicate:
        return sorted(dirs)

    latest: dict[str, tuple[tuple[int, str], Path]] = {}
    for run_dir in dirs:
        run_id, order_ns, path = _manifest_identity(run_dir)
        rank = (order_ns, path)
        if run_id not in latest or rank > latest[run_id][0]:
            latest[run_id] = (rank, run_dir)
    return sorted(item[1] for item in latest.values())


def _finite_number(value: object) -> bool:
    return (isinstance(value, (int, float)) and not isinstance(value, bool)
            and math.isfinite(value))


def derive_run_statuses(run_dir: str | Path) -> dict[str, str]:
    """Derive the four orthogonal run outcomes from the on-disk ledger.

    ``usage.status`` describes whether the agent process finished; it must not be
    overloaded to mean that an artifact was complete or that scoring succeeded.
    The derived fields are deliberately conservative and never turn a missing
    score into zero.
    """
    run = Path(run_dir)
    try:
        usage = json.loads((run / "usage.json").read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        usage = {}
    try:
        result = json.loads((run / "result.json").read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        result = {}

    run_status = str(usage.get("status") or "unknown")
    produced = usage.get("produced")
    if not isinstance(produced, list):
        produced = []

    # A complete artifact is one the evaluator actually consumed (a numeric score)
    # or the canonical solution file was captured.  Other changed files are useful
    # evidence, but only prove that the agent wrote something, hence ``partial``.
    scored = any(_finite_number(result.get(k))
                 for k in ("overall_score", "combined_score", "final_score"))
    has_solution = any(Path(str(p)).name == "solution.json" for p in produced)
    explicit_artifact = result.get("artifact_status")
    if explicit_artifact in {"none", "partial", "complete"}:
        artifact_status = explicit_artifact
    elif not produced:
        artifact_status = "none"
    elif scored or has_solution:
        artifact_status = "complete"
    else:
        artifact_status = "partial"

    # New runs may explicitly persist a terminal state even when no evaluator
    # was invoked (for example, the agent produced no deliverable).  Honour
    # that state instead of inferring ``unscored`` from a missing numeric score.
    explicit_scoring = result.get("scoring_status")
    if explicit_scoring in {"scored", "failed", "unscored", "not_applicable"}:
        scoring_status = explicit_scoring
    else:
        scoring_error = result.get("error") or (
            isinstance(result.get("error_info"), dict)
            and result["error_info"].get("scoring_failed")
        )
        if scoring_error:
            scoring_status = "failed"
        elif scored:
            scoring_status = "scored"
        else:
            scoring_status = "unscored"

    validity = result.get("validity_score", result.get("validity"))
    if _finite_number(validity):
        validity_status = "valid" if float(validity) > 0 else "invalid"
    else:
        validity_status = "unknown"

    return {
        "run_status": run_status,
        "artifact_status": artifact_status,
        "scoring_status": scoring_status,
        "validity_status": validity_status,
    }


def update_run_statuses(run_dir: str | Path) -> dict[str, str]:
    """Persist :func:`derive_run_statuses` in ``usage.json`` and return them."""
    run = Path(run_dir)
    statuses = derive_run_statuses(run)
    path = run / "usage.json"
    try:
        data = json.loads(path.read_text(encoding="utf-8")) if path.is_file() else {}
    except (OSError, json.JSONDecodeError):
        data = {}
    data.update(statuses)
    path.write_text(json.dumps(data, ensure_ascii=False, indent=2), encoding="utf-8")
    return statuses
