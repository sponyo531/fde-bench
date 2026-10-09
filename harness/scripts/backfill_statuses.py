"""Backfill explicit run/scoring status records for historical experiments.

The migration is deliberately conservative:

* numeric scores are never changed;
* a missing ``result.json`` is materialized only for a terminal run, with a
  null score and an explicit reason;
* every file changed by ``--apply`` is copied to one timestamped backup tree;
* without ``--apply`` the command is a read-only report.

Usage::

    python3 -m harness.scripts.backfill_statuses ROOT
    python3 -m harness.scripts.backfill_statuses ROOT --apply
"""

from __future__ import annotations

import argparse
import json
import math
import shutil
import time
from pathlib import Path

from ..results import discover_run_dirs


_TERMINAL = {"ok", "timeout", "error", "degraded"}
_VALID_SCORING = {"scored", "failed", "unscored", "not_applicable"}
_VALID_ARTIFACT = {"none", "partial", "complete"}


def _read(path: Path) -> dict:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
        return value if isinstance(value, dict) else {}
    except (OSError, json.JSONDecodeError):
        return {}


def _number(value: object) -> bool:
    return isinstance(value, (int, float)) and not isinstance(value, bool) and math.isfinite(value)


def _produced(usage: dict) -> list[str]:
    value = usage.get("produced")
    return value if isinstance(value, list) else []


def _terminal(run: Path, usage: dict, result: dict) -> bool:
    status = str(usage.get("status") or "")
    if status not in _TERMINAL:
        return False
    # An Interact run with neither result nor clarify score may still have an
    # active scoring job.  Do not close that state during a migration.
    manifest = _read(run / "manifest.json")
    condition = str(manifest.get("condition") or "")
    if not result and condition.startswith("Interact"):
        if not (run / "clarify_score.json").is_file() and status == "ok":
            return False
    return True


def _classify(run: Path, usage: dict, result: dict) -> tuple[str, str, dict]:
    produced = _produced(usage)
    numeric = any(_number(result.get(k)) for k in ("final_score", "overall_score", "combined_score"))
    if result.get("scoring_status") in _VALID_SCORING:
        scoring = result["scoring_status"]
    elif result.get("error") or (
            isinstance(result.get("error_info"), dict)
            and result["error_info"].get("scoring_failed")):
        scoring = "failed"
    elif numeric:
        scoring = "scored"
    elif result:
        scoring = "unscored"
    elif produced:
        scoring = "unscored"
    else:
        scoring = "not_applicable"

    if result.get("artifact_status") in _VALID_ARTIFACT:
        artifact = result["artifact_status"]
    elif not produced:
        artifact = "none"
    elif numeric or any(Path(str(p)).name == "solution.json" for p in produced):
        artifact = "complete"
    else:
        artifact = "partial"

    reason = {
        "category": "artifact_missing" if artifact == "none" and scoring == "not_applicable"
        else ("scoring_failed" if scoring == "failed" else "unscored"),
        "message": (
            "历史 run 未发现 Agent 交付文件，评分不适用。" if artifact == "none" and scoring == "not_applicable"
            else "历史记录已有评分失败信息。" if scoring == "failed"
            else "历史 run 有产物但没有可用数值评分。"
        ),
    }
    return artifact, scoring, reason


def _backup(path: Path, root: Path, backup_root: Path) -> None:
    rel = path.relative_to(root)
    target = backup_root / rel
    target.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(path, target)


def _write_json(path: Path, data: dict, root: Path, backup_root: Path | None) -> None:
    if backup_root is not None and path.is_file():
        _backup(path, root, backup_root)
    tmp = path.with_suffix(path.suffix + ".backfill-tmp")
    tmp.write_text(json.dumps(data, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    tmp.replace(path)


def migrate(root: Path, *, apply: bool = False) -> dict:
    runs = discover_run_dirs([root], deduplicate=False)
    stamp = time.strftime("%Y%m%d-%H%M%S", time.gmtime())
    backup_root = root / "_backfill_backups" / stamp if apply else None
    counts: dict[str, int] = {"scanned": 0, "changed": 0, "skipped_active": 0,
                              "missing_result": 0, "result_written": 0}
    changes: list[dict] = []
    for run in runs:
        usage_path = run / "usage.json"
        manifest = _read(run / "manifest.json")
        usage = _read(usage_path)
        if not usage:
            continue
        result_path = run / "result.json"
        result = _read(result_path) if result_path.is_file() else {}
        counts["scanned"] += 1
        if not _terminal(run, usage, result):
            counts["skipped_active"] += 1
            continue
        artifact, scoring, reason = _classify(run, usage, result)
        old_statuses = {k: usage.get(k) for k in ("artifact_status", "scoring_status", "validity_status")}
        validity = result.get("validity_score", result.get("validity"))
        validity_status = "valid" if _number(validity) and float(validity) > 0 else \
            "invalid" if _number(validity) else "unknown"
        changed = old_statuses != {
            "artifact_status": artifact,
            "scoring_status": scoring,
            "validity_status": validity_status,
        } or not result_path.is_file()
        if not changed:
            continue
        counts["changed"] += 1
        if not result_path.is_file():
            counts["missing_result"] += 1
        entry = {
            "run": str(run),
            "case": manifest.get("case"),
            "condition": manifest.get("condition"),
            "model": manifest.get("model"),
            "artifact_status": artifact,
            "scoring_status": scoring,
            "validity_status": validity_status,
            "result_written": not result_path.is_file(),
            "reason": reason["category"],
        }
        changes.append(entry)
        if not apply:
            continue
        usage.update({
            "artifact_status": artifact,
            "scoring_status": scoring,
            "validity_status": validity_status,
            "status_backfill": {"version": 1, "at": stamp, "reason": reason["category"]},
        })
        _write_json(usage_path, usage, root, backup_root)
        if not result_path.is_file():
            record = {
                "case": manifest.get("case"),
                "condition": manifest.get("condition"),
                "model": manifest.get("model"),
                "run_status": usage.get("status"),
                "artifact_status": artifact,
                "scoring_status": scoring,
                "final_score": None,
                "overall_score": None,
                "combined_score": None,
                "error_info": reason,
                "status_backfill": {"version": 1, "at": stamp},
            }
            _write_json(result_path, record, root, backup_root)
            counts["result_written"] += 1
    if apply and backup_root is not None and not changes:
        # Avoid leaving an empty directory on an idempotent second invocation.
        try:
            backup_root.rmdir()
            backup_root.parent.rmdir()
        except OSError:
            pass
    return {"counts": counts, "changes": changes,
            "backup_root": str(backup_root) if backup_root and changes else None}


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Backfill historical run/scoring statuses")
    parser.add_argument("root", type=Path)
    parser.add_argument("--apply", action="store_true", help="write changes; default is dry-run")
    args = parser.parse_args(argv)
    report = migrate(args.root.resolve(), apply=args.apply)
    print(json.dumps(report, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
