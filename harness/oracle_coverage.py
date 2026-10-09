"""Seeded, nested oracle subsets; private registries never enter solve pods.

The staging CLI runs on the host. Only answers and non-identifying provenance
are staged; point IDs/permutations live in the campaign plan outside pod mounts.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import re
from pathlib import Path

LEVELS = (0, 25, 50, 75, 100)
SEED = 12345
PAYLOAD_FILE = "oracle_answers.json"
_PATTERN = re.compile(r"oracle_cov_v1_p(0|25|50|75|100)_s(0|[1-9][0-9]*)\Z")


def condition_name(percent: int, seed: int = SEED) -> str:
    if percent not in LEVELS or seed < 0:
        raise ValueError("invalid oracle coverage level/seed")
    return f"oracle_cov_v1_p{percent}_s{seed}"


def parse_condition(condition: str) -> tuple[int, int] | None:
    match = _PATTERN.fullmatch(condition)
    return tuple(map(int, match.groups())) if match else None


def digest(value) -> str:
    raw = json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(raw.encode()).hexdigest()


def selection(case: Path, condition: str, run_index: int) -> tuple[dict, dict]:
    """Return (safe pod payload, private host audit) for one cell."""
    from .conditions import _load_blockers

    parsed = parse_condition(condition)
    if parsed is None or run_index not in (1, 2, 3):
        raise ValueError("expected oracle_cov_v1 condition and run_index in 1,2,3")
    percent, base_seed = parsed
    points = _load_blockers(case)
    if not points or any(not answer.strip() for _, answer in points):
        raise ValueError(f"missing/empty oracle answers in {case}/gt.json")
    # Hash-based random ordering is independent of Python's random version.
    # Neither the coverage level nor GT contents enter the repeat seed.
    selection_seed = digest(["oracle_cov_v1", base_seed, case.name, run_index])
    order = sorted(range(len(points)),
                   key=lambda i: (digest([selection_seed, points[i][0]]), i))
    k = (percent * len(points) + 99) // 100
    selected = sorted(order[:k])  # render in registry order, NOT permutation order
    ids = [points[i][0] for i in selected]
    answers = [points[i][1].strip() for i in selected]
    metadata = {
        "target_percent": percent, "base_seed": base_seed,
        "selection_seed": selection_seed, "n_points": len(points), "k": k,
        "actual_coverage": k / len(points),
        "gt_sha256": hashlib.sha256((case / "gt.json").read_bytes()).hexdigest(),
        "selected_ids_sha256": digest(ids), "answers_sha256": digest(answers),
        "selection_algorithm": "sha256-order-v1",
    }
    payload = {"schema_version": 1, "case": case.name, "condition": condition,
               "run_index": run_index, "metadata": metadata, "answers": answers}
    audit = {key: value for key, value in payload.items() if key != "answers"}
    audit.update(selected_ids=ids, permutation_ids=[points[i][0] for i in order],
                 payload_sha256=digest(payload))
    return payload, audit


def validate_payload(payload: dict, case: Path, condition: str, run_index: int) -> dict:
    """Fail closed on wrong-cell, partial, or corrupt thin-case inputs."""
    def require(ok):
        if not ok:
            raise ValueError("payload mismatch")

    try:
        require(set(payload) == {"schema_version", "case", "condition", "run_index",
                                 "metadata", "answers"})
        require(payload["schema_version"] == 1)
        require(payload["case"] == case.name and payload["condition"] == condition)
        require(payload["run_index"] == run_index and run_index in (1, 2, 3))
        percent, seed = parse_condition(condition)
        meta, answers = payload["metadata"], payload["answers"]
        require(set(meta) == {"target_percent", "base_seed", "selection_seed", "n_points",
                             "k", "actual_coverage", "gt_sha256", "selected_ids_sha256",
                             "answers_sha256", "selection_algorithm"})
        require(meta["target_percent"] == percent and meta["base_seed"] == seed)
        n = meta["n_points"]
        require(type(n) is int and n > 0)
        require(type(meta["k"]) is int and meta["k"] == (percent * n + 99) // 100)
        require(meta["actual_coverage"] == meta["k"] / n)
        require(meta["selection_seed"] == digest(["oracle_cov_v1", seed, case.name, run_index]))
        require(meta["selection_algorithm"] == "sha256-order-v1")
        require(isinstance(answers, list) and len(answers) == meta["k"])
        require(all(isinstance(a, str) and a.strip() for a in answers))
        require(meta["answers_sha256"] == digest(answers))
        for key in ("gt_sha256", "selected_ids_sha256"):
            require(re.fullmatch(r"[0-9a-f]{64}", meta[key]))
    except (KeyError, TypeError, ValueError) as exc:
        raise ValueError(f"invalid oracle payload for {case.name}/{condition}/try{run_index}") from exc
    return payload


def load_payload(case: Path, condition: str, run_index: int = 1) -> dict:
    path = case / PAYLOAD_FILE
    if path.is_file():
        payload = json.loads(path.read_text(encoding="utf-8"))
    else:
        payload, _ = selection(case, condition, run_index)
    return validate_payload(payload, case, condition, run_index)


def runtime_fingerprint(repo: Path) -> str:
    """Return the code fingerprint used to freeze an oracle campaign."""
    from .run import _code_fingerprint
    _code_fingerprint.cache_clear()
    return _code_fingerprint(repo)


def cell_key(case: str, condition: str, run_index: int) -> str:
    return f"{case}/{condition}/try{run_index}"


def check_plan(plan: dict, case: Path, condition: str, run_index: int,
               repo: Path) -> tuple[dict, dict]:
    from .run import _case_fingerprint
    payload, audit = selection(case, condition, run_index)
    _case_fingerprint.cache_clear()
    if (plan.get("runtime_fingerprint") != runtime_fingerprint(repo)
            or plan.get("case_fingerprints", {}).get(case.name) != _case_fingerprint(case)
            or plan.get("cells", {}).get(cell_key(case.name, condition, run_index)) != audit):
        raise ValueError("oracle campaign inputs/code changed or cell is not in the frozen plan")
    return payload, audit


def authorize_infra_repair(plan: dict, case: Path, condition: str,
                           run_index: int, run_dir: Path) -> None:
    """Explicit, one-use exception for an audited transport failure.

    The original frozen plan is never edited. A repair revision binds the old
    usage/manifest/response hashes and only permits a failed, artifact-free
    transport attempt, never a numerical zero or an unsuccessful solver.
    Called by an external campaign controller before archival.
    """
    repair = plan.get("repair_revision", {})
    source = Path(repair.get("source_plan", "/nonexistent-oracle-plan"))
    if (not source.is_file() or hashlib.sha256(source.read_bytes()).hexdigest()
            != repair.get("source_sha256")):
        raise ValueError("oracle repair requires an unchanged original frozen plan")
    original = json.loads(source.read_text())
    key = cell_key(case.name, condition, run_index)
    expected = (Path(plan["runs_root"]) / "codex-gpt-6-astra" / case.name /
                condition / f"try{run_index}")
    if run_dir.resolve() != expected.resolve():
        raise ValueError("oracle repair directory does not match the authorized cell")
    authorization = repair.get("solve_authorizations", {}).get(key)
    if not authorization or plan.get("cells", {}).get(key) != original.get("cells", {}).get(key):
        raise ValueError("cell is not authorized for oracle infrastructure repair")
    for field in ("base_seed", "scaffold", "model", "timeout_s", "image", "case_fingerprints"):
        if plan.get(field) != original.get(field):
            raise ValueError(f"oracle repair must preserve {field}")
    for name in ("manifest.json", "usage.json", "response.txt"):
        if hashlib.sha256((run_dir / name).read_bytes()).hexdigest() != authorization.get(name):
            raise ValueError("original attempt changed or repair was already consumed")
    usage = json.loads((run_dir / "usage.json").read_text())
    if usage.get("status") != "error" or usage.get("produced"):
        raise ValueError("oracle repair is restricted to artifact-free transport errors")
    if "stream disconnected before completion: response.failed" not in (run_dir / "response.txt").read_text():
        raise ValueError("oracle transport error evidence does not match")


def stage(case: Path, destination: Path, payload: dict) -> None:
    """Materialize ONLY approved answers, never registry IDs or full information."""
    if destination.resolve() == case.resolve() or destination.name != case.name:
        raise ValueError("oracle stage requires a separate thin directory with the real case name")
    if any((destination / name).exists() for name in ("gt.json", "tests", "information.md")):
        raise ValueError("oracle thin case contains forbidden knowledge")
    destination.mkdir(parents=True, exist_ok=True)
    (destination / PAYLOAD_FILE).write_text(
        json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--case", type=Path, required=True)
    parser.add_argument("--condition", required=True)
    parser.add_argument("--run-index", type=int, required=True)
    parser.add_argument("--plan", type=Path, required=True)
    parser.add_argument("--destination", type=Path)
    parser.add_argument("--repair-run", type=Path,
                        help="validate an explicitly authorized infrastructure repair")
    args = parser.parse_args(argv)
    plan = json.loads(args.plan.read_text(encoding="utf-8"))
    payload, _ = check_plan(plan, args.case, args.condition, args.run_index,
                            Path(__file__).resolve().parents[1])
    if args.repair_run:
        authorize_infra_repair(plan, args.case, args.condition, args.run_index, args.repair_run)
    if args.destination:
        stage(args.case, args.destination, payload)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
