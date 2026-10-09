import json
from pathlib import Path

from harness.analysis.metrics import load_runs
from harness.results import derive_run_statuses, discover_run_dirs
from harness.run import _hash_files


def _run(path: Path, run_id: str, started_at: str, score: float) -> Path:
    path.mkdir(parents=True)
    (path / "manifest.json").write_text(json.dumps({
        "run_id": run_id, "case": "case_clean", "condition": "Hidden",
        "model": "m", "scaffold": "opencode", "started_at": started_at,
    }))
    (path / "usage.json").write_text(json.dumps({"status": "ok"}))
    (path / "result.json").write_text(json.dumps({
        "combined_score": score, "validity_score": 1, "quality_score": score,
    }))
    return path


def test_discovers_batch_layout_and_keeps_latest_retry(tmp_path):
    old = _run(tmp_path / "old" / "results" / "same", "rid", "2026-01-01T00:00:00Z", 0.1)
    new = _run(tmp_path / "new" / "results" / "same", "rid", "2026-01-02T00:00:00Z", 0.9)
    unique = _run(tmp_path / "direct", "unique", "2026-01-01T00:00:00Z", 0.5)

    assert discover_run_dirs([tmp_path]) == sorted([new, unique])
    rows = load_runs([tmp_path])
    assert len(rows) == 2
    assert {r["run_id"]: r["score"] for r in rows} == {"rid": 0.9, "unique": 0.5}
    assert old not in discover_run_dirs([tmp_path])
    assert discover_run_dirs([tmp_path / "new"]) == [new]


def test_file_hash_is_independent_of_stage_root(tmp_path):
    roots = [tmp_path / "a", tmp_path / "b"]
    for root in roots:
        (root / "opencode").mkdir(parents=True)
        (root / "config.toml").write_text("x=1\n")
        (root / "opencode" / "opencode.jsonc").write_text("{}\n")

    def files(root):
        return [root / "config.toml", root / "opencode" / "opencode.jsonc"]

    assert _hash_files(files(roots[0]), base=roots[0]) == _hash_files(files(roots[1]), base=roots[1])
    (roots[1] / "config.toml").write_text("x=2\n")
    assert _hash_files(files(roots[0]), base=roots[0]) != _hash_files(files(roots[1]), base=roots[1])


def test_explicit_terminal_statuses_are_preserved(tmp_path):
    run = tmp_path / "run"
    run.mkdir()
    (run / "usage.json").write_text(json.dumps({
        "status": "ok", "produced": [],
    }))
    (run / "result.json").write_text(json.dumps({
        "artifact_status": "none",
        "scoring_status": "not_applicable",
        "final_score": None,
        "error_info": {"category": "artifact_missing"},
    }))
    assert derive_run_statuses(run) == {
        "run_status": "ok",
        "artifact_status": "none",
        "scoring_status": "not_applicable",
        "validity_status": "unknown",
    }
