import json

from harness.scripts.backfill_statuses import migrate


def _run(root, *, result=False):
    run = root / "model" / "001_demo_clean" / "Interact-Req" / "try1"
    run.mkdir(parents=True)
    (run / "manifest.json").write_text(json.dumps({
        "case": "001_demo_clean", "condition": "Interact-Req",
        "model": "model", "run_index": 1,
    }))
    (run / "usage.json").write_text(json.dumps({
        "status": "ok", "produced": [], "scoring_status": "unscored",
    }))
    (run / "clarify_score.json").write_text("{}")
    if result:
        (run / "result.json").write_text(json.dumps({
            "overall_score": 0.0,
        }))
    return run


def test_backfill_is_dry_run_then_atomic_and_idempotent(tmp_path):
    run = _run(tmp_path)
    dry = migrate(tmp_path)
    assert dry["counts"]["changed"] == 1
    assert not (run / "result.json").exists()

    applied = migrate(tmp_path, apply=True)
    assert applied["counts"]["result_written"] == 1
    result = json.loads((run / "result.json").read_text())
    assert result["final_score"] is None
    assert result["scoring_status"] == "not_applicable"
    assert (tmp_path / "_backfill_backups").is_dir()

    again = migrate(tmp_path)
    assert again["counts"]["changed"] == 0


def test_backfill_never_rewrites_numeric_zero(tmp_path):
    run = _run(tmp_path, result=True)
    applied = migrate(tmp_path, apply=True)
    assert applied["counts"]["changed"] == 1  # usage status is backfilled
    result = json.loads((run / "result.json").read_text())
    assert result["overall_score"] == 0.0
