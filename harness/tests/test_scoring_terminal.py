"""Terminal scoring records must distinguish agent outcomes from scorer failure."""

from __future__ import annotations

import json
import sys

import harness.scoring.__main__ as scoring_main


def _layout(tmp_path, *, produced, clarify=False):
    case = tmp_path / "001_demo_clean"
    (case / "tests").mkdir(parents=True)
    (case / "tests" / "evaluator.py").write_text("def evaluate(): pass\n")
    run = tmp_path / "results" / "try1"
    run.mkdir(parents=True)
    (run / "usage.json").write_text(json.dumps({
        "status": "ok", "produced": produced,
    }))
    (run / "manifest.json").write_text(json.dumps({
        "case": case.name, "condition": "Interact-Req", "model": "m",
    }))
    if clarify:
        (run / "clarify.json").write_text('{"rounds":[]}')
    return case, run


def _invoke(monkeypatch, case, run):
    monkeypatch.setattr(sys, "argv", [
        "harness.scoring", "--case", str(case), "--results", str(run.parent),
    ])
    return scoring_main.main()


def test_no_deliverable_writes_null_terminal_result(monkeypatch, tmp_path):
    case, run = _layout(tmp_path, produced=[])
    assert _invoke(monkeypatch, case, run) == 0
    result = json.loads((run / "result.json").read_text())
    marker = json.loads((run / "scoring_complete.json").read_text())
    assert result["scoring_status"] == "not_applicable"
    assert result["overall_score"] is None
    assert marker["status"] == "complete"


def test_clarify_failure_marks_job_incomplete(monkeypatch, tmp_path):
    case, run = _layout(tmp_path, produced=["solution.json"], clarify=True)

    def fake_score(case_dir, run_dir, timeout=None, **kwargs):
        result = {"overall_score": 0.5, "combined_score": 0.5}
        (run_dir / "result.json").write_text(json.dumps(result))
        return result

    import harness.scoring.solve as solve
    import harness.clarify.score as clarify_score
    monkeypatch.setattr(solve, "score_run", fake_score)
    monkeypatch.setattr(solve, "summarize", lambda result: "ok")
    monkeypatch.setattr(
        clarify_score, "score_clarification",
        lambda *args, **kwargs: (_ for _ in ()).throw(RuntimeError("judge down")))

    assert _invoke(monkeypatch, case, run) == 1
    marker = json.loads((run / "scoring_complete.json").read_text())
    assert marker["status"] == "failed"
    assert marker["solve_ok"] is True
    assert marker["clarify_ok"] is False


def test_failure_payload_from_scorer_is_not_mistaken_for_success(monkeypatch, tmp_path):
    case, run = _layout(tmp_path, produced=["solution.json"])

    def failed_score(case_dir, run_dir, timeout=None, **kwargs):
        result = {
            "overall_score": None,
            "scoring_status": "failed",
            "error_info": {"category": "extractor_failed"},
        }
        (run_dir / "result.json").write_text(json.dumps(result))
        return result

    import harness.scoring.solve as solve
    monkeypatch.setattr(solve, "score_run", failed_score)
    monkeypatch.setattr(solve, "summarize", lambda result: "failed")

    assert _invoke(monkeypatch, case, run) == 1
    marker = json.loads((run / "scoring_complete.json").read_text())
    assert marker["status"] == "failed"
    assert marker["solve_ok"] is False


def test_recorded_judge_errors_make_scoring_incomplete(monkeypatch, tmp_path):
    case, run = _layout(tmp_path, produced=["solution.json"], clarify=True)

    def fake_score(case_dir, run_dir, timeout=None, **kwargs):
        result = {"overall_score": 0.5, "combined_score": 0.5}
        (run_dir / "result.json").write_text(json.dumps(result))
        return result

    import harness.scoring.solve as solve
    import harness.clarify.score as clarify_score
    monkeypatch.setattr(solve, "score_run", fake_score)
    monkeypatch.setattr(solve, "summarize", lambda result: "ok")
    monkeypatch.setattr(clarify_score, "score_clarification", lambda *args, **kwargs: {
        "ask_f1": None, "recall": None, "precision": None,
        "n_questions": 1, "judge_errors": 1, "judge_failed": True,
    })

    assert _invoke(monkeypatch, case, run) == 1
    marker = json.loads((run / "scoring_complete.json").read_text())
    assert marker["status"] == "failed"
    assert marker["solve_ok"] is True
    assert marker["clarify_ok"] is False


def test_score_resume_reuses_complete_clarify_result(monkeypatch, tmp_path):
    case, run = _layout(tmp_path, produced=["solution.json"], clarify=True)
    (run / "result.json").write_text('{"overall_score":0.5}')
    (run / "clarify_score.json").write_text(json.dumps({
        "n_blockers": 2, "n_questions": 1, "judge_errors": 0,
        "judge_failed": False, "ask_f1": 0.5,
    }))

    import harness.clarify.score as clarify_score
    monkeypatch.setattr(
        clarify_score, "score_clarification",
        lambda *args, **kwargs: (_ for _ in ()).throw(
            AssertionError("complete cached judge result must be reused")))

    assert _invoke(monkeypatch, case, run) == 0
    marker = json.loads((run / "scoring_complete.json").read_text())
    assert marker["status"] == "complete"
    assert marker["clarify_ok"] is True


def test_only_prefers_direct_run_over_archived_retry(monkeypatch, tmp_path):
    """--only try1 must not recursively score _retries/.../try1."""
    case, run = _layout(tmp_path, produced=["solution.json"])
    archived = run.parent / "_retries" / "20260914-010203" / "try1"
    archived.mkdir(parents=True)
    (archived / "usage.json").write_text(json.dumps({
        "status": "ok", "produced": ["solution.json"],
    }))
    (archived / "manifest.json").write_text(json.dumps({
        "case": case.name, "condition": "Interact-Req", "model": "old",
    }))
    called = []

    def fake_score(case_dir, run_dir, timeout=None, **kwargs):
        called.append(run_dir)
        result = {"overall_score": 0.5, "combined_score": 0.5}
        (run_dir / "result.json").write_text(json.dumps(result))
        return result

    import harness.scoring.solve as solve
    monkeypatch.setattr(solve, "score_run", fake_score)
    monkeypatch.setattr(solve, "summarize", lambda result: "ok")
    monkeypatch.setattr(sys, "argv", [
        "harness.scoring", "--case", str(case),
        "--results", str(run.parent), "--only", "try1",
    ])

    assert scoring_main.main() == 0
    assert called == [run.resolve()]
    assert not (archived / "result.json").exists()
