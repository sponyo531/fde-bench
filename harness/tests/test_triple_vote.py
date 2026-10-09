import json
import signal
import subprocess
import threading
from pathlib import Path

import pytest

from harness.scoring import triple
from harness.scoring.conservation_role import load_roles
from harness.scoring.triple import _vote_with_index


def test_vote_selects_representative_model():
    assert _vote_with_index([0.8, 0.8, 0.2]) == (0.8, "majority", 0)
    assert _vote_with_index([0.8, 0.8, 0.8, 0.2, 0.1]) == (0.8, "majority", 0)


def test_five_way_vote_requires_strict_majority():
    assert _vote_with_index([0.8, 0.8, 0.2, 0.2, 0.1]) == (None, "no_majority", None)
    assert _vote_with_index([0.8, 0.8, 0.8, 0.8, 0.8]) == (0.8, "unanimous", 0)


def test_vote_no_majority_has_no_representative():
    assert _vote_with_index([0.8, 0.7, 0.2]) == (None, "no_majority", None)


def test_description_schema_role_is_non_decision_metadata(tmp_path):
    tests = tmp_path / "tests"
    tests.mkdir()
    (tests / "submission_schema.json").write_text(json.dumps({
        "files": [{"name": "solution.json", "fields": [
            {"name": "x", "role": "decision"},
            {"name": "purpose", "role": "description"},
        ]}],
    }))

    roles, unknown = load_roles(tmp_path)

    assert roles == {"x": "decision", "purpose": "description"}
    assert unknown == []


def test_failed_extractor_is_none_vote(tmp_path, monkeypatch):
    """抽取器没有落 payload 是无效票，不能把 evaluator 的缺文件 0 当成得分。"""
    case = tmp_path / "case"
    run = tmp_path / "run"
    (run / "workspace").mkdir(parents=True)
    case.mkdir()

    def fake_extract(*args, **kwargs):
        return {"status": "error", "notes": "provider failed", "returncode": 1}

    def must_not_score(*args, **kwargs):
        raise AssertionError("无 normalized payload 时不应调用 evaluator")

    monkeypatch.setattr(triple, "_extract_once", fake_extract)
    monkeypatch.setattr(triple, "_score_once", must_not_score)
    got = triple.one_model(case, run, "qwen38_27b", "chat/qwen3.8-27b")

    assert got["overall_score"] is None
    assert got["vote_eligible"] is False
    assert len(got["attempts"]) == 2
    assert "extraction_failed" in got["error_info"]
    assert got["extraction_failure"]["category"] == "provider_error"
    audit = json.loads((run / "extractor_run_qwen38_27b.json").read_text())
    assert audit["overall_score"] is None


def test_failed_rescore_cannot_reuse_stale_normalized_payload(tmp_path, monkeypatch):
    """重评时，新抽取失败不能把上一轮 routes.json 当作本次成功。"""
    case = tmp_path / "case"
    run = tmp_path / "run"
    old_out = run / "normalized_qwen38_27b"
    (run / "workspace").mkdir(parents=True)
    old_out.mkdir()
    (old_out / "routes.json").write_text('{"routes": ["stale"]}')
    case.mkdir()

    monkeypatch.setattr(
        triple, "_extract_once",
        lambda *args, **kwargs: {"status": "error", "notes": "provider failed", "returncode": 1},
    )
    monkeypatch.setattr(
        triple, "_score_once",
        lambda *args, **kwargs: (_ for _ in ()).throw(
            AssertionError("旧 payload 不得触发 evaluator")),
    )

    got = triple.one_model(case, run, "qwen38_27b", "chat/qwen3.8-27b")

    assert got["overall_score"] is None
    assert got["payload_files"] == []
    assert got["extraction_failure"]["category"] == "provider_error"
    assert not (old_out / "routes.json").exists()


def test_payload_can_recover_vote_when_result_ledger_missing(tmp_path, monkeypatch):
    """routes 等 payload 已落盘但漏写 _result.json 时，确定性 evaluator 可恢复票。"""
    case = tmp_path / "case"
    run = tmp_path / "run"
    (run / "workspace").mkdir(parents=True)
    case.mkdir()

    def fake_extract(case_dir, workspace, out, model, timeout=None, audit_output=None):
        (out / "routes.json").write_text('{"routes": []}')
        return {"status": "error", "notes": "missing _result.json", "returncode": 0}

    monkeypatch.setattr(triple, "_extract_once", fake_extract)
    monkeypatch.setattr(triple, "_score_once", lambda *a, **k: (0.42, {}))
    got = triple.one_model(case, run, "qwen38_27b", "chat/qwen3.8-27b")

    assert got["overall_score"] == 0.42
    assert got["vote_eligible"] is True
    assert got["extraction_status"] == "error"
    assert got["payload_files"] == ["routes.json"]


def test_extractor_cannot_invent_empty_payload_without_agent_source(tmp_path, monkeypatch):
    case = tmp_path / "case"
    run = tmp_path / "run"
    case.mkdir()
    (run / "workspace").mkdir(parents=True)

    def invented(case_dir, workspace, out, model, timeout=None, audit_output=None):
        (out / "solution.json").write_text('{"days": []}')
        (out / "_result.json").write_text(json.dumps({
            "status": "success", "files_found": [],
            "files_normalized": ["solution.json"],
            "notes": "No agent output artifact found; empty schema supplied",
        }))
        return {"status": "success", "notes": "No agent output artifact found"}

    monkeypatch.setattr(triple, "_extract_once", invented)
    monkeypatch.setattr(
        triple, "_score_once",
        lambda *a, **k: (_ for _ in ()).throw(
            AssertionError("invented payload must not reach evaluator")),
    )
    got = triple.one_model(case, run, "gemini36f", "chat/gemini-3.6-flash")

    assert got["overall_score"] is None
    assert got["payload_files"] == []
    assert got["extraction_failure"]["category"] == "source_provenance_violation"


def test_extractor_cannot_rewrite_executable_agent_source(tmp_path, monkeypatch):
    case = tmp_path / "case"
    run = tmp_path / "run"
    case.mkdir()
    (run / "workspace").mkdir(parents=True)
    (run / "workspace" / "agent.py").write_text("print('agent')\n")

    def rewritten(case_dir, workspace, out, model, timeout=None, audit_output=None):
        (out / "solution.py").write_text("print('extractor rewrite')\n")
        (out / "_result.json").write_text(json.dumps({
            "status": "success", "files_found": ["agent.py"],
            "files_normalized": ["solution.py"],
        }))
        return {"status": "success", "notes": "rewrote source"}

    monkeypatch.setattr(triple, "_extract_once", rewritten)
    monkeypatch.setattr(
        triple, "_score_once",
        lambda *a, **k: (_ for _ in ()).throw(
            AssertionError("rewritten executable must not reach evaluator")),
    )
    got = triple.one_model(case, run, "glm52", "direct/glm-5.2")

    assert got["overall_score"] is None
    assert got["payload_files"] == []


def test_extractor_cannot_generate_predictions_from_agent_code(tmp_path, monkeypatch):
    case = tmp_path / "case"
    run = tmp_path / "run"
    case.mkdir()
    (run / "workspace").mkdir(parents=True)
    (run / "workspace" / "solution.py").write_text("print('train model')\n")

    def generated(case_dir, workspace, out, model, timeout=None,
                  audit_output=None):
        (out / "predictions.csv").write_text("id,prediction\n1,0.7\n")
        (out / "_result.json").write_text(json.dumps({
            "status": "success",
            "files_found": ["solution.py"],
            "files_normalized": ["predictions.csv"],
            "notes": "Extracted the saved model. Generated predictions.csv.",
        }))
        return {"status": "success", "notes": "generated predictions"}

    monkeypatch.setattr(triple, "_extract_once", generated)
    monkeypatch.setattr(
        triple, "_score_once",
        lambda *a, **k: (_ for _ in ()).throw(
            AssertionError("extractor-generated decisions must not reach evaluator")),
    )

    got = triple.one_model(case, run, "gemini36f", "chat/gemini-3.6-flash")

    assert got["overall_score"] is None
    assert got["payload_files"] == []
    assert got["extraction_failure"]["category"] == "source_provenance_violation"


RECOMPUTATION_NOTES = [
    "I ran the solver to produce the deliverable the agent intended: "
    "'python3 cpsat_solver.py --time 120 --workers 8'.",
    "Running the agent's finished solver script cpsat_solver.py produced solution.json.",
    "Did not modify inputs; I ran the solver to generate the missing plan.",
    "Did not modify inputs, but ran the solver to generate the missing plan.",
    "Executed the agent's saved model to create predictions.csv.",
]


@pytest.mark.parametrize("note", RECOMPUTATION_NOTES)
def test_solver_execution_admissions_are_rejected(tmp_path, monkeypatch, note):
    case = tmp_path / "case"
    run = tmp_path / "run"
    case.mkdir()
    (run / "workspace").mkdir(parents=True)
    (run / "workspace" / "cpsat_solver.py").write_text("print('solver')\n")

    def generated(case_dir, workspace, out, model, timeout=None, audit_output=None):
        metadata = {"status": "success", "files_found": ["cpsat_solver.py"],
                    "files_normalized": ["solution.json"], "notes": note}
        (out / "_result.json").write_text(json.dumps(metadata))
        (out / "solution.json").write_text('{"operations": []}')
        return {"status": "success", "notes": note}

    monkeypatch.setattr(triple, "_extract_once", generated)
    monkeypatch.setattr(triple, "check_run", lambda *a, **k: {})
    monkeypatch.setattr(triple, "_score_once", lambda *a, **k: pytest.fail(
        "newly solver-generated data must never reach evaluator"))
    got = triple.one_model(case, run, "glm52", "direct/glm-5.2")
    assert got["vote_eligible"] is False
    assert got["overall_score"] is None
    assert got["payload_files"] == []
    assert got["extraction_failure"]["category"] == "source_provenance_violation"
    assert len(got["attempts"]) == 2


@pytest.mark.parametrize("note", [
    "Copied solution.json; did not re-run the agent's solver (forbidden).",
    "Copied solution.json without running the agent's finished solver.",
    "Never executed the solver; copied existing output only.",
    "The agent ran the solver and wrote solution.json. I copied that file.",
    "Agent already generated predictions.csv. Copied existing output.",
    "Ran the evaluator on copied solution.json.",
])
def test_copy_evaluation_and_negated_execution_are_not_recomputation(note):
    assert triple._admits_recomputation(note) is False


@pytest.mark.parametrize("note", RECOMPUTATION_NOTES[:2])
@pytest.mark.parametrize("keep_payload", [True, False])
def test_cached_solver_execution_vote_cannot_survive_missing_payload(tmp_path, note, keep_payload):
    run = tmp_path / "run"
    out = run / "normalized_glm52"
    (run / "workspace").mkdir(parents=True)
    if keep_payload:
        out.mkdir()
        (out / "solution.json").write_text('{"operations": []}')
        (out / "_result.json").write_text(json.dumps({
            "files_found": ["cpsat_solver.py"], "notes": note}))
    record = {"tag": "glm52", "model": "direct/glm-5.2", "overall_score": 0.1396,
              "extraction_notes": note, "payload_files": ["solution.json"]}
    # Both persistence paths must be checked, not just the first cache file.
    for prefix in ("score", "extractor_run"):
        (run / f"{prefix}_glm52.json").write_text(json.dumps(record))
    assert triple._load_reusable_vote(run, "glm52", "direct/glm-5.2") is None


def test_clean_retry_is_not_rejected_for_previous_cleared_attempt(tmp_path):
    record = {"tag": "glm52", "extraction_notes": "Copied existing solution.json",
              "attempts": [
                  {"notes": "I ran the solver", "failure_category": "source_provenance_violation"},
                  {"notes": "Copied existing solution.json"}]}
    assert triple._vote_has_source_provenance(tmp_path, record) is True


def test_merge_rejects_tainted_peer_ledgers_without_rescoring(tmp_path, monkeypatch):
    case = tmp_path / "case"
    run = tmp_path / "run"
    case.mkdir(); run.mkdir()
    votes = [{"tag": tag, "model": model, "overall_score": 0.1396,
              "extraction_notes": "I ran the solver to produce solution.json",
              "payload_files": ["solution.json"]}
             for tag, model in triple._MODELS]
    monkeypatch.setattr(triple.solve, "evaluate", lambda *a, **k: pytest.fail(
        "contaminated cached votes must never reach evaluator"))
    monkeypatch.setattr(triple, "_scorer_provenance", lambda *a, **k: {
        "event": "test", "scored_at": "now", "scorer_commit": "x",
        "scorer_code_fingerprint": "y"})
    got = triple._merge_vote_result(case, run, votes)
    assert got["final_score"] is None
    # Scorer violations alone do not establish that the Agent has no artifact.
    assert got["scoring_status"] == "failed"
    assert all(value is None for value in got["scores"].values())


def test_negative_recomputation_note_does_not_reject_copied_payload(tmp_path):
    workspace = tmp_path / "workspace"
    out = tmp_path / "normalized"
    workspace.mkdir(); out.mkdir()
    (workspace / "predictions.csv").write_text("id,prediction\n1,0.7\n")
    (out / "predictions.csv").write_text("id,prediction\n1,0.7\n")
    (out / "_result.json").write_text(json.dumps({
        "status": "success",
        "files_found": ["predictions.csv"],
        "files_normalized": ["predictions.csv"],
        "notes": "Copied predictions.csv; did not generate or recompute predictions.",
    }))

    assert triple._payload_has_source_provenance(out, workspace) is True


def test_reusable_generated_prediction_vote_is_rejected(tmp_path):
    run = tmp_path / "run"
    out = run / "normalized_gemini36f"
    (run / "workspace").mkdir(parents=True)
    out.mkdir(parents=True)
    (run / "workspace" / "solution.py").write_text("print('model')\n")
    (out / "predictions.csv").write_text("id,prediction\n1,0.7\n")
    (out / "_result.json").write_text(json.dumps({
        "status": "success",
        "files_found": ["solution.py"],
        "files_normalized": ["predictions.csv"],
        "notes": "Generated predictions.csv by executing solution.py.",
    }))
    record = {
        "tag": "gemini36f", "model": "chat/gemini-3.6-flash",
        "overall_score": 0.7, "payload_files": ["predictions.csv"],
    }
    (run / "score_gemini36f.json").write_text(json.dumps(record))

    assert triple._load_reusable_vote(
        run, "gemini36f", "chat/gemini-3.6-flash") is None


def test_recover_orphaned_exact_python_payload(tmp_path, monkeypatch):
    case = tmp_path / "case"
    run = tmp_path / "run"
    case.mkdir()
    (run / "workspace").mkdir(parents=True)
    source = "print('agent')\n"
    (run / "workspace" / "stowage.py").write_text(source)
    out = run / "normalized_glm52"
    out.mkdir()
    (out / "solution.py").write_text(source)
    monkeypatch.setattr(triple.solve, "evaluate", lambda *a, **k: {
        "overall_score": 0.25, "validity_score": 1.0, "quality_score": 0.25,
    })

    got = triple._recover_orphaned_vote(
        case, run, "glm52", "direct/glm-5.2")

    assert got is not None
    assert got["overall_score"] == 0.25
    assert got["extraction_status"] == "recovered"
    assert json.loads((run / "score_glm52.json").read_text())["overall_score"] == 0.25


def test_reject_orphaned_rewritten_python_payload(tmp_path, monkeypatch):
    case = tmp_path / "case"
    run = tmp_path / "run"
    case.mkdir()
    (run / "workspace").mkdir(parents=True)
    (run / "workspace" / "stowage.py").write_text("print('agent')\n")
    out = run / "normalized_glm52"
    out.mkdir()
    (out / "solution.py").write_text("print('extractor rewrite')\n")
    monkeypatch.setattr(
        triple.solve, "evaluate",
        lambda *a, **k: (_ for _ in ()).throw(
            AssertionError("rewritten orphan must not reach evaluator")),
    )

    got = triple._recover_orphaned_vote(
        case, run, "glm52", "direct/glm-5.2")

    assert got is None


def test_majority_no_artifact_is_agent_terminal_not_scorer_failure(tmp_path, monkeypatch):
    case = tmp_path / "case"
    run = tmp_path / "run"
    case.mkdir(); run.mkdir()
    failures = [{
        "tag": f"m{i}", "model": f"model-{i}", "overall_score": None,
        "payload_files": [],
        "extraction_failure": {"category": "no_artifact"},
    } for i in range(3)]
    infra = [{
        "tag": f"e{i}", "model": f"error-{i}", "overall_score": None,
        "payload_files": [],
        "extraction_failure": {"category": "provider_error"},
    } for i in range(2)]
    monkeypatch.setattr(triple, "_scorer_provenance", lambda *a, **k: {
        "event": "test", "scored_at": "now", "scorer_commit": "x",
        "scorer_code_fingerprint": "y",
    })

    got = triple._merge_vote_result(case, run, failures + infra)

    assert got["scoring_status"] == "not_applicable"
    assert got["artifact_status"] == "none"
    assert got["error_info"]["category"] == "artifact_missing"


def test_missing_final_artifact_notes_are_not_misclassified_as_invalid_json():
    for note in (
        "No graded deliverable exists; no solution.json was ever produced.",
        "No schedule deliverable exists to normalize; the agent never persisted one.",
        "No predictions output found; predictions.csv was never generated.",
        "No _agent_summary.md exists. No submission.csv is present in the workspace.",
        "The solver never serialized a materialized schedule to solution.json.",
        "There is no materialized prediction deliverable in the workspace.",
    ):
        got = triple._extraction_failure(
            [{"status": "failed", "notes": note, "timed_out": False}], [])
        assert got["category"] == "no_artifact"


@pytest.mark.parametrize("note", [
    "No agent-produced solution.json. Workspace has only unfinished solvers. "
    "Did not re-run solver, copy /tmp or sibling extractor artifacts, "
    "or invent operations (forbidden). Evaluator: fatal 缺 solution.json.",
    "No agent-produced deliverable. Re-solving is forbidden.",
    "Nothing to normalize; network flow has 403 nodes. Solver execution is forbidden.",
])
def test_no_artifact_rule_explanations_are_not_provider_failures(note):
    got = triple._extraction_failure([{"status": "failed", "notes": note}], [])
    assert got["category"] == "no_artifact"


@pytest.mark.parametrize("note", [
    "HTTP 403 Forbidden", "HTTP status 401", "statusCode: 429", "403 Forbidden",
    "provider failed", "connection reset by peer", "network error",
    "invalid API key", "rate limit exceeded",
])
def test_real_transport_errors_remain_retryable(note):
    got = triple._extraction_failure([{"status": "error", "notes": note}], [])
    assert got["category"] == "provider_error"


def test_unrecognized_extractor_crash_is_not_agent_no_artifact():
    got = triple._extraction_failure([
        {"status": "error", "returncode": 1, "notes": "Unhandled exception"}], [])
    assert got["category"] == "extractor_error"


def test_case076_missing_artifact_majority_with_two_rejected_scores(tmp_path, monkeypatch):
    case = tmp_path / "case"
    run = tmp_path / "run"
    case.mkdir(); run.mkdir()
    notes = [
        "I ran the solver to generate the missing solution.json.",
        "No agent output exists.",
        "No agent deliverable found.",
        "Running the agent's finished solver produced solution.json.",
        "No agent-produced solution.json. Did not re-run solver (forbidden).",
    ]
    votes = []
    for i, ((tag, model), note) in enumerate(zip(triple._MODELS, notes)):
        attempt = {"status": "failed", "notes": note}
        if i in (0, 3):
            attempt["failure_category"] = "source_provenance_violation"
        votes.append({"tag": tag, "model": model, "overall_score": None,
                      "payload_files": [],
                      "extraction_failure": triple._extraction_failure([attempt], [])})
    monkeypatch.setattr(triple, "_scorer_provenance", lambda *a, **k: {
        "event": "test", "scored_at": "now", "scorer_commit": "x",
        "scorer_code_fingerprint": "y"})
    got = triple._merge_vote_result(case, run, votes)
    assert got["scoring_status"] == "not_applicable"
    assert got["error_info"]["category"] == "artifact_missing"
    assert got["error_info"]["message"].startswith("3/5")
    assert got["final_score"] is None


def test_two_scores_beat_one_failed_extractor():
    assert _vote_with_index([0.8, 0.8, None]) == (0.8, "majority", 0)


def test_score_run_starts_all_extractors_in_parallel(tmp_path, monkeypatch):
    """score_run must overlap all five paid extractor calls.

    A barrier makes a serial implementation fail deterministically: the first
    extractor cannot return until every extractor has entered the call.
    """
    case = tmp_path / "case"
    run = tmp_path / "run"
    case.mkdir()
    (run / "workspace").mkdir(parents=True)
    barrier = threading.Barrier(len(triple._MODELS), timeout=2)
    entered = []

    def fake_one_model(case_dir, run_dir, tag, model, timeout=None):
        entered.append(tag)
        barrier.wait()
        out = run_dir / f"normalized_{tag}"
        out.mkdir(exist_ok=True)
        (out / "routes.json").write_text("{}")
        return {
            "tag": tag,
            "model": model,
            "overall_score": 0.8,
            "files_normalized": ["routes.json"],
            "payload_files": ["routes.json"],
            "conservation": {"ok": True},
        }

    monkeypatch.setattr(triple, "one_model", fake_one_model)
    monkeypatch.setenv("DELIVER_EXTRACTOR_CONCURRENCY", str(len(triple._MODELS)))
    monkeypatch.setattr(triple.solve, "evaluate", lambda *a, **k: {
        "overall_score": 0.8,
        "validity_score": 1.0,
        "quality_score": 0.8,
    })

    result = triple.score_run(case, run, timeout=2)

    assert sorted(entered) == sorted(tag for tag, _ in triple._MODELS)
    assert result["vote"] == "unanimous"
    assert result["overall_score"] == 0.8


def test_score_run_reuses_two_votes_and_stops_after_third(tmp_path, monkeypatch):
    """An interrupted score resumes one missing vote and stops at majority."""
    case = tmp_path / "case"
    run = tmp_path / "run"
    case.mkdir()
    (run / "workspace").mkdir(parents=True)

    for tag, model in triple._MODELS[:2]:
        out = run / f"normalized_{tag}"
        out.mkdir()
        (out / "routes.json").write_text("{}")
        record = {
            "tag": tag, "model": model, "overall_score": 0.0,
            "files_normalized": ["routes.json"],
            "payload_files": ["routes.json"], "conservation": {"ok": True},
        }
        (run / f"extractor_run_{tag}.json").write_text(json.dumps(record))

    calls = []

    def fake_one_model(case_dir, run_dir, tag, model, timeout=None):
        calls.append((tag, model))
        assert tag == triple._MODELS[2][0]
        out = run_dir / f"normalized_{tag}"
        out.mkdir()
        (out / "routes.json").write_text("{}")
        return {
            "tag": tag, "model": model, "overall_score": 0.0,
            "files_normalized": ["routes.json"],
            "payload_files": ["routes.json"], "conservation": {"ok": True},
        }

    monkeypatch.setattr(triple, "one_model", fake_one_model)
    monkeypatch.setattr(triple.solve, "evaluate", lambda *a, **k: {
        "overall_score": 0.0, "validity_score": 0.0, "quality_score": 0.0,
    })

    result = triple.score_run(case, run)

    assert calls == [triple._MODELS[2]]
    assert result["vote"] == "majority"
    assert result["overall_score"] == 0.0
    for tag, _ in triple._MODELS[3:]:
        skipped = json.loads((run / f"score_{tag}.json").read_text())
        assert skipped["extraction_failure"]["category"] == "not_needed_after_majority"


def test_merge_uses_majority_peer_with_live_payload(tmp_path, monkeypatch):
    """A stale first ledger must not select a deleted normalized directory."""
    case = tmp_path / "case"
    run = tmp_path / "run"
    case.mkdir(); run.mkdir()
    runs = []
    for i, (tag, model) in enumerate(triple._MODELS):
        runs.append({
            "tag": tag, "model": model,
            "overall_score": 0.4 if i < 3 else None,
            "payload_files": ["routes.json"] if i < 3 else [],
            "conservation": {"ok": True} if i < 3 else None,
        })
    # The vote's first matching ledger is stale; only its second peer remains.
    live = run / f"normalized_{triple._MODELS[1][0]}"
    live.mkdir()
    (live / "routes.json").write_text("{}")
    evaluated = []

    def fake_evaluate(case_dir, normalized, timeout=None):
        evaluated.append(normalized)
        return {"overall_score": 0.4, "validity_score": 1.0,
                "quality_score": 0.4}

    monkeypatch.setattr(triple.solve, "evaluate", fake_evaluate)
    got = triple._merge_vote_result(case, run, runs)

    assert got["selected_tag"] == triple._MODELS[1][0]
    assert evaluated == [live]


def test_minimal_extractor_config_has_no_business_agent_or_plugins():
    from harness.installer import _load_jsonc
    from harness.scripts.extractor_opencode import EXTRACTOR_OPENCODE

    cfg = _load_jsonc((EXTRACTOR_OPENCODE / "opencode.jsonc").read_text())
    assert "default_agent" not in cfg
    assert "agent" not in cfg
    assert "plugin" not in cfg
    assert cfg["provider"] == {}
    assert "model" not in cfg


def test_heavy_code_case_prompt_forbids_nested_evaluator(tmp_path):
    from harness.scripts.extractor_opencode import build_prompt

    case = tmp_path / "025_vessel_stowage_planning_clean"
    evaluator = case / "tests" / "evaluator.py"
    workspace = tmp_path / "workspace"
    output = tmp_path / "output"
    data = case / "data"
    evaluator.parent.mkdir(parents=True)
    evaluator.write_text('"""schema"""\n')
    for path in (workspace, output, data):
        path.mkdir(parents=True)

    prompt = build_prompt(evaluator, workspace, output, data)

    assert "Do not run" in prompt
    assert "byte-for-byte" in prompt
    assert "Verify by running the evaluator (required)" not in prompt


def test_extractor_uses_only_requested_private_provider(tmp_path, monkeypatch):
    """The isolated extractor loads exactly one model from a private registry."""
    from harness.scripts.extractor_opencode import _prepare_config
    from harness.installer import _load_jsonc

    private = tmp_path / "private.jsonc"
    private.write_text(json.dumps({"provider": {
        "responses": {"npm": "@ai-sdk/openai", "models": {"gpt-5.2": {}}},
        "chat": {"npm": "@ai-sdk/openai-compatible", "models": {"qwen3.8-27b": {}}},
    }}))
    monkeypatch.setenv("FDE_EXTRACTOR_OPENCODE_CONFIG", str(private))
    output_dir = tmp_path / "normalized"
    output_dir.mkdir()
    config_dir, routed = _prepare_config(output_dir, "responses/gpt-5.2")
    cfg = _load_jsonc((config_dir / "opencode.jsonc").read_text())

    assert routed == "responses/gpt-5.2"
    assert cfg["model"] == routed
    assert set(cfg["provider"]) == {"responses"}
    assert cfg["provider"]["responses"]["models"] == {"gpt-5.2": {}}


def test_extractor_rejects_missing_private_config(tmp_path, monkeypatch):
    from harness.scripts.extractor_opencode import _prepare_config

    monkeypatch.setenv("FDE_EXTRACTOR_OPENCODE_CONFIG", str(tmp_path / "missing.jsonc"))
    with pytest.raises(RuntimeError, match="provider config missing"):
        _prepare_config(tmp_path / "workspace", "direct/glm-5.2")


def test_extractor_rejects_model_absent_from_private_config(tmp_path, monkeypatch):
    from harness.scripts.extractor_opencode import _prepare_config

    private = tmp_path / "private.jsonc"
    private.write_text(json.dumps({"provider": {"chat": {
        "npm": "@ai-sdk/openai-compatible", "models": {"qwen3.8-27b": {}}
    }}}))
    monkeypatch.setenv("FDE_EXTRACTOR_OPENCODE_CONFIG", str(private))
    with pytest.raises(RuntimeError, match="is absent"):
        _prepare_config(tmp_path / "workspace", "direct/glm-5.2")


def test_reextract_one_model_reuses_peer_scores_and_backs_up_old_files(tmp_path,
                                                                       monkeypatch):
    case = tmp_path / "case"
    run = tmp_path / "run"
    case.mkdir()
    (run / "workspace").mkdir(parents=True)
    old_normalized = run / "normalized_qwen38_27b"
    old_normalized.mkdir()
    (old_normalized / "stale.json").write_text("{}")

    def score(tag, model, value):
        return {"tag": tag, "model": model, "overall_score": value,
                "files_normalized": ["routes.json"], "conservation": {"ok": True}}

    glm = score("glm52", "direct/glm-5.2", 0.8)
    old_qwen = score("qwen38_27b", "old/qwen", 0.0)
    ds = score("dsv4f", "chat/deepseek-v4-flash", 0.8)
    gemini = score("gemini36f", "chat/gemini-3.6-flash", 0.8)
    grok = score("grok45", "chat/grok-4.5", 0.8)
    for tag, value in (("glm52", glm), ("qwen38_27b", old_qwen), ("dsv4f", ds),
                       ("gemini36f", gemini), ("grok45", grok)):
        (run / f"score_{tag}.json").write_text(json.dumps(value))
    glm_before = (run / "score_glm52.json").read_bytes()
    ds_before = (run / "score_dsv4f.json").read_bytes()
    (run / "extractor_run_qwen38_27b.json").write_text(json.dumps(old_qwen))
    (run / "result.json").write_text(json.dumps({"status": "ok", "overall_score": 0.8}))

    calls = []

    def fake_one_model(case_dir, run_dir, tag, model, timeout=None):
        calls.append((tag, model))
        assert not (run_dir / "normalized_qwen38_27b").exists()
        # 旧账本在新抽取完成前仍留在原位，Job 被杀也不会出现 score 缺口。
        assert json.loads((run_dir / "score_qwen38_27b.json").read_text())["overall_score"] == 0.0
        (run_dir / "normalized_qwen38_27b").mkdir()
        (run_dir / "normalized_qwen38_27b" / "routes.json").write_text("{}")
        return {**score(tag, model, 0.8), "extraction_status": "success",
                "payload_files": ["routes.json"]}

    monkeypatch.setattr(triple, "one_model", fake_one_model)
    monkeypatch.setattr(triple.solve, "evaluate", lambda *a, **k: {
        "overall_score": 0.8, "validity_score": 1, "quality_score": 0.8,
    })

    got = triple.reextract_model(case, run, "qwen38_27b")

    assert calls == [("qwen38_27b", "chat/qwen3.8-27b")]
    assert got["overall_score"] == 0.8
    assert got["vote"] == "unanimous"
    assert (run / "score_glm52.json").read_bytes() == glm_before
    assert (run / "score_dsv4f.json").read_bytes() == ds_before
    backup = Path(got["backup"])
    assert (backup / "normalized_qwen38_27b" / "stale.json").is_file()
    assert (backup / "score_qwen38_27b.json").is_file()
    assert (backup / "result.json").is_file()
    result = json.loads((run / "result.json").read_text())
    assert result["scores"] == {"glm52": 0.8, "qwen38_27b": 0.8, "dsv4f": 0.8,
                                 "gemini36f": 0.8, "grok45": 0.8}
    assert result["scorer_code_fingerprint"]
    assert result["scoring_history"][-1]["event"] == "reextract"
    assert result["scoring_history"][-1]["extractor_tag"] == "qwen38_27b"
    assert json.loads((run / "score_qwen38_27b.json").read_text())["scoring_provenance"]


def test_extractor_needs_rerun_ignores_opencode_config(tmp_path):
    out = tmp_path / "normalized_qwen38_27b" / ".opencode"
    out.mkdir(parents=True)
    (out / "opencode.jsonc").write_text("{}")
    assert triple.extractor_needs_rerun(tmp_path, "qwen38_27b") is True
    (tmp_path / "normalized_qwen38_27b" / "routes.json").write_text("{}")
    assert triple.extractor_needs_rerun(tmp_path, "qwen38_27b") is False


def test_extract_timeout_terminates_whole_process_group(tmp_path, monkeypatch):
    class FakeProcess:
        pid = 4321
        returncode = None

        def __init__(self):
            self.calls = 0

        def communicate(self, timeout=None):
            self.calls += 1
            if self.calls == 1:
                raise subprocess.TimeoutExpired(["extractor"], timeout)
            self.returncode = -signal.SIGTERM
            return "partial stdout", "partial stderr"

        def poll(self):
            return self.returncode

    fake = FakeProcess()
    popen_kwargs = {}
    killed = []

    def fake_popen(*args, **kwargs):
        popen_kwargs.update(kwargs)
        return fake

    monkeypatch.setattr(triple.subprocess, "Popen", fake_popen)
    monkeypatch.setattr(triple.os, "killpg", lambda pid, sig: killed.append((pid, sig)))
    case = tmp_path / "case"
    workspace = tmp_path / "workspace"
    out = tmp_path / "out"
    for path in (case / "tests", workspace, out):
        path.mkdir(parents=True)

    got = triple._extract_once(case, workspace, out, "chat/qwen3.8-27b", timeout=1)

    assert got["timed_out"] is True
    assert got["status_source"] == "timeout"
    assert killed == [(4321, signal.SIGTERM)]
    assert popen_kwargs["start_new_session"] is True
    # Parallel extractors must not contend for OpenCode's default SQLite DB.
    assert popen_kwargs["env"]["XDG_DATA_HOME"] == str(
        out.parent / ".extractor_runtime" / out.name / "data")
    assert popen_kwargs["env"]["XDG_CACHE_HOME"] == str(
        out.parent / ".extractor_runtime" / out.name / "cache")
    assert popen_kwargs["env"]["XDG_STATE_HOME"] == str(
        out.parent / ".extractor_runtime" / out.name / "state")
    assert got["stdout_tail"] == "partial stdout"
