import json
from pathlib import Path

from harness.analysis.attribution import (
    calibration_report,
    infer_failure_label,
    make_review_sample,
    write_auto_label,
)
from harness.analysis.metrics import load_run


def _run(tmp_path: Path, *, status="ok", validity=0.0, error_info=None,
         failure_reason=None):
    run = tmp_path / "case__opencode__model__Interact__run1"
    run.mkdir()
    (run / "manifest.json").write_text(json.dumps({
        "run_id": run.name, "case": "x_clean", "condition": "Interact",
        "model": "chat/test", "scaffold": "opencode", "run_index": 1,
    }))
    usage = {"status": status, "produced": [], "elapsed_s": 2}
    if failure_reason:
        usage["failure_reason"] = failure_reason
    (run / "usage.json").write_text(json.dumps(usage))
    (run / "result.json").write_text(json.dumps({
        "validity_score": validity, "combined_score": 0.0,
        "error_info": error_info or {},
    }))
    return run


def test_auto_artifact_label_and_metrics_fallback(tmp_path):
    run = _run(tmp_path, error_info={"schema": ["missing solution.json"]})
    label = infer_failure_label(run)
    assert label["failure_stage"] == "artifact"
    assert label["label_source"] == "auto_rule_v1"
    assert write_auto_label(run) == run / "failure_stage.auto.json"
    assert load_run(run)["failure_stage"] == "artifact"


def test_auto_degraded_is_operational_not_agent_failure(tmp_path):
    run = _run(tmp_path, status="degraded", validity=None,
               failure_reason="tool_schema_validation_loop")
    label = infer_failure_label(run)
    assert label["failure_scope"] == "infrastructure"
    assert label["failure_stage"] == "execution"
    assert label["failure_subtype"] == "tool_schema_validation_loop"
    assert load_run(run)["failure_scope"] == "infrastructure"


def test_review_sample_and_calibration(tmp_path):
    run = _run(tmp_path, error_info={"hard_violations": ["overlap"]})
    sample = make_review_sample([tmp_path], fraction=1.0)
    assert len(sample) == 1
    sample[0]["reviewed_failure_scope"] = "agent"
    sample[0]["reviewed_failure_stage"] = "evaluator"
    sample[0]["reviewed_failure_subtype"] = "evaluator_hard_violation"
    report = calibration_report(sample)
    assert report["n_reviewed"] == 1
    assert report["agreement"]["failure_stage"] == 1.0
