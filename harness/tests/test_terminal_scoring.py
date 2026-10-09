import json

from harness.cli import _write_terminal_score_record, _completed_despite_nonzero_exit
from harness.run import RunSpec
from harness.scoring import solve


def test_missing_artifact_gets_null_terminal_record(tmp_path):
    case = tmp_path / "012_demo_clean"
    run = tmp_path / "run"
    case.mkdir()
    run.mkdir()
    spec = RunSpec(case=case, condition="Interact-Req", model="m")

    _write_terminal_score_record(run, spec, "ok", [], {})
    record = json.loads((run / "result.json").read_text())
    assert record["final_score"] is None
    assert record["scoring_status"] == "not_applicable"
    assert record["error_info"]["category"] == "artifact_missing"


def test_degraded_run_is_not_mislabeled_as_agent_missing_artifact(tmp_path):
    case = tmp_path / "012_demo_clean"
    run = tmp_path / "run"
    case.mkdir()
    run.mkdir()
    spec = RunSpec(case=case, condition="Interact-Req", model="m")

    _write_terminal_score_record(run, spec, "degraded", [], {})
    record = json.loads((run / "result.json").read_text())
    assert record["scoring_status"] == "not_applicable"
    assert record["error_info"]["category"] == "run_failed"


def test_scorer_exception_is_not_written_as_zero(tmp_path, monkeypatch):
    case = tmp_path / "012_demo_clean"
    run = tmp_path / "run"
    case.mkdir()
    run.mkdir()
    monkeypatch.setattr(
        "harness.scoring.triple.score_run",
        lambda *args, **kwargs: (_ for _ in ()).throw(RuntimeError("boom")),
    )

    result = solve._score_run_triple(case, run)
    assert result["overall_score"] is None
    assert result["scoring_status"] == "failed"
    assert json.loads((run / "result.json").read_text())["combined_score"] is None


def test_completed_despite_nonzero_exit_rescues_backend_anomaly_with_deliverable():
    """CLI 退出非零但产物齐全 → 改判可评分，避免 36 分钟的完成 run 被三重丢弃。"""
    # 有产物：从 error 抢救回来
    assert _completed_despite_nonzero_exit("error", ["solution.json"]) is True
    # 无产物：真失败，保持 error（下游 EXCLUDED_STATUS 照常剔除）
    assert _completed_despite_nonzero_exit("error", []) is False
    assert _completed_despite_nonzero_exit("error", None) is False
    # degraded 按定义本应无产物；若协议异常但已经写出产物，仍应送评。
    assert _completed_despite_nonzero_exit("timeout", ["solution.json"]) is False
    assert _completed_despite_nonzero_exit("degraded", ["solution.json"]) is True
    assert _completed_despite_nonzero_exit("degraded", []) is False
    assert _completed_despite_nonzero_exit("ok", ["solution.json"]) is False
