"""离线 case-level 指标的确定性单元测试。"""

import json

import pytest

from harness.analysis.metrics import (analyze, failure_attribution,
                                      grouped_failure_attribution, holm_adjust,
                                      pass_metrics)
from harness.results import derive_run_statuses, update_run_statuses


def _run(case, condition, run_index, score, *, quality=None, validity=1.0):
    return {
        "case": case, "condition": condition, "scaffold": "opencode",
        "model": "chat/test", "status": "ok", "run_index": run_index,
        "score": score, "quality": score if quality is None else quality,
        "validity": validity,
    }


def test_paired_effect_averages_repeats_before_cases():
    rows = []
    for case, hidden, interact, full in (("a", 1, 3, 5), ("b", 2, 4, 8)):
        for i in (1, 2, 3):
            rows += [_run(case, "Hidden", i, hidden),
                     _run(case, "Interact", i, interact),
                     _run(case, "Full", i, full)]
    out = analyze(rows, pairs=[("Interact", "Hidden")], n_resamples=100, seed=7)
    effect = out["effects"][0]
    assert effect["n_pairs"] == 2
    assert effect["mean_difference"] == 2.0
    three = out["three_condition_effects"][0]
    assert three["input_gap"]["mean"] == 3.0
    assert three["g_info"]["mean"] == 2.0
    assert round(three["rho_info"]["mean"], 6) == round((2 / 4 + 2 / 6) / 2, 6)


def test_holm_preserves_order_and_missing_values():
    adjusted = holm_adjust([0.01, None, 0.04, 0.2])
    assert adjusted[1] is None
    assert adjusted[0] <= adjusted[2] <= adjusted[3]
    assert all(0 <= p <= 1 for p in adjusted if p is not None)


def test_pass_at_k_and_pass_power_are_case_macro_averages():
    rows = []
    # case a: 2/3 pass; case b: 3/3 pass. Macro SuccessRate=(2/3+1)/2.
    for i, passed in enumerate((True, True, False), 1):
        rows.append(_run("a", "Interact", i, 1, quality=1 if passed else 0))
    for i in (1, 2, 3):
        rows.append(_run("b", "Interact", i, 1, quality=1))
    metrics = pass_metrics(rows, {"a": 0.5, "b": 0.5}, 3)
    assert len(metrics) == 1
    result = metrics[0]
    assert result["n_cases_with_k"] == 2
    assert result["success_rate"] == pytest.approx(5 / 6)
    # pass@3 is 1 for a (at least one success) and 1 for b.
    assert result["pass_at_k"] == 1.0
    # pass^3 is 0 for a and 1 for b.
    assert result["pass_power_k"] == 0.5


def test_pass_power_k_uses_combinations_when_more_than_k_runs_exist():
    rows = [_run("a", "Interact", i, 1 if passed else 0,
                 quality=1 if passed else 0, validity=1 if passed else 0)
            for i, passed in enumerate((True, True, False), 1)]
    one = pass_metrics(rows, None, 1)[0]["cases"][0]
    two = pass_metrics(rows, None, 2)[0]["cases"][0]
    assert one["pass_power_k"] == pytest.approx(2 / 3)
    assert two["pass_power_k"] == pytest.approx(1 / 3)


def test_feasibility_is_default_pass_predicate_without_thresholds():
    rows = [_run("a", "Interact", 1, 0.0, quality=0.0, validity=0)]
    out = analyze(rows, ks=(1,), n_resamples=10)
    assert out["pass_predicate"] == "validity_score > 0"
    assert out["success_rate"][0]["success_rate"] == 0.0


def test_feasibility_pass_does_not_require_continuous_score():
    rows = [{**_run("a", "Interact", 1, 0.0, validity=1), "score": None}]
    out = analyze(rows, ks=(1,), n_resamples=10)
    assert out["success_rate"][0]["success_rate"] == 1.0


def test_failure_attribution_uses_validity_and_full_reference():
    rows = [
        _run("a", "Full", 1, 0.8, quality=0.8, validity=1),
        _run("b", "Full", 1, 0.9, quality=0.9, validity=1),
        _run("a", "Interact", 1, 0.2, quality=0.2, validity=0),
        _run("b", "Interact", 1, 0.4, quality=0.4, validity=0),
    ]
    rows[2]["failure_stage"] = "data"
    rows[3]["failure_stage"] = "modeling"
    result = failure_attribution(rows)
    assert result["n_failed"] == 2
    assert result["n_paired_for_loss"] == 2
    rates = {x["stage"]: x["rate"] for x in result["ffr"]}
    assert rates["data"] == 0.5 and rates["modeling"] == 0.5
    sal = {x["stage"]: x["mean_loss"] for x in result["sal"]}
    assert sal["data"] == pytest.approx(0.6) and sal["modeling"] == pytest.approx(0.5)
    shares = {x["stage"]: x["loss_share"] for x in result["loss_share"]}
    assert shares["data"] == pytest.approx(0.6 / 1.1)


def test_failure_loss_pairs_full_by_run_index_not_repeat_average():
    rows = [
        _run("a", "Full", 1, 0.9, validity=1),
        _run("a", "Full", 2, 0.3, validity=1),
        _run("a", "Interact", 2, 0.1, validity=0),
    ]
    rows[-1].update({"failure_scope": "agent", "failure_stage": "execution"})
    result = failure_attribution(rows)
    assert result["n_paired_for_loss"] == 1
    assert result["n_unpaired_for_loss"] == 0
    assert result["paired_rows"][0]["reference_score"] == pytest.approx(0.3)
    assert result["paired_rows"][0]["loss"] == pytest.approx(0.2)


def test_missing_run_index_is_unpaired_for_loss():
    rows = [
        _run("a", "Full", 1, 0.9, validity=1),
        {**_run("a", "Interact", 1, 0.1, validity=0), "run_index": None,
         "failure_scope": "agent", "failure_stage": "execution"},
    ]
    result = failure_attribution(rows)
    assert result["n_paired_for_loss"] == 0
    assert result["n_unpaired_for_loss"] == 1


def test_four_run_statuses_are_orthogonal_and_persisted(tmp_path):
    run = tmp_path / "run"
    run.mkdir()
    (run / "usage.json").write_text(json.dumps({
        "status": "timeout", "produced": ["solver.py"],
    }))
    (run / "result.json").write_text(json.dumps({
        "combined_score": 0.0, "validity_score": 0,
    }))
    got = derive_run_statuses(run)
    assert got == {
        "run_status": "timeout", "artifact_status": "complete",
        "scoring_status": "scored", "validity_status": "invalid",
    }
    assert update_run_statuses(run) == got
    saved = json.loads((run / "usage.json").read_text())
    assert {k: saved[k] for k in got} == got


def test_scoring_error_is_not_hidden_by_placeholder_zero(tmp_path):
    run = tmp_path / "run"
    run.mkdir()
    (run / "usage.json").write_text(json.dumps({
        "status": "ok", "produced": ["solver.py"],
    }))
    (run / "result.json").write_text(json.dumps({
        "combined_score": 0.0, "error": "extract failed",
    }))
    assert derive_run_statuses(run)["scoring_status"] == "failed"


def test_failure_attribution_excludes_non_agent_scopes_from_agent_ffr():
    rows = [
        _run("a", "Full", 1, 0.9, validity=1),
        _run("a", "Interact", 1, 0.2, validity=0),
        _run("b", "Interact", 1, 0.0, validity=0),
    ]
    rows[1].update({"failure_scope": "agent", "failure_stage": "modeling",
                    "failure_subtype": "hard_constraint_omitted"})
    rows[2].update({"failure_scope": "benchmark", "failure_stage": "evaluator",
                    "failure_subtype": "case_data_conflict"})

    result = failure_attribution(rows)
    assert result["n_candidate_failures"] == 2
    assert result["n_failed"] == 1
    assert result["n_non_agent_failures"] == 1
    assert result["candidate_failures_by_scope"]["benchmark"] == 1
    assert {x["stage"]: x["rate"] for x in result["ffr"]}["modeling"] == 1.0


def test_invalid_failure_scope_is_unknown_not_agent_failure():
    rows = [_run("a", "Interact", 1, 0.0, validity=0)]
    rows[0]["failure_scope"] = "typo"
    result = failure_attribution(rows)
    assert result["n_candidate_failures"] == 1
    assert result["n_failed"] == 0
    assert result["candidate_failures_by_scope"]["unknown"] == 1


def test_operational_attribution_is_separate_from_feasibility_failures():
    rows = [
        _run("a", "Hidden", 1, 0.0, validity=0),
        {**_run("b", "Hidden", 1, 0.0, validity=0), "status": "degraded",
         "failure_scope": "infrastructure", "killed_reason": "provider_unavailable"},
        {**_run("c", "Hidden", 1, 0.0, validity=0), "status": "error",
         "failure_scope": "interaction"},
    ]
    out = analyze(rows, ks=(1,), n_resamples=10)
    operational = out["operational_attribution"]
    assert operational["n_operational_failures"] == 2
    assert operational["by_status"] == {
        "degraded": 1, "error": 1, "timeout_unscored": 0,
        "incomplete": 0, "unscored": 0,
    }
    assert operational["by_scope"]["infrastructure"] == 1
    assert operational["by_scope"]["interaction"] == 1
    assert out["failure_attribution"]["n_failed"] == 1


def test_unscored_timeout_and_incomplete_are_operational_failures():
    rows = [
        {**_run("a", "Hidden", 1, 0), "status": "timeout", "score": None,
         "validity": None},
        {**_run("b", "Hidden", 1, 0), "status": None, "score": None,
         "validity": None},
        {**_run("c", "Hidden", 1, 0), "status": "ok", "score": None,
         "validity": None},
        {**_run("d", "Hidden", 1, 0), "status": "timeout", "score": 0.2,
         "validity": 1},
    ]
    out = analyze(rows, ks=(1,), n_resamples=10)["operational_attribution"]
    assert out["n_operational_failures"] == 3
    assert out["by_status"]["timeout_unscored"] == 1
    assert out["by_status"]["incomplete"] == 1
    assert out["by_status"]["unscored"] == 1
    assert analyze(rows, ks=(1,), n_resamples=10)["n_excluded"] == 3


def test_failure_attribution_is_grouped_by_system_and_condition():
    rows = [
        _run("a", "Full", 1, 0.9, validity=0),
        _run("a", "Interact", 1, 0.2, validity=0),
        _run("a", "Hidden", 1, 0.4, validity=0),
    ]
    rows[0]["failure_stage"] = "evaluator"
    rows[1]["failure_stage"] = "clarification"
    rows[2]["failure_stage"] = "data"
    by_condition = {x["condition"]: x for x in grouped_failure_attribution(rows)}
    assert by_condition["Interact"]["n_failed"] == 1
    assert by_condition["Interact"]["paired_rows"][0]["loss"] == pytest.approx(0.7)
    assert {x["stage"]: x["rate"] for x in by_condition["Interact"]["ffr"]}[
        "clarification"] == 1.0
    assert {x["stage"]: x["rate"] for x in by_condition["Hidden"]["ffr"]}[
        "data"] == 1.0
