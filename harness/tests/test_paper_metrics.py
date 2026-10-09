"""论文实验依赖的三个指标：ATC、白问检测、Sufficiency 分层。

三者的共性是**错了不会报错，只会让论文数字悄悄偏移**：
  - ATC 算错 → 「问得早不早」的结论反了
  - 白问漏检 → R 被空转压低，`F − R` 虚高（且虚高方向"有利于结论"）
  - Sufficiency 误剔 → 少算一批 case 而无人察觉

    python3 -m harness.tests.test_paper_metrics
"""

from __future__ import annotations

import json
import sys
import threading
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from harness.clarify.score import ClarifyScore, _judge_call, _judge_once  # noqa: E402
import harness.clarify.score as clarify_score                         # noqa: E402
from harness.scoring.report import (insufficient_cases,             # noqa: E402
                                    split_by_sufficiency, summarize,
                                    _agg_judge_stats)


# ── ATC ──────────────────────────────────────────────────────────────────────

def test_atc_first_turn_is_one():
    s = ClarifyScore(n_blockers=3)
    s.covered_at_turn = {"a": 1, "b": 1, "c": 1}
    assert s.atc == 1.0
    print("✓ 全在首轮问到 → ATC=1.0")


def test_atc_averages_only_covered():
    """没问出来的考点没有「第几轮」，不该被记成 max_rounds 拉高均值。

    否则「少问但问得准」会输给「问了一堆也没问到」——方向正好反了。
    覆盖率由 recall 承担，ATC 只回答「问到的那些，多早问到」。
    """
    s = ClarifyScore(n_blockers=5)          # 5 个考点，只覆盖 2 个
    s.covered_at_turn = {"a": 1, "b": 3}
    assert s.atc == 2.0, s.atc
    print("✓ 只对已覆盖考点求平均（未覆盖不计入）")


def test_atc_none_when_nothing_covered():
    s = ClarifyScore(n_blockers=3)
    assert s.atc is None
    assert s.to_dict()["atc"] is None
    print("✓ 无覆盖时 ATC=None（而非 0——0 会被误读成首轮就问到）")


def test_atc_in_output():
    s = ClarifyScore(n_blockers=2)
    s.covered_at_turn = {"a": 2}
    d = s.to_dict()
    assert d["atc"] == 2.0 and d["covered_at_turn"] == {"a": 2}
    print("✓ ATC 与逐考点轮次都写进输出")


# ── Sufficiency 分层 ──────────────────────────────────────────────────────────

def _run(case, cond, validity=None):
    return {"case": case, "condition": cond, "validity": validity,
            "scaffold": "opencode", "model": "m"}


def test_drops_case_failing_under_full_info():
    """F 条件都 validity=0 → 卡在求解能力，剔除。"""
    runs = [_run("hard", "F", 0.0), _run("hard", "R", 0.0),
            _run("easy", "F", 1.0), _run("easy", "R", 1.0)]
    assert insufficient_cases(runs) == {"hard"}
    keep, drop, bad = split_by_sufficiency(runs)
    assert {r["case"] for r in keep} == {"easy"}
    assert len(drop) == 2 and bad == {"hard"}
    print("✓ F 也做不出的 case 被剔除")


def test_drops_current_full_condition_name():
    assert insufficient_cases([_run("hard", "Full", 0.0)]) == {"hard"}


def test_unscored_run_does_not_lower_valid_rate():
    rows = [
        {**_run("x", "Hidden", 1), "status": "ok", "score": 0.8},
        {**_run("x", "Hidden", None), "status": "ok", "score": None},
    ]
    result = summarize(rows)["opencode/m/Hidden"]
    assert result["valid_rate"] == 1.0
    assert result["n_unscored"] == 1

    rows[1]["validity"] = 1
    result = summarize(rows)["opencode/m/Hidden"]
    assert result["valid_rate"] == 1.0


def test_degraded_protocol_run_does_not_pollute_quality_or_usage_metrics():
    """脚手架协议故障不能拉低模型分，也不能抬高工具/token/时长均值。"""
    rows = [
        {**_run("x", "Hidden", 1), "status": "ok", "score": 0.8,
         "tool_calls": 4, "tokens_total": 1000, "minutes": 2},
        {**_run("y", "Hidden", 0), "status": "degraded", "score": 0.0,
         "tool_calls": 200, "tokens_total": 999999, "minutes": 180},
    ]
    result = summarize(rows)["opencode/m/Hidden"]
    assert result["n_runs"] == 2 and result["n_excluded"] == 1
    assert result["valid_rate"] == 1.0 and result["avg_all"] == 0.8
    assert result["avg_tools"] == 4 and result["avg_tokens_total"] == 1000
    assert result["avg_minutes"] == 2


def test_keeps_case_if_any_F_run_succeeds():
    """多次 run 里只要有一次成功，就说明信息给全是做得出来的。"""
    runs = [_run("flaky", "F", 0.0), _run("flaky", "F", 1.0)]
    assert insufficient_cases(runs) == set()
    print("✓ 取 F 的最好一次，不因单次失败误剔")


def test_keeps_case_without_F_runs():
    """没跑过 F 就无从判断——不能悄悄少算一批 case。"""
    runs = [_run("nof", "R", 0.0), _run("nof", "CF", 0.0)]
    assert insufficient_cases(runs) == set()
    print("✓ 没跑过 F 的 case 保留（不做无依据的剔除）")


def test_ignores_missing_validity():
    """validity 为 None（未评分）不等于 0。"""
    runs = [_run("x", "F", None), _run("x", "R", None)]
    assert insufficient_cases(runs) == set()
    print("✓ 未评分不当作失败")


# ── judge 失败与零覆盖的区分 ────────────────────────────────────────────────

def test_judge_failure_yields_none_not_zero():
    """judge 全部调用失败时，指标应为 None 而非 0。

    实测：kimi-k3 作 judge 时三个考点判定全失败（judge_errors=3），
    recall 却报 0.0——基础设施故障被伪装成"agent 一个都没问到"。
    批量跑几千 run 时这类失败会被完全淹没在低分里。
    """
    s = ClarifyScore(n_blockers=3, n_questions=12)
    s.judge_errors = 3
    assert s.judge_failed is True
    assert s.recall is None and s.precision is None and s.ask_f1 is None
    assert s.to_dict()["judge_failed"] is True
    print("✓ judge 全败 → 指标 None")


def test_genuine_zero_coverage_stays_zero():
    """judge 正常但确实没问到 → recall=0.0，不能与失败混为一谈。"""
    s = ClarifyScore(n_blockers=3, n_questions=12)
    s.judge_errors = 0
    assert s.judge_failed is False
    assert s.recall == 0.0
    print("✓ 真·零覆盖仍为 0.0")


def test_partial_judge_failure_still_scores():
    """只失败一部分时仍出分——否则一次抖动就废掉整个 run 的澄清指标。"""
    s = ClarifyScore(n_blockers=3, n_questions=10)
    s.judge_errors = 1
    s.covered = ["a"]
    assert s.judge_failed is False
    assert s.recall is not None
    print("✓ 部分失败仍出分")


def test_judge_accepts_fenced_json_with_prose():
    """格式噪声不应把本来有效的单票判定变成失败。"""
    class Client:
        def complete(self, **kwargs):
            return '分析如下：\n```json\n{"attempted": true, "covered": false, "question_indices": []}\n```\n以上。'

    result = _judge_call(Client(), "k", "d", ["q"])
    assert result == {"attempted": True, "covered": False,
                      "question_indices": []}


def test_judge_retries_invalid_json_and_records_two_attempts():
    """JSONDecodeError/截断类协议错误应触发一次 JSON-only 重试。"""
    class Client:
        def __init__(self):
            self.calls = 0

        def complete(self, **kwargs):
            self.calls += 1
            if self.calls == 1:
                return '{"attempted": true, "covered":'
            return '{"attempted": true, "covered": true, "question_indices": [1]}'

    client = Client()
    result = _judge_call(client, "k", "d", ["q"])
    assert result["covered"] is True
    assert client.calls == 2


def test_judge_parse_failure_after_retry_exposes_attempt_count():
    class Client:
        def complete(self, **kwargs):
            return "not json"

    import pytest
    with pytest.raises(ValueError) as caught:
        _judge_call(Client(), "k", "d", ["q"])
    assert getattr(caught.value, "judge_attempts") == 2


def test_judge_array_payload_is_not_accepted():
    class Client:
        def complete(self, **kwargs):
            return "[]"

    import pytest
    with pytest.raises(ValueError, match="expected JSON object"):
        _judge_call(Client(), "k", "d", ["q"])


def test_judge_uses_five_distinct_models_once(monkeypatch):
    """生产 judge 每个抽取模型调用一次，而不是同一 glm 重复五次。"""
    class Client:
        def __init__(self, model):
            self.model = model

    models = [m for _, m in clarify_score.JUDGE_MODELS]
    clients = [Client(m) for m in models]
    calls = []

    def fake_call(client, kid, desc, questions, triggers=None):
        calls.append(client.model)
        return {"attempted": True, "covered": True, "question_indices": [1]}

    monkeypatch.setattr(clarify_score, "_judge_call", fake_call)
    result = _judge_once(None, "k", "d", ["q"], clients=clients)

    assert sorted(calls) == sorted(models)
    assert result["_votes"] == 5
    assert result["_agreement"] == 1.0


def test_judge_models_are_called_in_parallel(monkeypatch):
    """五个独立 judge client 应同时进入调用，而不是逐个等待。"""
    class Client:
        def __init__(self, model):
            self.model = model

    clients = [Client(m) for _, m in clarify_score.JUDGE_MODELS]
    barrier = threading.Barrier(len(clients))

    def fake_call(client, kid, desc, questions, triggers=None):
        # 串行实现会在第一个 client 处超时；并行实现五个线程可同时通过。
        barrier.wait(timeout=2)
        return {"attempted": True, "covered": True, "question_indices": [1]}

    monkeypatch.setattr(clarify_score, "_judge_call", fake_call)
    result = _judge_once(None, "k", "d", ["q"], clients=clients)
    assert result["_votes"] == len(clients)


def test_judge_vote_records_expose_model_disagreement(monkeypatch):
    """逐模型投票应可用于定位系统性偏差，而不影响多数派结果。"""
    class Client:
        def __init__(self, model):
            self.model = model

    clients = [Client(m) for _, m in clarify_score.JUDGE_MODELS]
    yes_models = {clients[0].model, clients[1].model, clients[2].model}

    def fake_call(client, kid, desc, questions, triggers=None):
        covered = client.model in yes_models
        return {"attempted": True, "covered": covered,
                "question_indices": [1] if covered else []}

    monkeypatch.setattr(clarify_score, "_judge_call", fake_call)
    result = _judge_once(None, "k", "d", ["q"], clients=clients)

    assert result["covered"] is True
    assert result["_agreement"] == 0.6
    assert len(result["_ballots"]) == 5
    assert sum(bool(v.get("covered_disagree")) for v in result["_ballots"]) == 2


def test_judge_vote_errors_keep_model_reason_and_attempts(monkeypatch):
    """单票技术失败必须可定位，但不改变其余票的多数派结果。"""
    class Client:
        def __init__(self, model):
            self.model = model

    clients = [Client(m) for _, m in clarify_score.JUDGE_MODELS]
    failed_model = clients[0].model

    def fake_call(client, kid, desc, questions, triggers=None):
        if client.model == failed_model:
            raise RuntimeError("HTTP 401 Authorization: Bearer sk-secret-token")
        return {"attempted": True, "covered": True, "question_indices": [1]}

    monkeypatch.setattr(clarify_score, "_judge_call", fake_call)
    result = _judge_once(None, "k", "d", ["q"], clients=clients)

    assert result["_votes"] == 4
    assert result["_agreement"] == 1.0
    failed = [v for v in result["_ballots"] if v.get("error")]
    assert len(failed) == 1
    assert failed[0]["model"] == failed_model
    assert failed[0]["error_type"] == "RuntimeError"
    assert failed[0]["attempts"] == 1
    assert "sk-secret-token" not in failed[0]["error_message"]
    assert "[REDACTED]" in failed[0]["error_message"]


def test_judge_non_object_json_is_recorded_as_error(monkeypatch):
    """非对象 JSON 也应进入逐模型错误审计，而不是在汇总阶段失去上下文。"""
    class Client:
        def __init__(self, model):
            self.model = model

    clients = [Client(m) for _, m in clarify_score.JUDGE_MODELS]
    failed_model = clients[0].model

    def fake_call(client, kid, desc, questions, triggers=None):
        if client.model == failed_model:
            # 模拟 _judge_call 已解析出 JSON 数组后进行类型校验。
            raise ValueError("judge returned list, expected JSON object")
        return {"attempted": True, "covered": False, "question_indices": []}

    monkeypatch.setattr(clarify_score, "_judge_call", fake_call)
    result = _judge_once(None, "k", "d", ["q"], clients=clients)

    failed = [v for v in result["_ballots"] if v.get("error")]
    assert len(failed) == 1
    assert failed[0]["model"] == failed_model
    assert failed[0]["error_type"] == "ValueError"
    assert "expected JSON object" in failed[0]["error_message"]
    assert failed[0]["attempts"] == 1


def test_optional_arbitrator_resolves_disagreement_without_changing_base_votes(monkeypatch):
    """仲裁只在显式传入时触发，并保留五票基线与仲裁决定。"""
    class Client:
        def __init__(self, model):
            self.model = model
            self.calls = 0

        def complete(self, **kwargs):
            self.calls += 1
            return '{"attempted": true, "covered": false, "question_indices": []}'

    clients = [Client(m) for _, m in clarify_score.JUDGE_MODELS]
    arb = Client("responses/gpt-5.6-sol")
    yes_models = {clients[0].model, clients[1].model, clients[2].model}

    def fake_call(client, kid, desc, questions, triggers=None):
        return {"attempted": True, "covered": client.model in yes_models,
                "question_indices": [1] if client.model in yes_models else []}

    monkeypatch.setattr(clarify_score, "_judge_call", fake_call)
    result = _judge_once(None, "k", "d", ["q"], clients=clients,
                         arbitrator=arb, arbitration_model="responses/gpt-5.6-sol")

    assert result["covered"] is False  # arbiter's explicit decision wins
    assert result["_agreement"] == 0.6  # baseline vote remains auditable
    assert result["_arbitration"]["triggered"] is True
    assert result["_arbitration"]["used"] is True
    assert result["_arbitration"]["model"] == "responses/gpt-5.6-sol"
    assert arb.calls == 1


def test_optional_arbitrator_failure_falls_back_to_majority(monkeypatch):
    """仲裁服务失败时不污染原有多数派结果，并记录脱敏错误。"""
    class Client:
        def __init__(self, model):
            self.model = model

        def complete(self, **kwargs):
            raise RuntimeError("HTTP 401 Authorization: Bearer sk-arb-secret")

    clients = [Client(m) for _, m in clarify_score.JUDGE_MODELS]
    arb = Client("responses/gpt-5.6-sol")

    def fake_call(client, kid, desc, questions, triggers=None):
        return {"attempted": True, "covered": client is clients[0],
                "question_indices": [1] if client is clients[0] else []}

    monkeypatch.setattr(clarify_score, "_judge_call", fake_call)
    result = _judge_once(None, "k", "d", ["q"], clients=clients,
                         arbitrator=arb, arbitration_model="responses/gpt-5.6-sol")

    assert result["covered"] is False
    error = result["_arbitration"]["error"]
    assert error["error_type"] == "RuntimeError"
    assert "sk-arb-secret" not in error["error_message"]
    assert "[REDACTED]" in error["error_message"]


def test_score_persists_partial_judge_error_details(monkeypatch, tmp_path):
    """部分失败票必须同时进入顶层索引和考点级审计。"""
    case_dir = tmp_path / "case"
    run_dir = tmp_path / "run"
    case_dir.mkdir()
    run_dir.mkdir()
    monkeypatch.setattr(clarify_score, "load_blockers", lambda _: {"b1": "desc"})
    monkeypatch.setattr(clarify_score, "load_triggers", lambda _: {})
    monkeypatch.setattr(clarify_score, "load_tags", lambda _: {})
    monkeypatch.setattr(clarify_score, "load_questions", lambda _: [(1, "question?")])
    monkeypatch.setattr(clarify_score, "_task_advanced", lambda _: None)
    monkeypatch.setattr(clarify_score, "_judge_once", lambda *args, **kwargs: {
        "attempted": True,
        "covered": True,
        "question_indices": [1],
        "_agreement": 1.0,
        "_vote_errors": 1,
        "_ballots": [{
            "model": "judge-a",
            "error": True,
            "error_type": "TimeoutError",
            "error_message": "timed out",
            "attempts": 2,
        }],
    })

    result = clarify_score.score_clarification(
        case_dir, run_dir, client=object(), model="judge-a")
    persisted = json.loads((run_dir / "clarify_score.json").read_text())

    assert result == persisted
    assert result["judge_error_details"] == [{
        "blocker": "b1",
        "model": "judge-a",
        "error_type": "TimeoutError",
        "error_message": "timed out",
        "attempts": 2,
    }]
    assert result["audit"][0]["judge_ballots"][0]["model"] == "judge-a"
    assert result["judge_vote_stats"]["judge-a"]["errors"] == 1


def test_score_persists_all_failed_judge_details(monkeypatch, tmp_path):
    """考点全票失败时没有多数结果，但逐模型原因仍必须落盘。"""
    case_dir = tmp_path / "case"
    run_dir = tmp_path / "run"
    case_dir.mkdir()
    run_dir.mkdir()
    monkeypatch.setattr(clarify_score, "load_blockers", lambda _: {"b1": "desc"})
    monkeypatch.setattr(clarify_score, "load_triggers", lambda _: {})
    monkeypatch.setattr(clarify_score, "load_tags", lambda _: {})
    monkeypatch.setattr(clarify_score, "load_questions", lambda _: [(1, "question?")])
    monkeypatch.setattr(clarify_score, "_task_advanced", lambda _: None)

    def fail_all(*args, **kwargs):
        exc = ValueError("judge 全部失败")
        exc.judge_error_details = [{
            "model": "judge-a",
            "error": True,
            "error_type": "ConnectionError",
            "error_message": "upstream unavailable",
            "attempts": 1,
        }]
        raise exc

    monkeypatch.setattr(clarify_score, "_judge_once", fail_all)
    result = clarify_score.score_clarification(
        case_dir, run_dir, client=object(), model="judge-a")

    assert result["judge_failed"] is True
    assert result["judge_errors"] == 1
    assert result["audit"] == []
    assert result["judge_error_details"][0]["blocker"] == "b1"
    assert result["judge_error_details"][0]["model"] == "judge-a"
    assert result["judge_vote_stats"]["judge-a"]["errors"] == 1


def test_aggregate_judge_consistency_and_model_rates():
    rows = [{
        "judge_decisions": 5,
        "judge_failed_decisions": 1,
        "judge_agreement_mean": 0.8,
        "judge_unanimous_rate": 0.5,
        "judge_vote_stats": {
            "m1": {"votes": 5, "errors": 1, "covered_yes": 2,
                   "covered_disagree": 1, "attempted_yes": 3,
                   "attempted_disagree": 0},
        },
    }]
    got = _agg_judge_stats(rows)
    assert got["judge_total_decisions"] == 5
    assert got["judge_failed_decisions"] == 1
    assert got["judge_failure_rate"] == 0.2
    assert got["judge_successful_decisions"] == 4
    assert got["judge_agreement_mean"] == 0.8
    assert got["judge_disagreement_rate"] == 0.5
    assert got["judge_by_model"]["m1"]["error_rate"] == 0.2
    assert got["judge_by_model"]["m1"]["covered_disagreement_rate"] == 0.25


if __name__ == "__main__":
    for fn in [v for k, v in sorted(globals().items()) if k.startswith("test_")]:
        fn()
    print("\nall paper-metrics tests passed")
