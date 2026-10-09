"""token / 费用采集。

这条链路失败时**完全无声**：`_find_db` 返回 None → collect_usage 全 None →
cli.py 的 `if any(v is not None ...)` 不成立 → usage.json 里连 token_usage
字段都不出现。于是成本分析、cache 命中率、"澄清多花多少钱"全部拿不到，
而日志里一个字都没有。实测路径过期后一直如此。

    python3 -m harness.tests.test_usage
"""

from __future__ import annotations

import json
import sqlite3
import sys
import tempfile
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from harness.scoring.usage import _find_db, collect_usage      # noqa: E402
from harness.backends.usage import UsageLedger, extract_usage  # noqa: E402
from harness.cli import _sync_clarify_phase_stats              # noqa: E402


def _mk_db(path: Path, steps: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(str(path))
    conn.execute("CREATE TABLE part (data TEXT, time_created INTEGER)")
    for i, s in enumerate(steps):
        conn.execute("INSERT INTO part VALUES (?, ?)", (json.dumps({
            "type": "step-finish",
            "tokens": {"input": s["input"], "output": s["output"],
                       "reasoning": s.get("reasoning", 0),
                       "cache": {"read": s.get("cache_read", 0), "write": 0}},
        }), s.get("t", 1_000_000 + i * 1000)))
    conn.commit()
    conn.close()


STEPS = [{"input": 1000, "output": 500, "cache_read": 4000}]


def test_finds_db_in_agent_home():
    """权威位置：environment.py 建的私有 XDG。"""
    with tempfile.TemporaryDirectory() as td:
        run = Path(td)
        _mk_db(run / ".agent_home/data/opencode/opencode.db", STEPS)
        assert _find_db(run) is not None, "找不到 .agent_home 下的 DB"
    print("✓ 找得到 .agent_home/data/opencode/ 下的 DB")


def test_finds_db_in_legacy_layout():
    """旧布局仍要兼容，否则历史 run 重新聚合时费用全丢。"""
    with tempfile.TemporaryDirectory() as td:
        run = Path(td)
        _mk_db(run / "workspace/.opencode_data/opencode/opencode.db", STEPS)
        assert _find_db(run) is not None
    print("✓ 兼容旧布局 workspace/.opencode_data/")


def test_collects_tokens_and_cost():
    with tempfile.TemporaryDirectory() as td:
        run = Path(td)
        _mk_db(run / ".agent_home/data/opencode/opencode.db", STEPS)
        got = collect_usage(run, "chat/glm-5.2")
    assert got["tokens"]["input"] == 1000, got
    assert got["tokens"]["cache_read"] == 4000, got
    # 命中率分母是 input + cache_read（opencode 的 input 已扣除命中部分）
    assert got["cache_hit_rate"] == 0.8, got
    assert got["cost_usd"] is not None and got["cost_usd"] > 0, got
    assert got["pricing_model"] == "glm-5.2", got
    print("✓ token / cache 命中率 / 费用均采到")


def test_missing_db_returns_none_not_zero():
    """采不到必须是 None，不能是 0——否则"没花钱"和"没采到"混为一谈。"""
    with tempfile.TemporaryDirectory() as td:
        got = collect_usage(Path(td), "chat/glm-5.2")
    assert got["cost_usd"] is None and got["tokens"] is None, got
    assert got["llm_calls"] is None
    assert not any(got["availability"].values())
    print("✓ 采不到时返回 None 而非 0")


def test_backend_metadata_fallback_and_null_cache():
    """非 OpenCode metadata 进入统一 schema，缺 cache 不伪造命中率。"""
    with tempfile.TemporaryDirectory() as td:
        got = collect_usage(Path(td), "chat/glm-5.2", backend_usage={
            "source": "backend_metadata",
            "tokens": {"input": 100, "output": 50, "reasoning": None,
                       "cache_read": None, "cache_write": None, "steps": 1},
        })
    assert got["tokens"]["input"] == 100
    assert got["cache_hit_rate"] is None
    assert got["cost_usd"] is not None
    assert got["source"] == "backend_metadata"

    with tempfile.TemporaryDirectory() as td:
        got = collect_usage(Path(td), "chat/glm-5.2", backend_usage={
            "tokens": {"input": 100, "output": 50, "cache_read": 25},
        })
    # 与 opencode 分支同一定义：input 不含缓存，分母 = input + cache_read
    assert got["cache_hit_rate"] == 0.2


def test_ledger_normalizes_input_to_exclude_cache_read():
    """OpenAI/Gemini 协议的 prompt_tokens 含 cached；账本入账时减掉，口径与 Anthropic/opencode 对齐。"""
    openai_like = UsageLedger(input_includes_cache_read=True)
    openai_like.record({"prompt_tokens": 100, "completion_tokens": 5,
                        "prompt_tokens_details": {"cached_tokens": 40}})
    anthropic_like = UsageLedger(input_includes_cache_read=False)
    anthropic_like.record({"input_tokens": 60, "output_tokens": 5, "cache_read_input_tokens": 40})
    a, b = openai_like.snapshot()["tokens"], anthropic_like.snapshot()["tokens"]
    assert a["input"] == b["input"] == 60 and a["cache_read"] == b["cache_read"] == 40


def test_usage_ledger_accumulates_and_splits_phase():
    ledger = UsageLedger()
    ledger.set_phase("clarify")
    ledger.record({"prompt_tokens": 10, "completion_tokens": 4}, elapsed_s=1.2)
    ledger.set_phase("solve")
    ledger.record({"input_tokens": 20, "output_tokens": 8,
                   "prompt_tokens_details": {"cached_tokens": 5}}, elapsed_s=2.3)
    got = ledger.snapshot()
    assert got["tokens"]["input"] == 30
    assert got["tokens"]["output"] == 12
    assert got["tokens"]["cache_read"] == 5
    assert got["phase"]["clarify"]["input"] == 10
    assert got["phase"]["solve_secs"] == 2.3
    assert len(got["calls"]) == 2
    assert got["precision"] == "native_event"


def test_backend_phase_keeps_real_zero_and_exposes_contract():
    """无澄清是明确的 0，不是未采集到；LLM call 与 send 单位不可混用。"""
    with tempfile.TemporaryDirectory() as td:
        got = collect_usage(Path(td), "chat/glm-5.2", backend_usage={
            "tokens": {"input": 100, "output": 20, "reasoning": 5,
                       "cache_read": 0, "cache_write": 0, "steps": 2},
            "steps_unit": "llm_call",
            "phase": {
                "clarify": {"input": 0, "output": 0, "reasoning": 0,
                            "cache_read": 0, "cache_write": 0, "steps": 0},
                "solve": {"input": 100, "output": 20, "reasoning": 5,
                          "cache_read": 0, "cache_write": 0, "steps": 2},
                "clarify_secs": 0.0, "solve_secs": 12.0,
            },
        })
    assert got["phase"]["clarify_tokens"] == 0
    assert got["phase"]["solve_tokens"] == 125
    assert got["llm_calls"] == 2
    assert got["availability"]["phase_tokens"] is True
    assert got["units"]["llm_calls"] == "llm_call"


def test_send_granularity_never_masquerades_as_llm_calls():
    with tempfile.TemporaryDirectory() as td:
        got = collect_usage(Path(td), "chat/gemini-3.1-pro-preview", backend_usage={
            "tokens": {"input": 210_000, "output": 20, "reasoning": None,
                       "cache_read": 0, "cache_write": None, "steps": 1},
            "steps_unit": "send", "precision": "aggregate",
        })
    assert got["llm_calls"] is None
    assert got["availability"]["llm_calls"] is False
    assert got["precision"] == "aggregate"
    assert got["cost_precision"] == "aggregate_estimate"


def test_opencode_phase_tokens_include_output_and_reasoning_without_clarify():
    with tempfile.TemporaryDirectory() as td:
        run = Path(td)
        _mk_db(run / ".agent_home/data/opencode/opencode.db", [
            {"input": 100, "output": 20, "reasoning": 7, "t": 1_000_000},
            {"input": 200, "output": 30, "reasoning": 11, "t": 1_010_000},
        ])
        got = collect_usage(run, "chat/glm-5.2")
    assert got["phase"]["clarify_tokens"] == 0
    assert got["phase"]["solve_tokens"] == 368
    assert got["phase"]["solve_reasoning"] == 18
    assert got["source"] == "opencode_sqlite"
    assert got["precision"] == "native_event"
    assert got["llm_calls"] == 2


def test_opencode_phase_tokens_include_reasoning_on_both_sides():
    with tempfile.TemporaryDirectory() as td:
        run = Path(td)
        _mk_db(run / ".agent_home/data/opencode/opencode.db", [
            {"input": 100, "output": 10, "reasoning": 5, "t": 1_000_000},
            {"input": 900, "output": 90, "reasoning": 15, "t": 1_060_000},
        ])
        (run / "clarify.json").write_text(json.dumps({
            "total_rounds": 1,
            "rounds": [{"answered_at_ms": 1_030_000}],
        }), encoding="utf-8")
        got = collect_usage(run, "chat/glm-5.2")
    assert got["phase"]["clarify_tokens"] == 115
    assert got["phase"]["solve_tokens"] == 1005
    assert got["phase"]["clarify_reasoning"] == 5
    assert got["phase"]["solve_reasoning"] == 15


def test_extracts_nested_cli_usage():
    got = extract_usage({"type": "turn.completed", "response": {
        "usage": {"prompt_tokens": 3, "completion_tokens": 2}}})
    assert got == {"input": 3, "output": 2, "reasoning": None,
                   "cache_read": None, "cache_write": None}


def test_unknown_model_reports_unpriced_explicitly():
    """未知单价必须显式告警，不能把缺失费用伪装成 0。"""
    with tempfile.TemporaryDirectory() as td:
        run = Path(td)
        _mk_db(run / ".agent_home/data/opencode/opencode.db", STEPS)
        got = collect_usage(run, "chat/model-without-price")
    assert got["cost_usd"] is None
    assert got["pricing_model"] is None
    assert "no pricing entry" in got["pricing_warning"]
    print("✓ 未知模型单价显式告警")


def test_native_clarify_stats_use_event_timeline():
    """native send 的全会话墙钟不能冒充澄清耗时。"""
    with tempfile.TemporaryDirectory() as td:
        run = Path(td)
        path = run / "clarify.json"
        path.write_text(json.dumps({
            "total_rounds": 1,
            "total_questions": 2,
            "total_tokens": 0,
            "clarify_secs": 3600.0,
            "rounds": [],
        }), encoding="utf-8")
        got = _sync_clarify_phase_stats(run, {"phase": {
            "clarify_secs": 68.3,
            "clarify_tokens": 1234,
            "solve_secs": 3500.0,
        }})
        on_disk = json.loads(path.read_text(encoding="utf-8"))
    assert got is not None
    assert on_disk["clarify_secs"] == 68.3
    assert on_disk["total_tokens"] == 1234
    assert on_disk["phase_stats_source"] == "opencode_sqlite"          # 按真实来源命名


def test_clarify_stats_unchanged_without_phase_timeline():
    """其他脚手架没有 phase 切分时不能擅自覆盖原始统计。"""
    with tempfile.TemporaryDirectory() as td:
        run = Path(td)
        path = run / "clarify.json"
        original = {"clarify_secs": 12.5, "total_tokens": 99}
        path.write_text(json.dumps(original), encoding="utf-8")
        got = _sync_clarify_phase_stats(run, {"phase": None})
        on_disk = json.loads(path.read_text(encoding="utf-8"))
    assert got is None
    assert on_disk == original


if __name__ == "__main__":
    for fn in [v for k, v in sorted(globals().items()) if k.startswith("test_")]:
        fn()
    print("\nall usage tests passed")


def test_opencode_phase_anchor_from_text_clarify_round():
    """opencode 上 agent 走正文提问（无 question tool）：锚点取 clarify.json 的 answered_at_ms。

    修前锚点只认 question tool → clarify 全记 0 → cli._sync 再用 0 覆盖 loop 的真实值
    （实测 9 问 → clarify_secs 0.0）。
    """
    with tempfile.TemporaryDirectory() as td:
        run = Path(td)
        _mk_db(run / ".agent_home/data/opencode/opencode.db", [
            {"input": 100, "output": 10, "t": 1_000_000},     # 提问轮
            {"input": 900, "output": 90, "t": 1_060_000},     # 求解轮
            {"input": 900, "output": 90, "t": 1_120_000},
        ])
        (run / "clarify.json").write_text(json.dumps({
            "total_rounds": 1, "rounds": [{"turn": 1, "channel": "text",
                                           "questions": [{"question": "q?"}],
                                           "answered_at_ms": 1_030_000}]}), encoding="utf-8")
        got = collect_usage(run, "chat/glm-5.2")
    assert got["phase"]["clarify_tokens"] == 110
    assert got["phase"]["solve_tokens"] == 1980
    assert got["phase"]["clarify_secs"] == 30.0


def test_opencode_phase_unknown_when_clarified_but_no_anchor():
    """有澄清轮次却无任何锚点（旧格式）：记 None 不记 0，cli._sync 据此不覆盖。"""
    with tempfile.TemporaryDirectory() as td:
        run = Path(td)
        _mk_db(run / ".agent_home/data/opencode/opencode.db", STEPS)
        (run / "clarify.json").write_text(json.dumps({
            "total_rounds": 1, "rounds": [{"turn": 1, "questions": [{"question": "q?"}]}]}),
            encoding="utf-8")
        got = collect_usage(run, "chat/glm-5.2")
        assert got["phase"]["clarify_secs"] is None
        assert _sync_clarify_phase_stats(run, got) is None


def test_phase_stats_source_names_backend_ledger_for_non_opencode():
    """非 opencode 后端的阶段统计来源不能冒充 opencode 时间线。"""
    with tempfile.TemporaryDirectory() as td:
        run = Path(td)
        (run / "clarify.json").write_text(json.dumps({"total_rounds": 1, "rounds": []}), encoding="utf-8")
        got = _sync_clarify_phase_stats(run, {"source": "backend_metadata",
                                              "phase": {"clarify_secs": 12.0, "clarify_tokens": 5}})
    assert got["phase_stats_source"] == "backend_ledger"
