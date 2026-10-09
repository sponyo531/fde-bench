"""跨 run 聚合：产出端到端指标表。

指标定义（与澄清类基准口径一致，见 harness/clarify_score.py）：

  有效率            valid_rate    = validity>0 的 run 占比
  Avg(all)          avg_all       = 全部 run 平均分（含 0 分）→ 稳定性 + 质量
  Avg(valid)        avg_valid     = 仅有效解的平均分 → "做出来时做得多好"
  Avg 时长/try      avg_minutes   = 平均耗时（分钟）
  Avg 工具调用/try  avg_tools     = 平均工具调用次数
  Cache 命中率      cache_hit     = 命中 token / 全部读入 token
  Avg 费用/try      avg_cost_usd  = 平均 API 费用（含缓存折扣）
  澄清指标          ask_f1 / recall / precision / kqc_first_turn / redundant
  CE-A / CE-B       问对方向的比例 / 问对且真正推进的比例
  Judge 一致性      agreement / 非全票一致率 / 全模型失败率 / 逐模型偏差
  条件成功率        valid_rate_asked   vs  valid_rate_not_asked
                    Dialogue-SWEBench：按"该次是否至少问过一次"拆分有效率，
                    检验"选择去问"是否真与任务需要澄清相关，而非问了也白问

统计口径两条原则：
  1. degraded run（无文本无产物，基础设施故障）不计入任何分数统计，只报数量——
     否则会把"没跑起来"伪装成"agent 能力差"。
  2. 未采集到的字段用 None 表示并跳过，不用 0 填充。
"""

from __future__ import annotations

import json
import statistics
from pathlib import Path

from ..results import derive_run_statuses

EXCLUDED_STATUS = {"degraded", "error"}

_CONDITION_ALIASES = {
    "R": "Hidden", "C": "Interact", "CF": "Interact-Req", "F": "Full",
    "CF-confirm": "Interact-Conf", "F_base": "Full-Base",
    "F_data": "Full-Data", "F_rule": "Full-Rule",
}


def _read(path: Path) -> dict:
    if not path.is_file():
        return {}
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError:
        return {}


def load_run(run_dir: Path) -> dict:
    """读取一次 run 的全部指标，扁平化成一行记录。"""
    man = _read(run_dir / "manifest.json")
    use = _read(run_dir / "usage.json")
    res = _read(run_dir / "result.json")
    clr = _read(run_dir / "clarify_score.json")
    tok = use.get("token_usage") or {}
    token_values = tok.get("tokens") or {}
    availability = dict(tok.get("availability") or {})
    steps_unit = tok.get("steps_unit")
    raw_steps = token_values.get("steps")
    llm_calls = tok.get("llm_calls")
    if llm_calls is None and steps_unit == "llm_call":
        # 兼容新增 llm_calls 字段之前的 run；单位明确时才允许回推。
        llm_calls = raw_steps
    # 旧 usage.json 没有 availability；从已有值安全回推，绝不从缺失值猜 0。
    availability.setdefault("tokens", bool(token_values and any(
        token_values.get(k) is not None for k in ("input", "output", "reasoning"))))
    for public, raw in (("input_tokens", "input"), ("output_tokens", "output"),
                        ("reasoning_tokens", "reasoning"),
                        ("cache_read_tokens", "cache_read"),
                        ("cache_write_tokens", "cache_write")):
        availability.setdefault(public, token_values.get(raw) is not None)
    availability.setdefault("cache_hit_rate", tok.get("cache_hit_rate") is not None)
    availability.setdefault("cost_usd", tok.get("cost_usd") is not None)
    availability.setdefault("llm_calls", llm_calls is not None)
    phase_values = tok.get("phase") or {}
    availability.setdefault("phase_tokens", all(
        phase_values.get(k) is not None for k in ("clarify_tokens", "solve_tokens")))
    availability.setdefault("phase_time", all(
        phase_values.get(k) is not None for k in ("clarify_secs", "solve_secs")))

    score = res.get("combined_score", res.get("overall_score"))
    validity = res.get("validity_score", res.get("validity"))
    statuses = derive_run_statuses(run_dir)

    effective = tok.get("effective_models") or use.get("effective_models") or []
    requested = (man.get("agent_env") or {}).get("model") or man.get("model")
    raw_judge_errors = clr.get("judge_errors", 0)
    try:
        judge_failed_decisions = max(0, int(raw_judge_errors or 0))
    except (TypeError, ValueError):
        judge_failed_decisions = 0
    raw_judge_decisions = clr.get("judge_total_decisions")
    if raw_judge_decisions is None:
        raw_judge_decisions = len(clr.get("audit") or []) + judge_failed_decisions
    try:
        judge_decisions = max(0, int(raw_judge_decisions or 0))
    except (TypeError, ValueError):
        judge_decisions = 0

    return {
        "run_id": man.get("run_id", run_dir.name),
        "case": man.get("source_case") or man.get("case") or run_dir.name.split("__", 1)[0],
        "source_case": man.get("source_case"),
        "case_variant": man.get("case_variant"),
        "condition": _CONDITION_ALIASES.get(man.get("condition"), man.get("condition")),
        "scaffold": man.get("scaffold"),
        "model": requested,
        "effective_models": effective,
        "model_mismatch": bool(man.get("model_mismatch")),
        "status": use.get("status"),
        **statuses,
        "score": score,
        "validity": validity,
        "minutes": (use.get("elapsed_s") or 0) / 60 or None,
        "agent_sends": use.get("agent_sends"),
        "tool_calls": use.get("tool_calls"),
        "tool_calls_by_kind": use.get("tool_calls_by_kind"),
        "cache_hit_rate": tok.get("cache_hit_rate"),
        "cost_usd": tok.get("cost_usd"),
        "pricing_warning": tok.get("pricing_warning"),
        # Usage comparability is explicit: a missing value means the backend
        # did not expose reliable metadata, never zero usage.
        "usage_source": tok.get("source") or ("opencode_sqlite" if tok.get("tokens") else None),
        "tokens_available": availability["tokens"],
        "cache_available": availability["cache_hit_rate"],
        # 有阶段 dict 但全是 None（切不出来）不算可用
        "phase_available": any(v is not None for v in (tok.get("phase") or {}).values()),
        "steps_unit": steps_unit,
        "llm_calls": llm_calls,
        "usage_precision": tok.get("precision"),
        "phase_precision": tok.get("phase_precision"),
        "cost_precision": tok.get("cost_precision"),
        "usage_availability": availability,
        # 真正的数：总 token / 阶段 token / 阶段耗时（秒）。缺失为 None，聚合时跳过
        "tokens_total": (lambda t: (sum(v for v in (t.get("input"), t.get("output"), t.get("reasoning"))
                                        if v is not None) if t else None))(tok.get("tokens") or {}),
        "tokens_input": token_values.get("input"),
        "tokens_output": token_values.get("output"),
        "steps": raw_steps,
        "clarify_tokens": (tok.get("phase") or {}).get("clarify_tokens"),
        "solve_tokens": (tok.get("phase") or {}).get("solve_tokens"),
        "clarify_secs": (tok.get("phase") or {}).get("clarify_secs"),
        "solve_secs": (tok.get("phase") or {}).get("solve_secs"),
        "ask_f1": clr.get("ask_f1"),
        "ask_recall": clr.get("recall"),
        "ask_precision": clr.get("precision"),
        "kqc_first_turn": clr.get("kqc_first_turn"),
        "atc": clr.get("atc"),
        # 白问：无人应答的条件下仍提问，见 cli.py。R 被这类空转压低会让
        # `F − R` 虚高，必须能在报表里看见。
        "unanswered_asks": (use.get("unanswered_asks") or {}).get("n_questions"),
        "redundant_questions": clr.get("redundant_questions"),
        "n_questions": clr.get("n_questions"),
        "ce_a": clr.get("ce_a"),
        "ce_b": clr.get("ce_b"),
        "judge_agreement_mean": clr.get("judge_agreement_mean"),
        "judge_unanimous_rate": clr.get("judge_unanimous_rate"),
        "judge_decisions": judge_decisions,
        "judge_failed_decisions": judge_failed_decisions,
        "judge_vote_stats": clr.get("judge_vote_stats") or {},
        # 保留逐票失败详情，报告消费者无需绕过 load_run 再读原始 JSON。
        # 内容已在 clarify.score 写入时脱敏；旧结果没有该字段时为空列表。
        "judge_error_details": clr.get("judge_error_details") or [],
        "arbitration_model": clr.get("arbitration_model"),
        "arbitration_calls": clr.get("arbitration_calls", 0),
        "arbitration_used": clr.get("arbitration_used", 0),
        "arbitration_errors": clr.get("arbitration_errors", 0),
        "arbitration_error_details": clr.get("arbitration_error_details") or [],
        # {标签: {n, covered, asked, ...}}。聚合时**累加计数再相除**，不是
        # 对各 run 的 recall 求平均——各 case 每档的条数差得很远（「评分口径」
        # 187 条 vs「数据冲突」13 条），比率平均会让小样本档位话语权虚高。
        "by_tag": clr.get("by_tag") or {},
    }


def collect(results_root: Path) -> list[dict]:
    """扫描直接或 batch 布局，并按 run_id 保留最新一次重试。"""
    return collect_many([results_root])


def collect_many(results_roots) -> list[dict]:
    from ..results import discover_run_dirs
    return [load_run(d) for d in discover_run_dirs(results_roots)]


def _sum_kinds(dicts: list) -> dict:
    """tool_calls_by_kind 逐类累加（shell / read / edit / search / other）。"""
    out: dict = {}
    for d in dicts:
        if isinstance(d, dict):
            for k, v in d.items():
                if isinstance(v, (int, float)):
                    out[k] = out.get(k, 0) + v
    return out


def _mean(values: list) -> float | None:
    nums = [v for v in values if isinstance(v, (int, float))]
    return round(statistics.fmean(nums), 4) if nums else None


def _count_values(rows: list[dict], field: str) -> dict[str, int]:
    """计数离散状态，保留 unknown/缺失而不静默丢弃。"""
    out: dict[str, int] = {}
    for row in rows:
        value = str(row.get(field) or "unknown")
        out[value] = out.get(value, 0) + 1
    return dict(sorted(out.items()))


def _conditional_valid_rate(usable: list[dict]) -> dict:
    """按"该次 run 是否至少问过一次"拆分有效率（Dialogue-SWEBench 口径）。

    只在能确定提问次数的 run 上统计；n_questions 为 None 表示未采集，跳过。
    """
    known = [r for r in usable if isinstance(r.get("n_questions"), int)
             and isinstance(r.get("validity"), (int, float))]
    asked = [r for r in known if r["n_questions"] > 0]
    silent = [r for r in known if r["n_questions"] == 0]

    def rate(rows: list[dict]) -> float | None:
        if not rows:
            return None
        return round(sum(1 for r in rows if (r["validity"] or 0) > 0) / len(rows), 4)

    return {
        "n_asked": len(asked),
        "n_not_asked": len(silent),
        "valid_rate_asked": rate(asked),
        "valid_rate_not_asked": rate(silent),
    }


def _agg_judge_stats(rows: list[dict]) -> dict:
    """跨 run 汇总五模型 judge 的一致性与逐模型偏差。

    ``judge_agreement_mean`` 是每个 run 内成功考点判定的均值，因此这里按
    判定次数加权，而不是让考点很少的 run 与大 case 等权。逐模型字段来自
    ``clarify_score.json.judge_vote_stats``；旧版结果没有该字段时只报告可用的
    run 级统计，不伪造 0。
    """
    total = failed = successful = unanimous_known = 0
    agreement_sum = unanimous_sum = 0.0
    model_acc: dict[str, dict[str, int]] = {}

    for row in rows:
        try:
            n = int(row.get("judge_decisions", 0) or 0)
            f = int(row.get("judge_failed_decisions", 0) or 0)
        except (TypeError, ValueError):
            n = f = 0
        if n < 0 or f < 0:
            continue
        total += n
        failed += f
        mean = row.get("judge_agreement_mean")
        unanimous = row.get("judge_unanimous_rate")
        if isinstance(mean, (int, float)) and n > f:
            ok = n - f
            successful += ok
            agreement_sum += float(mean) * ok
            if isinstance(unanimous, (int, float)):
                unanimous_known += ok
                unanimous_sum += float(unanimous) * ok

        for model, stats in (row.get("judge_vote_stats") or {}).items():
            if not isinstance(stats, dict):
                continue
            acc = model_acc.setdefault(model, {
                "votes": 0, "errors": 0, "covered_yes": 0,
                "covered_disagree": 0, "attempted_yes": 0,
                "attempted_disagree": 0,
            })
            for key in acc:
                try:
                    acc[key] += int(stats.get(key, 0) or 0)
                except (TypeError, ValueError):
                    pass

    out = {
        "judge_total_decisions": total,
        "judge_failed_decisions": failed,
        "judge_failure_rate": round(failed / total, 4) if total else None,
        "judge_successful_decisions": successful,
        "judge_agreement_mean": (
            round(agreement_sum / successful, 4) if successful else None),
        "judge_unanimous_rate": (
            round(unanimous_sum / unanimous_known, 4) if unanimous_known else None),
        "judge_disagreement_rate": (
            round(1 - unanimous_sum / unanimous_known, 4)
            if unanimous_known else None),
    }

    by_model = {}
    for model, stats in sorted(model_acc.items()):
        votes = stats["votes"]
        errors = stats["errors"]
        ok = max(0, votes - errors)
        by_model[model] = {
            **stats,
            "error_rate": round(errors / votes, 4) if votes else None,
            "covered_yes_rate": (
                round(stats["covered_yes"] / ok, 4) if ok else None),
            "covered_disagreement_rate": (
                round(stats["covered_disagree"] / ok, 4) if ok else None),
            "attempted_yes_rate": (
                round(stats["attempted_yes"] / ok, 4) if ok else None),
            "attempted_disagreement_rate": (
                round(stats["attempted_disagree"] / ok, 4) if ok else None),
        }
    out["judge_by_model"] = by_model
    return out



# ── Sufficiency 分层 ─────────────────────────────────────────────────────────

def insufficient_cases(runs: list[dict], *, min_validity: float = 1.0) -> set[str]:
    """F 条件也做不出来的 case。

    这类 case 卡的是**求解能力**而非需求理解——信息都给全了仍不合格，说明
    澄清与否根本影响不到结果。把它们混进 `F − R` 只会稀释信号、放大噪声，
    所以澄清分析要能把它们摘出去（Ambig-DS 的 Δ 口径同理：配对分数里若
    S_full 本身就是 0，那个差值没有意义）。

    判据取 validity 而非 quality：validity=0 是"违反硬约束"，是硬失败；
    quality 低只是方案差，仍属可分析范围。

    只看 F（信息最全的那档）；没跑过 F 的 case 无从判断，视为充分（不剔除），
    以免因为没跑 F 就悄悄少算一批 case。
    """
    seen_f: dict[str, list[float]] = {}
    for r in runs:
        if _CONDITION_ALIASES.get(r.get("condition"), r.get("condition")) != "Full":
            continue
        v = r.get("validity")
        if v is not None:
            seen_f.setdefault(r["case"], []).append(float(v))
    return {c for c, vs in seen_f.items()
            if vs and max(vs) < min_validity}


def split_by_sufficiency(runs: list[dict]) -> tuple[list[dict], list[dict], set[str]]:
    """按 case 是否"信息给全就能做出来"拆成两组。返回 (充分, 不充分, case 名集合)。"""
    bad = insufficient_cases(runs)
    keep = [r for r in runs if r.get("case") not in bad]
    drop = [r for r in runs if r.get("case") in bad]
    return keep, drop, bad


def summarize(runs: list[dict]) -> dict:
    """按 (scaffold, model, condition) 分组汇总。"""
    groups: dict[tuple, list[dict]] = {}
    for r in runs:
        groups.setdefault((r["scaffold"], r["model"], r["condition"]), []).append(r)

    out = {}
    for key, rows in sorted(groups.items(), key=lambda kv: [str(x) for x in kv[0]]):
        excluded = [r for r in rows if r["status"] in EXCLUDED_STATUS]
        usable = [r for r in rows if r["status"] not in EXCLUDED_STATUS]
        scored = [r for r in usable if isinstance(r.get("score"), (int, float))]
        validity_known = [r for r in usable
                          if isinstance(r.get("validity"), (int, float))]
        scores = [r["score"] for r in scored]
        valid = [r["score"] for r in scored
                 if isinstance(r.get("validity"), (int, float))
                 and r["validity"] > 0]
        n_valid = sum(1 for r in validity_known if r["validity"] > 0)

        out["/".join(str(k) for k in key)] = {
            "n_runs": len(rows),
            "n_excluded": len(excluded),
            "n_unscored": len(usable) - len(scored),
            "excluded_status": sorted({r["status"] for r in excluded if r["status"]}),
            "run_status_counts": _count_values(rows, "run_status"),
            "artifact_status_counts": _count_values(rows, "artifact_status"),
            "scoring_status_counts": _count_values(rows, "scoring_status"),
            "validity_status_counts": _count_values(rows, "validity_status"),
            "valid_rate": (round(n_valid / len(validity_known), 4)
                           if validity_known else None),
            "avg_all": _mean(scores),
            "avg_valid": _mean(valid),
            "avg_minutes": _mean([r.get("minutes") for r in usable]),
            "avg_agent_sends": _mean([r.get("agent_sends") for r in usable]),
            "avg_llm_calls": _mean([r.get("llm_calls") for r in usable]),
            "avg_tools": _mean([r.get("tool_calls") for r in usable]),
            "cache_hit_rate": _mean([r.get("cache_hit_rate") for r in usable]),
            "avg_cost_usd": _mean([r.get("cost_usd") for r in usable]),
            "n_unpriced": sum(1 for r in usable if r.get("pricing_warning")),
            "usage_coverage": {
                "tokens": sum(bool(r.get("tokens_available")) for r in usable),
                "cache": sum(bool(r.get("cache_available")) for r in usable),
                "phase": sum(bool(r.get("phase_available")) for r in usable),
                "reasoning": sum(bool((r.get("usage_availability") or {}).get(
                    "reasoning_tokens")) for r in usable),
                "cache_write": sum(bool((r.get("usage_availability") or {}).get(
                    "cache_write_tokens")) for r in usable),
                "llm_calls": sum(r.get("llm_calls") is not None for r in usable),
                "phase_tokens": sum(bool((r.get("usage_availability") or {}).get(
                    "phase_tokens")) for r in usable),
                "phase_time": sum(bool((r.get("usage_availability") or {}).get(
                    "phase_time")) for r in usable),
                "n_runs": len(usable),
            },
            # 真正的用量汇总（缺失跳过而非记 0）。跨脚手架比较前看 steps_units /
            # usage_precisions：steps 单位不一致（send vs llm_call）的组不可比 steps，
            # partial 表示超时后的下界估计
            "avg_tokens_total": _mean([r.get("tokens_total") for r in usable]),
            "avg_tokens_input": _mean([r.get("tokens_input") for r in usable]),
            "avg_tokens_output": _mean([r.get("tokens_output") for r in usable]),
            "avg_steps": _mean([r.get("steps") for r in usable]),
            "avg_clarify_tokens": _mean([r.get("clarify_tokens") for r in usable]),
            "avg_solve_tokens": _mean([r.get("solve_tokens") for r in usable]),
            "avg_clarify_secs": _mean([r.get("clarify_secs") for r in usable]),
            "avg_solve_secs": _mean([r.get("solve_secs") for r in usable]),
            "steps_units": sorted({str(r.get("steps_unit")) for r in usable if r.get("steps_unit")}),
            "usage_precisions": sorted({str(r.get("usage_precision")) for r in usable if r.get("usage_precision")}),
            "phase_precisions": sorted({str(r.get("phase_precision")) for r in usable
                                         if r.get("phase_precision")}),
            "cost_precisions": sorted({str(r.get("cost_precision")) for r in usable
                                        if r.get("cost_precision")}),
            "tool_kinds": _sum_kinds([r.get("tool_calls_by_kind") for r in usable]),
            **_conditional_valid_rate(usable),
            "ce_a": _mean([r.get("ce_a") for r in usable]),
            "ce_b": _mean([r.get("ce_b") for r in usable]),
            **_agg_judge_stats(usable),
            "ask_f1": _mean([r.get("ask_f1") for r in usable]),
            "ask_recall": _mean([r.get("ask_recall") for r in usable]),
            "ask_precision": _mean([r.get("ask_precision") for r in usable]),
            "kqc_first_turn": _mean([r.get("kqc_first_turn") for r in usable]),
            "avg_questions": _mean([r.get("n_questions") for r in usable]),
            "avg_atc": _mean([r.get("atc") for r in usable]),
            # 白问按「有多少个 run 出现过」计，比均值直观：它是个是非问题
            "n_runs_unanswered_asks": sum(
                1 for r in usable if (r.get("unanswered_asks") or 0) > 0),
            "avg_redundant": _mean([r.get("redundant_questions") for r in usable]),
            "by_tag": _agg_by_tag(usable),
        }
    return out


def _agg_by_tag(rows: list[dict]) -> dict:
    """把各 run 的 by_tag 汇成一档一行。

    累加 n / covered / asked 再相除，而不是对各 run 的 recall 求平均：
    每档条数悬殊（全库「评分口径」187 条、「数据冲突」只有 13 条），
    比率平均会把 13 条那档抬到与 187 条同等话语权。
    """
    acc: dict[str, dict] = {}
    for r in rows:
        for tag, d in (r.get("by_tag") or {}).items():
            a = acc.setdefault(tag, {"n": 0, "covered": 0, "asked": 0, "n_runs": 0})
            a["n"] += d.get("n", 0)
            a["covered"] += d.get("covered", 0)
            a["asked"] += d.get("asked", 0)
            a["n_runs"] += 1
    for a in acc.values():
        a["recall"] = round(a["covered"] / a["n"], 4) if a["n"] else None
        a["ce_a"] = round(a["covered"] / a["asked"], 4) if a["asked"] else None
    # 按 recall 升序——最瞎的那一类排最前，这才是要看的
    return dict(sorted(acc.items(), key=lambda kv: (kv[1]["recall"] is None,
                                                    kv[1]["recall"])))


def _pad(s: str, width: int, right: bool = False) -> str:
    """按显示宽度对齐——汉字占两列，str.ljust 按字符数算会歪。"""
    import unicodedata
    w = sum(2 if unicodedata.east_asian_width(ch) in "WF" else 1 for ch in s)
    fill = " " * max(0, width - w)
    return fill + s if right else s + fill


def render_tag_table(summary: dict) -> str:
    """按信息缺口标签渲染覆盖率表。每个分组一块，最瞎的标签排最前。"""
    lines: list[str] = []
    for name, s in summary.items():
        bt = s.get("by_tag") or {}
        if not bt:
            continue
        head = (_pad("标签", 24) + _pad("条数", 8, True) + _pad("问对", 8, True)
                + _pad("问过", 8, True) + _pad("Recall", 9, True)
                + _pad("CE-A", 9, True))
        lines.append(f"\n【{name}】")
        lines.append(head)
        lines.append("-" * 66)
        for tag, d in bt.items():
            rec = f"{d['recall']:.1%}" if d["recall"] is not None else "-"
            cea = f"{d['ce_a']:.1%}" if d["ce_a"] is not None else "-"
            lines.append(_pad(tag, 24) + _pad(str(d["n"]), 8, True)
                         + _pad(str(d["covered"]), 8, True)
                         + _pad(str(d["asked"]), 8, True)
                         + _pad(rec, 9, True) + _pad(cea, 9, True))
    return "\n".join(lines) if lines else "(无 by_tag 数据——run 是旧版打的分)"


def render_table(summary: dict) -> str:
    """渲染为纯文本表格。"""
    cols = [
        ("有效率", "valid_rate", "{:.1%}"),
        ("Avg(all)", "avg_all", "{:.4f}"),
        ("Avg(valid)", "avg_valid", "{:.4f}"),
        ("分钟", "avg_minutes", "{:.1f}"),
        ("发送", "avg_agent_sends", "{:.1f}"),
        ("LLM步", "avg_llm_calls", "{:.1f}"),
        ("工具", "avg_tools", "{:.1f}"),
        ("kTok", "avg_tokens_total", "{:.0f}"),
        ("澄清s", "avg_clarify_secs", "{:.0f}"),
        ("Cache", "cache_hit_rate", "{:.1%}"),
        ("费用$", "avg_cost_usd", "{:.4f}"),
        ("未定价", "n_unpriced", "{:d}"),
        ("Ask-F1", "ask_f1", "{:.3f}"),
        ("CE-A", "ce_a", "{:.3f}"),
        ("CE-B", "ce_b", "{:.3f}"),
        ("提问", "avg_questions", "{:.1f}"),
        ("J一致", "judge_agreement_mean", "{:.3f}"),
        ("J非全同", "judge_disagreement_rate", "{:.1%}"),
        ("J失败", "judge_failure_rate", "{:.1%}"),
    ]
    head = (f"{'分组':<44}{'n':>4}{'排除':>5}{'未评分':>7}  "
            + "".join(f"{c[0]:>11}" for c in cols))
    lines = [head, "-" * len(head)]
    for name, s in summary.items():
        row = (f"{name[:43]:<44}{s['n_runs']:>4}{s['n_excluded']:>5}"
               f"{s.get('n_unscored', 0):>7}  ")
        for _, key, fmt in cols:
            v = s.get(key)
            if key == "avg_tokens_total" and isinstance(v, (int, float)):
                v = v / 1000                          # 列名 kTok：千 token
            row += f"{fmt.format(v) if isinstance(v, (int, float)) else '-':>11}"
        lines.append(row)
    return "\n".join(lines)


def main() -> None:
    import argparse
    p = argparse.ArgumentParser(description="聚合 run 结果，产出指标表")
    p.add_argument("results", type=Path, nargs="+", help="results 目录（可多个）")
    p.add_argument("--json", type=Path, help="同时写出 JSON")
    p.add_argument("--sufficient-only", action="store_true",
                   help="剔除 F 条件下仍 validity=0 的 case（求解能力问题，"
                        "不该计入澄清分析）")
    p.add_argument("--by-tag", action="store_true",
                   help="另打一张按信息缺口标签拆开的覆盖率表（数据语义/"
                        "评分口径/计算口径…），看模型是在哪一类信息上瞎")
    args = p.parse_args()

    runs = collect_many(args.results)
    if not runs:
        raise SystemExit("未找到任何 run")

    dropped: list[dict] = []
    bad: set[str] = set()
    if args.sufficient_only:
        runs, dropped, bad = split_by_sufficiency(runs)
        if not runs:
            raise SystemExit("按 sufficiency 过滤后无剩余 run")

    summary = summarize(runs)
    print(render_table(summary))
    if args.by_tag:
        print("\n" + "=" * 62)
        print("按信息缺口标签拆开（条数=该档澄清项总数，累加计数后相除）")
        print("=" * 62)
        print(render_tag_table(summary))
    if bad:
        print(f"\n[sufficiency] 剔除 {len(bad)} 个 case（F 条件下 validity 仍为 0，"
              f"卡在求解能力而非需求理解），涉及 {len(dropped)} 个 run：")
        for c in sorted(bad):
            print(f"  - {c}")
    if args.json:
        args.json.write_text(
            json.dumps({"runs": runs, "summary": summary,
                        "insufficient_cases": sorted(bad)},
                       ensure_ascii=False, indent=2),
            encoding="utf-8",
        )
        print(f"\n已写出 {args.json}")


if __name__ == "__main__":
    main()
