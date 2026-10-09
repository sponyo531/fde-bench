"""给已经跑完的 run 目录补评分 —— `python3 -m harness.scoring`。

为什么需要一个独立入口：正常路径是 `harness.cli` 求解完直接评分（cli.py:204
调 score_run）。在隔离调度环境中求解与评分可以分成两个 job：求解 job 用
受限挂载把可见范围收窄到 run 目录，好让 agent 物理上够不到 gt.json 与
tests/；而评分恰恰要读 tests/evaluator.py 与 gt.json，非挂全库不可。于是
求解 job 带 `--no-eval`，评分留给本入口在另一个 job 里做。

本入口**复刻 cli.py 的评分门槛**，不是另立一套：

  求解分  status ∈ {ok, timeout} 且 produced 非空 才评
          —— 超时但有产物代表"做出来了但违约/没做完"，与 degraded（根本没
             跑起来）是两种结论，混在一起会一起记 0 分，抹掉区别。
  澄清分  只要有 clarify.json 就评，与产物无关
          —— "问得对但没做出来"和"没问也没做出来"是不同结论，瓶颈在哪一段
             要能分开看。

用法：
    python3 -m harness.scoring --case <case 目录> --results <run 的 results 目录>
    python3 -m harness.scoring --case ... --results ... --only <run_id>   # 重评单个
"""
from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path


def _load(p: Path) -> dict:
    try:
        return json.loads(p.read_text(encoding="utf-8"))
    except Exception:                                    # noqa: BLE001
        return {}


def _number(value: object) -> bool:
    return (isinstance(value, (int, float)) and not isinstance(value, bool)
            and math.isfinite(value))


def _valid_numeric_result(result: dict) -> bool:
    """Whether the scorer produced a real benchmark score, including zero."""
    if result.get("scoring_status") == "failed":
        return False
    return any(_number(result.get(k))
               for k in ("final_score", "overall_score", "combined_score"))


def _valid_solve_result(result: dict) -> bool:
    """A score or an explicit no-deliverable terminal outcome closes solving."""
    return (_valid_numeric_result(result)
            or result.get("scoring_status") == "not_applicable")


def _valid_clarify_result(result: dict) -> bool:
    """Whether a persisted clarification score has all required decisions."""
    if not result or "n_questions" not in result or "n_blockers" not in result:
        return False
    if result.get("judge_failed") is True:
        return False
    try:
        return int(result.get("judge_errors", 0) or 0) <= 0
    except (TypeError, ValueError):
        return False


def _write_terminal_result(rd: Path, status: str, produced: list,
                           *, scoring_failed: bool = False) -> None:
    """Materialize a null-score terminal record without inventing a zero."""
    path = rd / "result.json"
    if path.is_file():
        return
    manifest = _load(rd / "manifest.json")
    if scoring_failed or produced:
        scoring_status, artifact_status = "failed", "partial"
        category = "scoring_failed" if scoring_failed else "run_not_scorable"
        message = ("评分入口执行失败，未得到可用数值分数" if scoring_failed
                   else f"求解状态为 {status}，未得到可用数值分数")
    elif status in {"ok", "timeout"}:
        scoring_status, artifact_status = "not_applicable", "none"
        category = "artifact_missing"
        message = "求解正常结束但 Agent 未产出交付物，评分不适用"
    else:
        scoring_status, artifact_status = "not_applicable", "none"
        category = "run_failed"
        message = f"求解阶段状态为 {status} 且无交付物，评分不适用"
    record = {
        "case": manifest.get("case"), "condition": manifest.get("condition"),
        "model": manifest.get("model"), "run_status": status,
        "artifact_status": artifact_status, "scoring_status": scoring_status,
        "final_score": None, "overall_score": None, "combined_score": None,
        "error_info": {"category": category, "message": message},
    }
    path.write_text(json.dumps(record, ensure_ascii=False, indent=2), encoding="utf-8")


def _write_scoring_marker(rd: Path, *, solve_required: bool, solve_ok: bool,
                          clarify_required: bool, clarify_ok: bool) -> None:
    complete = ((not solve_required or solve_ok)
                and (not clarify_required or clarify_ok))
    payload = {
        "version": 1, "status": "complete" if complete else "failed",
        "solve_required": solve_required, "solve_ok": solve_ok,
        "clarify_required": clarify_required, "clarify_ok": clarify_ok,
    }
    tmp = rd / "scoring_complete.json.tmp"
    tmp.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    tmp.replace(rd / "scoring_complete.json")


def _run_dirs(results: Path) -> list[Path]:
    """results 下的 run 目录 = 含 usage.json 的目录（最深两层足够）。

    不写死目录名规则：run_id 的构造在 harness/run.py 里，将来改了名这里不该跟着改。
    以 usage.json 为准是因为它由 finalize() 无条件写出，是 run 完成的唯一标志。
    """
    out = [p.parent for p in sorted(results.rglob("usage.json"))]
    # 去重且保持顺序（rglob 已排序，但同目录可能被上层 glob 重复命中）
    seen, uniq = set(), []
    for p in out:
        if p not in seen:
            seen.add(p)
            uniq.append(p)
    return uniq


def main() -> int:
    from .triple import _MODELS as extractor_models

    ap = argparse.ArgumentParser(description="给已跑完的 run 补评分")
    ap.add_argument("--case", required=True, help="case 目录（要能读到 tests/ 与 gt.json）")
    ap.add_argument("--results", required=True, help="run 的 results 目录")
    ap.add_argument("--only", default="", help="只评这个 run 目录名")
    ap.add_argument("--timeout", type=int, default=None, help="单次评分秒数上限")
    ap.add_argument("--rescore", action="store_true",
                    help="已有 result.json 也重评（默认跳过，避免重复烧抽取器 token）")
    ap.add_argument("--rerun-extractor", choices=tuple(tag for tag, _ in extractor_models),
                    help="只重跑指定抽取模型，复用另外两票并重新投票")
    ap.add_argument("--failed-only", action="store_true",
                    help="与 --rerun-extractor 联用：仅重抽没有归一化 payload 的票")
    ap.add_argument("--arbitrate", action="store_true",
                    help="澄清 Judge 出现分歧/失败时，追加一次仲裁（默认 gpt-5.6-sol）")
    ap.add_argument("--arbitration-model", default="responses/gpt-5.6-sol",
                    help="仲裁模型 ID（仅与 --arbitrate 联用）")
    a = ap.parse_args()

    if a.failed_only and not a.rerun_extractor:
        ap.error("--failed-only 必须与 --rerun-extractor 联用")
    if a.rerun_extractor and a.rescore:
        ap.error("--rerun-extractor 与 --rescore 不能同时使用")

    case = Path(a.case).resolve()
    results = Path(a.results).resolve()
    if not case.is_dir():
        print(f"FATAL: case 目录不存在：{case}", file=sys.stderr)
        return 66
    # 评分必须看得到 tests/ —— 少了它就是挂载范围不对（比如误带了 --subpath），
    # 这时 evaluator 会 import 失败并被 except 兜成 combined_score=0.0，
    # 整批 run 静默记 0 分。宁可在这里炸掉。
    if not (case / "tests" / "evaluator.py").is_file():
        print(f"FATAL: {case} 下没有 tests/evaluator.py。评分 job 必须挂评测目录，"
              f"不能带 --subpath。", file=sys.stderr)
        return 66
    if not results.is_dir():
        print(f"FATAL: results 目录不存在：{results}", file=sys.stderr)
        return 66

    from ..results import update_run_statuses

    # Direct-layout runs live at ``<condition>/tryN``.  Prefer that exact
    # directory when --only is supplied: a recursive name filter also matches
    # ``_retries/<timestamp>/tryN`` and used to score every archived retry in
    # addition to the current run.
    direct = results / a.only if a.only else None
    if direct is not None and (direct / "usage.json").is_file():
        runs = [direct]
    else:
        runs = _run_dirs(results)
        if a.only:
            runs = [r for r in runs if r.name == a.only]
    if not runs:
        print(f"FATAL: {results} 下没有找到任何 run（判据：存在 usage.json）",
              file=sys.stderr)
        return 66

    if a.rerun_extractor:
        from .triple import extractor_needs_rerun, reextract_model

        n_done = n_skipped = n_failed = 0
        for rd in runs:
            tag = f"[{rd.name}]"
            if a.failed_only and not extractor_needs_rerun(rd, a.rerun_extractor):
                print(f"{tag} {a.rerun_extractor} 已有 payload，跳过")
                n_skipped += 1
                continue
            try:
                r = reextract_model(case, rd, a.rerun_extractor, a.timeout)
                print(f"{tag} 重抽 {a.rerun_extractor}: "
                      f"status={r['extraction_status']} score={r['score']} "
                      f"→ final={r['overall_score']} ({r['vote']})")
                update_run_statuses(rd)
                n_done += 1
            except Exception as e:                       # noqa: BLE001
                print(f"{tag} 重抽失败：{type(e).__name__}: {e}")
                n_failed += 1
        print(f"\n[reextract] run={len(runs)} 完成={n_done} 跳过={n_skipped} "
              f"失败={n_failed}")
        return 1 if n_failed else 0

    from .solve import score_run, summarize

    n_scored = n_skipped = n_clarify = 0
    n_score_failed = n_clarify_failed = n_terminal = 0
    for rd in runs:
        u = _load(rd / "usage.json")
        status, produced = u.get("status"), (u.get("produced") or [])
        tag = f"[{rd.name}]"
        solve_required = status in ("ok", "timeout") and bool(produced)
        clarify_required = (rd / "clarify.json").is_file()
        solve_ok = not solve_required
        clarify_ok = not clarify_required

        if (rd / "result.json").is_file() and not a.rescore:
            existing = _load(rd / "result.json")
            numeric = _valid_numeric_result(existing)
            terminal_na = existing.get("scoring_status") == "not_applicable"
            if numeric or terminal_na or not solve_required:
                print(f"{tag} 已有终态 result.json，跳过（--rescore 可强制重评）")
                solve_ok = numeric or terminal_na or not solve_required
                n_skipped += 1
            else:
                try:
                    r = score_run(
                        case, rd, a.timeout,
                        reuse_completed=not a.rescore,
                    )
                    print(f"{tag} 重试评分 → {summarize(r)}")
                    solve_ok = _valid_solve_result(r)
                    if solve_ok:
                        n_scored += 1
                    else:
                        print(f"{tag} 评分器返回了非数值失败结果")
                        n_score_failed += 1
                except Exception as e:                   # noqa: BLE001
                    print(f"{tag} 评分异常：{type(e).__name__}: {e}")
                    n_score_failed += 1
        elif status in ("ok", "timeout") and produced:
            try:
                r = score_run(
                    case, rd, a.timeout,
                    reuse_completed=not a.rescore,
                )
                print(f"{tag} status={status} 产物={produced} → {summarize(r)}")
                solve_ok = _valid_solve_result(r)
                if solve_ok:
                    n_scored += 1
                else:
                    print(f"{tag} 评分器返回了非数值失败结果")
                    n_score_failed += 1
            except Exception as e:                       # noqa: BLE001
                # 与 cli.py 同口径：单个 run 评分失败不中断整批
                print(f"{tag} 评分异常（不中断）：{type(e).__name__}: {e}")
                n_score_failed += 1
        else:
            why = "无产物" if not produced else f"status={status}"
            print(f"{tag} 不评求解分（{why}）—— 与 cli.py 门槛一致")
            n_skipped += 1

        # 澄清分独立于产物，单独走
        if (rd / "clarify.json").is_file():
            cached_clarify = _load(rd / "clarify_score.json")
            if not a.rescore and not a.arbitrate and _valid_clarify_result(cached_clarify):
                clarify_ok = True
                n_clarify += 1
                print(f"{tag} 已有完整 clarify_score.json，复用")
            else:
                try:
                    from ..clarify.score import score_clarification
                    cs = score_clarification(
                        case, rd,
                        arbitration_model=(a.arbitration_model if a.arbitrate else None))
                    print(f"{tag} clarify: ask_f1={cs.get('ask_f1')} "
                          f"recall={cs.get('recall')} precision={cs.get('precision')} "
                          f"n_q={cs.get('n_questions')}")
                    # A whole blocker with no usable judge ballot makes recall and
                    # Ask-F1 incomplete.  score_clarification records this instead
                    # of raising, so the entrypoint must not mistake “JSON written”
                    # for a successful judge pass.  Individual ballot failures are
                    # still valid when the remaining multi-model majority exists.
                    clarify_ok = _valid_clarify_result(cs)
                    if clarify_ok:
                        n_clarify += 1
                    else:
                        print(f"{tag} clarify judge 有 {cs.get('judge_errors')} 个考点无有效票")
                        n_clarify_failed += 1
                except Exception as e:                       # noqa: BLE001
                    # judge 是外部 LLM，挂了不该让求解分作废
                    print(f"{tag} clarify 评分失败（不影响求解分）：{type(e).__name__}: {e}")
                    n_clarify_failed += 1

        if not (rd / "result.json").is_file():
            _write_terminal_result(rd, str(status or "unknown"), list(produced),
                                   scoring_failed=solve_required and not solve_ok)
        _write_scoring_marker(
            rd, solve_required=solve_required, solve_ok=solve_ok,
            clarify_required=clarify_required, clarify_ok=clarify_ok)
        if solve_ok and clarify_ok:
            n_terminal += 1

        # 评分入口也要写四态账本；该入口通常处理 no-eval 的远程求解结果，
        # 不能假设状态只会由 harness.cli 写入。
        update_run_statuses(rd)

    print(f"\n[scoring] run={len(runs)} 求解分={n_scored} 跳过={n_skipped} "
          f"澄清分={n_clarify} 求解失败={n_score_failed} "
          f"澄清失败={n_clarify_failed}")
    # 每个 run 都必须得到明确终态；无交付物是可分析的 Agent 结果，不是假 0，
    # 但求解评分或澄清 judge 的异常必须让 Job 失败，以便 batch 准确报告。
    return 0 if n_terminal == len(runs) else 1


if __name__ == "__main__":
    sys.exit(main())
