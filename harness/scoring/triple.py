"""多模型投票抽取：同一份 workspace 用五个不同模型各抽一次，投票定分。

为什么三个**不同**模型而非同一模型跑三遍：同一模型会稳定地犯同一个错——三次
都挑错文件，结果一致但一致地错，一致性反而成了错误的伪证。换三个跨厂商模型，
各自的偏好和盲区不同，**分歧本身就是「这里有歧义」的信号**。

投票规则：
    全部一致    → 采信（unanimous）
    严格多数    → 取多数（majority）
    无严格多数  → 判无效，不取任何一个（no_majority）
    None（抽取崩了）也算一票：崩溃就是该模型没能完成抽取，不该当作弃权让另外
    两票说了算（majority_failed）

运行时用 opencode（extractor_opencode.py），模型通过本地私有 provider 配置连接：
    glm52       = direct/glm-5.2
    qwen38_27b  = chat/qwen3.8-27b
    dsv4f       = chat/deepseek-v4-flash
    gemini36f   = chat/gemini-3.6-flash
    grok45      = chat/grok-4.5
不能用 claude_agent_sdk：ce CLI 只说 Anthropic 协议，glm/kimi 返回 thinking
block 却不带 `signature` 字段，协议校验直接拒（这是协议层硬约束，改 prompt 或
环境变量都绕不开）。opencode 的 Chat provider 走 OpenAI 协议，没有这层校验。

每模型产物写在独立目录 normalized_{tag}，互不覆盖；分歧时可对比三方各自选了
哪个源文件、做了什么转换。分数是浮点，投票前四舍五入到 4 位——低于此精度的
差异是数值噪声而非抽取分歧。
"""

from __future__ import annotations

import json
import os
import re
import signal
import shutil
import subprocess
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

from . import solve
from .conservation_role import check_run

# 五个抽取模型（tag, opencode model ID）。跨厂商模型，避免同源盲区。
_MODELS = [
    ("glm52", "direct/glm-5.2"),
    ("qwen38_27b", "chat/qwen3.8-27b"),
    ("dsv4f", "chat/deepseek-v4-flash"),
    ("gemini36f", "chat/gemini-3.6-flash"),
    ("grok45", "chat/grok-4.5"),
]
_ROUND = 4          # 投票前分数四舍五入位数


def _first_json(text: str) -> dict:
    """从 stdout 取第一个完整 JSON 对象。

    不能用 `text[text.find("{"):text.rfind("}")+1]` 切一刀：抽取器 stdout 里常有
    多个 JSON（_result.json 之后还跟别的输出），那一刀会把两个对象连在一起，
    json.loads 抛 `Extra data`。实测 106/108 份 extraction_notes 全被这条毁掉，
    extraction_status 全成 error。
    """
    text = (text or "").strip()
    if not text:
        return {}
    try:
        obj = json.loads(text)
        return obj if isinstance(obj, dict) else {}
    except json.JSONDecodeError:
        pass
    dec = json.JSONDecoder()
    i = text.find("{")
    while i >= 0:
        try:
            obj, _ = dec.raw_decode(text, i)
            if isinstance(obj, dict):
                return obj
        except json.JSONDecodeError:
            pass
        i = text.find("{", i + 1)
    return {}


def _score(d: dict | None) -> float | None:
    if not d:
        return None
    s = d.get("overall_score")
    return None if s is None else round(float(s), _ROUND)


def _extractor_concurrency() -> int:
    """Bound extractor/evaluator fan-out inside one scoring pod.

    An extractor may run an evaluator which starts several solver processes
    (case 025 starts six).  Five concurrent extractors in a 4-CPU/16-GiB pod
    could therefore fan out to 30 solver processes.  Default to one; roomy
    environments can opt in to more parallelism explicitly.
    """
    raw = os.environ.get("DELIVER_EXTRACTOR_CONCURRENCY", "1")
    try:
        value = int(raw)
    except ValueError:
        value = 1
    return max(1, min(len(_MODELS), value))


def _load_reusable_vote(run_dir: Path, tag: str, model: str) -> dict | None:
    """Load one terminal vote left by an interrupted scoring Job.

    Numeric evaluator results (including zero) and explicit no-artifact
    decisions are stable for an immutable Agent workspace.  Provider,
    extractor and evaluator failures remain retryable.  Explicit --rescore
    bypasses this helper at the caller.
    """
    for path in (
            run_dir / f"score_{tag}.json",
            run_dir / f"extractor_run_{tag}.json"):
        try:
            record = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            continue
        if record.get("tag") != tag or record.get("model") != model:
            continue
        if _score(record) is not None:
            # Old scoring runs occasionally kept a numeric vote even though
            # the extractor admitted that it had generated/recomputed the
            # decision payload from Agent code.  Such a vote is not an
            # immutable extraction result and must not survive a later
            # correctness fix merely because its ledger is present.
            if not _vote_has_source_provenance(run_dir, record):
                continue
            return record
        failure = record.get("extraction_failure") or {}
        if failure.get("category") == "no_artifact":
            return record
    return None


def _majority_is_fixed(completed: list[dict]) -> bool:
    """Whether remaining models can no longer change the five-way outcome."""
    tally: dict[float | None, int] = {}
    # Failed extraction is not a vote.  Counting None here can stop after
    # three failed extractors, preventing the remaining extractors from
    # establishing whether the run is genuinely no-artifact or merely had a
    # scorer/provider failure.
    for record in completed:
        value = _score(record)
        if value is None:
            continue
        tally[value] = tally.get(value, 0) + 1
    return any(count * 2 > len(_MODELS) for count in tally.values())


def vote(scores: list[float | None]) -> tuple[float | None, str]:
    """严格多数投票；``None`` 也算一票，避免抽取失败变成弃权。"""
    if not scores:
        return None, "no_majority"
    tally: dict = {}
    for s in scores:
        tally[s] = tally.get(s, 0) + 1
    top, n = max(tally.items(), key=lambda kv: kv[1])
    if n * 2 <= len(scores):
        return None, "no_majority"
    if top is None:
        return None, "majority_failed"      # 多数票是「抽取崩了」
    return top, "unanimous" if n == len(scores) else "majority"


def _vote_with_index(scores: list[float | None]) -> tuple[float | None, str, int | None]:
    """严格多数投票，并返回代表性产物索引。"""
    if not scores:
        return None, "no_majority", None
    tally: dict[float | None, list[int]] = {}
    for i, score in enumerate(scores):
        tally.setdefault(score, []).append(i)
    top, indices = max(tally.items(), key=lambda kv: len(kv[1]))
    if len(indices) * 2 <= len(scores):
        return None, "no_majority", None
    if top is None:
        return None, "majority_failed", None
    return top, "unanimous" if len(indices) == len(scores) else "majority", indices[0]


def _redact(text: object, limit: int = 2000) -> str:
    """持久化子进程诊断前移除常见凭证格式。"""
    value = str(text or "")
    value = re.sub(r"(?i)bearer\s+[A-Za-z0-9._~+/=-]+", "Bearer [REDACTED]", value)
    value = re.sub(r"sk-[A-Za-z0-9_-]{12,}", "sk-[REDACTED]", value)
    return value[-limit:]


def _scorer_provenance(event: str, details: dict | None = None) -> dict:
    """记录实际修改 result.json 的评分代码版本。"""
    from ..run import _code_fingerprint, _git_commit
    root = Path(__file__).resolve().parents[2]
    return {
        "event": event,
        "scored_at": datetime.now(timezone.utc).isoformat(),
        "scorer_commit": os.environ.get("DELIVER_BENCH_COMMIT") or _git_commit(root),
        "scorer_code_fingerprint": _code_fingerprint(root),
        **(details or {}),
    }


def _extract_once(case_dir: Path, workspace: Path, out: Path, model: str,
                  timeout: int | None = None, audit_output: Path | None = None) -> dict:
    """跑一个模型的抽取器，返回可持久化的单次尝试审计记录。

    状态以**磁盘上的 `_result.json` 为准**，stdout 只作兜底：extractor 把
    `_result.json` 打到 stdout 的同时，opencode 自己也会往 stdout 写东西，
    混在一起时 `_first_json` 可能解析不出来，于是抽取明明 success 却被记成
    error（实测三个模型全中：分数 0.1096 正常出、routes.json 正常写，
    extraction_status 却齐刷刷是 error，排查时会误以为抽取全崩了）。
    """
    exe = Path(__file__).resolve().parents[1] / "scripts" / "extractor_opencode.py"
    cmd = [sys.executable, str(exe),
           "--evaluator", str(case_dir / "tests" / "evaluator.py"),
           "--workspace", str(workspace),
           "--output", str(out),
           "--model", model]
    if audit_output is not None:
        cmd += ["--audit-output", str(audit_output)]

    # score_run 并发启动五个 extractor，而每个 extractor 都会再启动一个
    # ``opencode serve``。OpenCode 默认把 SQLite 状态库放在
    # /root/.local/share/opencode；共享该库会在首次 migration 时竞争建表，导致
    # ``CREATE TABLE data_migration`` 失败。配置目录仍沿用评分 job 的
    # XDG_CONFIG_HOME（它包含 provider 设置），只隔离可写的 data/cache/state。
    #
    # 不放在 normalized_{tag} 下：这些运行时文件不能被 evaluator 当成交付物。
    runtime = out.parent / ".extractor_runtime" / out.name
    data_home = runtime / "data"
    cache_home = runtime / "cache"
    state_home = runtime / "state"
    for directory in (data_home, cache_home, state_home):
        directory.mkdir(parents=True, exist_ok=True)
    env = dict(os.environ)
    env.update({
        "XDG_DATA_HOME": str(data_home),
        "XDG_CACHE_HOME": str(cache_home),
        "XDG_STATE_HOME": str(state_home),
    })
    started = time.monotonic()
    proc: subprocess.Popen | None = None
    try:
        # 必须单独建进程组：extractor_opencode.py 还会启动 opencode serve。
        # subprocess.run(timeout=...) 只 kill 直接子进程，serve 继续持有 stdout/stderr
        # 管道时 communicate 会永远等不到 EOF，表面上就成了“超时也停不下来”。
        proc = subprocess.Popen(
            cmd,
            stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True,
            cwd=str(Path(__file__).resolve().parents[2]),
            env=env,
            start_new_session=True,
        )
        stdout, stderr = proc.communicate(timeout=timeout or 2400)
    except subprocess.TimeoutExpired as exc:
        # 先温和终止整个进程组，短暂等待清理；仍不退出则强杀。不能只 kill
        # proc，否则它拉起的 opencode server 会继续运行并占住管道。
        if proc is not None:
            try:
                os.killpg(proc.pid, signal.SIGTERM)
            except ProcessLookupError:
                pass
            try:
                stdout, stderr = proc.communicate(timeout=5)
            except subprocess.TimeoutExpired:
                try:
                    os.killpg(proc.pid, signal.SIGKILL)
                except ProcessLookupError:
                    pass
                stdout, stderr = proc.communicate()
        else:
            stdout, stderr = exc.stdout or "", exc.stderr or ""
        return {"status": "error", "notes": "extractor timeout", "timed_out": True,
                "returncode": None, "elapsed_s": round(time.monotonic() - started, 3),
                "stdout_tail": _redact(stdout), "stderr_tail": _redact(stderr),
                "status_source": "timeout"}
    except Exception as exc:
        if proc is not None and proc.poll() is None:
            try:
                os.killpg(proc.pid, signal.SIGKILL)
            except ProcessLookupError:
                pass
            proc.communicate()
        return {"status": "error", "notes": f"{type(exc).__name__}: {_redact(exc)}",
                "timed_out": False, "returncode": None,
                "elapsed_s": round(time.monotonic() - started, 3),
                "stdout_tail": "", "stderr_tail": "", "status_source": "exception"}

    record = {"status": "error", "notes": "", "timed_out": False,
              "returncode": proc.returncode,
              "elapsed_s": round(time.monotonic() - started, 3),
              "stdout_tail": _redact(stdout), "stderr_tail": _redact(stderr)}

    # 1) 权威来源：抽取器落盘的 _result.json
    res = out / "_result.json"
    if res.is_file():
        try:
            d = json.loads(res.read_text(encoding="utf-8"))
            if isinstance(d, dict) and d.get("status"):
                record.update({"status": d["status"],
                               # Failure conclusions often come after a long
                               # provenance preamble.  Truncating at 400 chars
                               # lost the final "no deliverable" statement and
                               # misclassified Agent failures as invalid JSON
                               # or extractor timeouts.
                               "notes": (d.get("notes") or d.get("note") or "")[:2000],
                               "status_source": "_result.json"})
                return record
        except (json.JSONDecodeError, OSError):
            pass

    # 2) 兜底：stdout（抽取器崩在写盘前时只剩这个）
    d = _first_json(stdout)
    if d.get("status"):
        record.update({"status": d["status"],
                       "notes": (d.get("notes") or d.get("note") or "")[:2000],
                       "status_source": "stdout"})
        return record
    record.update({"notes": _redact(stderr, 400) or "extractor 未产出 _result.json",
                   "status_source": "fallback"})
    return record


def _score_once(case_dir: Path, out: Path, timeout: int | None = None):
    """对归一化产物重算分数（复用 solve.evaluate）。失败返回 (None, err_info)。"""
    try:
        r = solve.evaluate(case_dir, out, timeout)
        score = r.get("overall_score")
        if score is None:
            return None, {"scoring_failed": "evaluator 无 overall_score", "raw": str(r)[:400]}
        return score, r.get("error_info")
    except Exception as exc:
        return None, {"scoring_failed": f"{type(exc).__name__}: {exc}"}


def _extraction_failure(attempts: list[dict], payload_files: list[str]) -> dict | None:
    """把没有有效 payload 的抽取结果归一成可聚合的失败原因。"""
    if payload_files:
        return None
    last = attempts[-1] if attempts else {}
    note = str(last.get("notes") or "")
    low = note.lower()
    if last.get("failure_category") == "source_provenance_violation":
        # A scorer violating the normalization contract is not evidence that
        # the Agent failed to deliver. Do not count it as a no-artifact vote.
        category = "source_provenance_violation"
    elif any(phrase in low for phrase in (
            "no agent output", "no agent deliverable", "no deliverable",
            "no graded deliverable", "no gradable deliverable",
            "no relevant agent deliverable", "no schedule deliverable",
            "no forecast deliverable", "no predictions deliverable",
            "no predictions output", "no prediction output",
            "no relevant agent output", "no agent-designated deliverable",
            "no final artifact", "designated no artifact",
            "no output files", "no output json", "no valid solution",
            "no solution.json", "no plan.json", "no predictions.csv",
            "no agent-produced solution", "no agent-produced deliverable",
            "no agent-produced artifact", "no agent-produced output",
            "no submission.csv", "no recommendation.json",
            "no solution deliverable", "no solution artifact",
            "no output solution", "no valid output", "nothing to normalize",
            "no materialized", "never materialized", "never serialized",
            "no graded output", "no required deliverable",
            "never produced the graded deliverable",
            "agent never produced", "agent did not produce",
            "was never produced", "were never produced",
            "never generated", "never persisted", "never written")):
        category = "no_artifact"
    elif last.get("timed_out") or "timeout" in low or "timed out" in low:
        category = "extractor_timeout"
    elif re.search(
            r"\b(?:http(?:\s+status)?|status(?:\s+code)?|statuscode|error(?:\s+code)?)"
            r"[\s:=\"']+(?:401|403|429)\b"
            r"|\b(?:401\s+unauthorized|403\s+forbidden|429\s+too many requests)\b"
            r"|\b(?:provider|connection|network|api|authentication)\s+"
            r"(?:error|failed|failure|unavailable|refused|reset|unreachable)\b"
            r"|\b(?:invalid|missing|expired)\s+api[ _]key\b"
            r"|\b(?:rate[ -]limit(?:ed)?|too many requests|unauthorized)\b", low):
        # "forbidden" alone often describes benchmark rules, and numbers
        # such as 403 can be case data. Require actual transport/auth context.
        category = "provider_error"
    elif "json" in low or "decode" in low:
        category = "invalid_json"
    elif any(word in low for word in (
            "schema", "missing", "缺", "column", "attribute", "module")):
        category = "schema_mismatch"
    elif last.get("status") == "error" or last.get("returncode") not in (None, 0):
        # Unknown process failures must remain retryable, not become an
        # affirmative no-artifact vote just because no keyword matched.
        category = "extractor_error"
    else:
        category = "no_artifact"
    return {
        "category": category,
        "message": note[:400] or "extractor 未产出归一化文件",
        "attempts": len(attempts),
        "last_status": last.get("status"),
        "timed_out": bool(last.get("timed_out")),
    }


def _normalized_payload_files(out: Path) -> list[str]:
    """真正供 evaluator 使用的归一化文件，不含配置、日志和状态账本。"""
    ignored = {"hook.log", "_result.json"}
    files = []
    # 不能用 rglob 后再过滤 .opencode：它仍会递归遍历 node_modules，共享文件系统上一次
    # payload 判定就可能耗几十秒。os.walk 在入口剪枝，完全不进入配置树。
    for parent, dirs, names in os.walk(out):
        dirs[:] = [d for d in dirs if d != ".opencode"]
        base = Path(parent)
        for name in names:
            if name in ignored:
                continue
            files.append(str((base / name).relative_to(out)))
    return sorted(files)


def _admits_recomputation(notes: str) -> bool:
    """Recognize positive recomputation admissions in current and legacy ledgers.

    This is an audit guard, not a sandbox. In particular, "I ran the solver"
    and "Running the agent's finished solver" must not evade the gate simply
    because they lack the prefix "re-". Negation is local to a clause; a
    preceding "did not ...; but ..." must not excuse a later positive action.
    """
    target = (
        r"(?:the\s+)?(?:agent(?:['’]s)?\s+)?"
        r"(?:(?:finished|saved|existing|completed|original)\s+)*"
        r"(?:solver|model|pipeline|solution)\b")
    patterns = (
        r"\bgenerated\s+(?:the\s+)?(?:predictions?|submission|solution|plan)\b",
        r"\bre-?computed\s+(?:the\s+)?(?:predictions?|submission|solution|plan)\b",
        r"\b(?:ran|running|re-?ran|re-?running|executed|executing)\s+" + target,
    )
    for pattern in patterns:
        for match in re.finditer(pattern, notes, flags=re.IGNORECASE):
            prefix = notes[max(0, match.start() - 160):match.start()].lower()
            clause = re.split(r"[.!?;\n]|\b(?:but|however|then)\b", prefix)[-1]
            if re.search(r"\b(?:not|never|without|didn't|didn’t)\b", clause):
                continue
            # Reporting what the original Agent did is not an admission of
            # extractor execution. First-person admissions remain rejected.
            if re.search(r"\b(?:the\s+)?agent\s+(?:had\s+|already\s+)*$", clause):
                continue
            return True
    return False


def _payload_has_source_provenance(out: Path, workspace: Path | None = None) -> bool:
    """Reject normalized files that an extractor invented from no Agent output.

    New extractors explicitly write ``files_found``.  An empty list means they
    found no source artifact, even if they then manufactured an empty schema
    such as ``{"days": []}``.  Missing metadata remains accepted for legacy
    extractors and for crash recovery where a real payload reached disk before
    ``_result.json`` was written.
    """
    result_path = out / "_result.json"
    result: dict = {}
    if result_path.is_file():
        try:
            result = json.loads(result_path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            result = {}
        found = result.get("files_found")
        if isinstance(found, list) and not found:
            return False

        # Normalization may rename/reshape an existing Agent artifact, but it
        # may never execute the Agent's solver or model to create decisions
        # that did not exist when the Agent stopped.  This happened for a
        # timed-out forecasting run: one extractor explicitly reported that
        # it "Generated predictions.csv" from solution.py/model checkpoints,
        # producing a numeric score for an output the Agent never delivered.
        # Reject explicit positive admissions of recomputation.  Keep the
        # phrases narrow so statements such as "did not generate" remain
        # valid audit notes.
        if _admits_recomputation(str(result.get("notes") or result.get("note") or "")):
            return False

    # Executable submissions are code, not a tabular representation. Renaming
    # an Agent source file to the required filename is safe; rewriting its CLI,
    # schema, algorithm, or decisions is not. Require every normalized Python
    # file to byte-match some Python source in the immutable Agent workspace.
    if workspace is not None:
        normalized_python = [
            path for path in out.rglob("*.py") if ".opencode" not in path.parts]
        if normalized_python:
            source_bytes = {
                path.read_bytes() for path in workspace.rglob("*.py")
                if ".opencode" not in path.parts and "data" not in path.parts
            }
            if not source_bytes or any(
                    path.read_bytes() not in source_bytes
                    for path in normalized_python):
                return False
    return True


def _vote_has_source_provenance(run_dir: Path, record: dict) -> bool:
    """Check cached notes even if an interrupted retry removed the payload."""
    notes = [record.get("extraction_notes", "")]
    # Earlier attempts may have been rejected and cleared before a clean
    # retry. Only the attempt that produced the cached vote is authoritative.
    for attempt in (record.get("attempts") or [])[-1:]:
        if isinstance(attempt, dict):
            if attempt.get("failure_category") == "source_provenance_violation":
                return False
            notes.append(attempt.get("notes", ""))
    if any(_admits_recomputation(str(note or "")) for note in notes):
        return False
    return _payload_has_source_provenance(
        run_dir / f"normalized_{record['tag']}", run_dir / "workspace")


def one_model(case_dir: Path, run_dir: Path, tag: str, model: str,
              timeout: int | None = None) -> dict:
    """一个模型的抽取 + 守恒 + 评估。产物落 run_dir/normalized_{tag}。"""
    workspace = run_dir / "workspace"
    out = run_dir / f"normalized_{tag}"
    # ``--rescore`` 会在已有 run 上重新调用本函数。若保留上一次的
    # normalized_<tag>，本次 extractor 在启动阶段失败时，下面的 payload 检查会
    # 把旧 routes.json 当成本次的成功产物，造成失败模型被错误纳入投票。每次调用
    # 都从空目录开始；需要保留历史时由 reextract_model 显式归档到
    # reextract_backups，而不是让旧 payload 隐式参与本次评分。
    if out.exists():
        shutil.rmtree(out)
    out.mkdir(parents=True, exist_ok=True)

    # 空产物重试一次：瞬时失败 vs 真坏产物，重试一次能区分。
    ext_status, ext_notes = "error", ""
    attempts: list[dict] = []
    for attempt_no in range(1, 3):
        audit_path = run_dir / f"extractor_session_{tag}_attempt{attempt_no}.json"
        attempt = _extract_once(case_dir, workspace, out, model, timeout, audit_path)
        attempts.append(attempt)
        ext_status, ext_notes = attempt["status"], attempt.get("notes", "")
        payload = _normalized_payload_files(out)
        if payload and _payload_has_source_provenance(out, workspace):
            break
        if payload:
            # Keep this structured reason even after removing a contaminated
            # payload; it must neither be evaluated nor count as no-artifact.
            attempt["failure_category"] = "source_provenance_violation"
            # Do not let an invented empty-schema payload survive into the next
            # attempt or the evaluator.
            for item in list(out.iterdir()):
                if item.name == ".opencode":
                    continue
                shutil.rmtree(item) if item.is_dir() else item.unlink()

    # 抽取器崩溃且没有任何 normalized payload 时是「无有效票」，不是 Agent 得 0。
    # 只有 payload 已落盘时才交给 evaluator；即使漏写 _result.json，确定性 evaluator
    # 仍可恢复该票，同时保留 extraction_status=error 供审计。
    payload_files = (_normalized_payload_files(out)
                     if _payload_has_source_provenance(out, workspace) else [])
    if payload_files:
        score, err_info = _score_once(case_dir, out, timeout)
    else:
        score, err_info = None, {
            "extraction_failed": ext_notes or "extractor 未产出归一化文件",
        }
    extraction_failure = _extraction_failure(attempts, payload_files)
    scoring_failure = None
    if payload_files and score is None:
        message = (err_info.get("scoring_failed") if isinstance(err_info, dict)
                   else str(err_info or "evaluator 未返回分数"))
        scoring_failure = {
            "category": "evaluator_invalid",
            "message": str(message)[:400],
        }

    # 守恒校验：只记录不改分——先积累数据看准确率，再决定是否据此判无效
    conservation = None
    try:
        conservation = check_run(case_dir, run_dir, tag)
    except Exception as exc:
        conservation = {"ok": None, "error": f"{type(exc).__name__}: {exc}"}

    result = {
        "model": model, "tag": tag,
        "overall_score": score,
        "extraction_status": ext_status,
        "extraction_notes": ext_notes,
        "vote_eligible": score is not None,
        "error_info": err_info,
        "extraction_failure": extraction_failure,
        "scoring_failure": scoring_failure,
        "conservation": conservation,
        "files_normalized": sorted(q.name for q in out.glob("*")
                                   if q.is_file() and q.name != "hook.log"),
        "payload_files": payload_files,
        "attempts": attempts,
    }
    (run_dir / f"extractor_run_{tag}.json").write_text(
        json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
    return result


def _recover_orphaned_vote(case_dir: Path, run_dir: Path, tag: str, model: str,
                           timeout: int | None = None) -> dict | None:
    """Evaluate an exact Agent-derived payload left by an interrupted Pod.

    A Pod may disappear after normalized files are written but before either
    per-model ledger is persisted. Recovery is strict: executable payloads must
    byte-match Agent workspace sources, so extractor-authored code is rejected.
    """
    out = run_dir / f"normalized_{tag}"
    workspace = run_dir / "workspace"
    payload_files = _normalized_payload_files(out)
    if (not payload_files
            or not _payload_has_source_provenance(out, workspace)):
        return None
    try:
        evaluation = solve.evaluate(case_dir, out, timeout)
        score = evaluation.get("overall_score")
        if score is None:
            return None
    except Exception:
        return None
    try:
        conservation = check_run(case_dir, run_dir, tag)
    except Exception as exc:
        conservation = {"ok": None, "error": f"{type(exc).__name__}: {exc}"}
    record = {
        "model": model,
        "tag": tag,
        "overall_score": score,
        "extraction_status": "recovered",
        "extraction_notes": "recovered exact payload from interrupted scoring Pod",
        "vote_eligible": True,
        "error_info": evaluation.get("error_info"),
        "evaluation": evaluation,
        "extraction_failure": None,
        "scoring_failure": None,
        "conservation": conservation,
        "files_normalized": sorted(
            path.name for path in out.glob("*") if path.is_file()),
        "payload_files": payload_files,
        "attempts": [],
    }
    for path in (run_dir / f"extractor_run_{tag}.json",
                 run_dir / f"score_{tag}.json"):
        path.write_text(json.dumps(record, ensure_ascii=False, indent=2),
                        encoding="utf-8")
    return record


def _merge_vote_result(case_dir: Path, run_dir: Path, runs: list[dict],
                       timeout: int | None = None, *, scoring_event: str = "triple_score",
                       scoring_details: dict | None = None) -> dict:
    """用各份单模型账本重算投票并合并进 result.json，不调用抽取模型。"""
    # The single-model reextract path can load peer ledgers directly. Enforce
    # the same guard here so it cannot bypass score_run's cache validation.
    runs = [
        {**record, "overall_score": None, "vote_eligible": False,
         "payload_files": [], "extraction_failure": {
             "category": "source_provenance_violation",
             "message": "cached vote rejected: extractor violated source provenance"}}
        if _score(record) is not None and not _vote_has_source_provenance(run_dir, record)
        else record
        for record in runs
    ]
    scores = [_score(r) for r in runs]
    final, how, selected_idx = _vote_with_index(scores)
    # A reusable ledger can outlive its normalized directory (for example, a
    # later interrupted retry may have cleared that tag before extraction
    # failed).  Prefer a majority peer whose payload still exists so the final
    # detailed evaluator pass never follows a stale representative index.
    if final is not None:
        for i, record in enumerate(runs):
            if (_score(record) == final
                    and _normalized_payload_files(
                        run_dir / f"normalized_{record['tag']}")):
                selected_idx = i
                break
    no_artifact_votes = sum(
        1 for r in runs
        if (r.get("extraction_failure") or {}).get("category") == "no_artifact")
    agent_no_artifact = (final is None and no_artifact_votes * 2 > len(runs))

    # 采信分数写回 result.json。原件（agent run 的 state 信息，如 status/elapsed）
    # 保留：先把已有内容读进来，叠加投票结果，不覆盖 agent 运行期字段。
    # 首评前先备份一次原件，供对比单模型 vs 三模型分数。
    res_path = run_dir / "result.json"
    if final is not None:
        bak = run_dir / "result.json.pre_triple"
        if not bak.exists() and res_path.exists():
            import shutil
            shutil.copy2(res_path, bak)
    try:
        cur = json.loads(res_path.read_text(encoding="utf-8")) if res_path.exists() else {}
    except json.JSONDecodeError:
        cur = {}

    provenance = _scorer_provenance(scoring_event, scoring_details)
    history = cur.get("scoring_history")
    if not isinstance(history, list):
        history = []

    result = {
        "case": case_dir.name,
        "scores": dict((r["tag"], _score(r)) for r in runs),
        "final_score": final,
        "vote": how,
        "triple_vote": how,
        "scoring_status": ("scored" if final is not None else
                           "not_applicable" if agent_no_artifact else "failed"),
        "overall_score": final,      # 采信的多模型票分数，覆盖单模型结果
        "combined_score": final,     # 兼容只读 combined_score 的下游
        # 分歧时保留各方文件，用于定位该 case 的歧义
        "disagreement": None if how == "unanimous" else {
            # ``_result.json`` is extractor bookkeeping, not a gradeable
            # payload.  Report payload_files here so a failed extractor is not
            # mistakenly presented as having produced a normalized solution.
            r["tag"]: {"score": _score(r),
                       "files": r.get("payload_files") or []}
            for r in runs
        },
        "conservation_by_model": {r["tag"]: r.get("conservation") for r in runs},
        "selected_model": runs[selected_idx]["model"] if selected_idx is not None else None,
        "selected_tag": runs[selected_idx]["tag"] if selected_idx is not None else None,
        "scored_at": provenance["scored_at"],
        "scorer_commit": provenance["scorer_commit"],
        "scorer_code_fingerprint": provenance["scorer_code_fingerprint"],
        "scoring_history": [*history, provenance],
    }

    # evaluator 的 validity/quality/error_info 取自「分数最高且抽取成功」的那个
    # 模型的产物——避免重复评估：_score_once 已拿分数，这里只补 eval 的其余字段。
    selected = runs[selected_idx] if selected_idx is not None else None
    if selected is not None and final is not None:
        # overall_score、validity_score、quality_score、error_info 全部来自同一个
        # 投票胜出的 normalized 产物，避免字段之间指向不同模型的结果。
        cached_evaluation = selected.get("evaluation")
        ev = (cached_evaluation if isinstance(cached_evaluation, dict)
              and _score(cached_evaluation) == final else
              solve.evaluate(case_dir, run_dir / f"normalized_{selected['tag']}", timeout))
        for k, v in ev.items():
            if k not in ("overall_score", "final_score", "vote"):
                result[k] = v
    elif agent_no_artifact:
        result.update({
            "artifact_status": "none",
            "validity_score": None,
            "quality_score": None,
            "error_info": {
                "category": "artifact_missing",
                "message": (f"{no_artifact_votes}/{len(runs)} extractors independently "
                            "found no Agent deliverable"),
            },
        })
    else:
        result.update({"validity_score": None, "quality_score": None,
                       "error_info": {"scoring_failed": "no_majority"}})

    merged = {**cur, **result}      # 旧字段保底，投票结果优先
    res_path.write_text(json.dumps(merged, ensure_ascii=False, indent=2), encoding="utf-8")
    return merged


def score_run(case_dir: Path, run_dir: Path, timeout: int | None = None,
              *, reuse_completed: bool = True) -> dict:
    """对单次 run 做多模型投票抽取+评估，结果写 run_dir/result.json。

    与 solve.score_run 的差异：solve 用单模型 extractor_agent（快、依赖少）；
    这里用 opencode 多模型投票（慢但稳，抗单模型盲区）。
    """
    workspace = run_dir / "workspace"
    if not workspace.is_dir():
        return {"case": case_dir.name, "error": "无 workspace 目录"}

    # Reuse terminal per-model ledgers left by an interrupted scoring Job.  A
    # provider/evaluator failure is deliberately not reusable and is retried.
    completed: dict[str, dict] = {}
    if reuse_completed:
        for tag, model in _MODELS:
            record = _load_reusable_vote(run_dir, tag, model)
            if record is not None:
                completed[tag] = record
        for tag, model in _MODELS:
            if tag in completed or _majority_is_fixed(list(completed.values())):
                continue
            record = _recover_orphaned_vote(
                case_dir, run_dir, tag, model, timeout)
            if record is not None:
                completed[tag] = record

    # Extract missing votes in bounded batches.  Evaluators may themselves
    # fork several solver processes, so the default concurrency is one.  Stop
    # once three equal votes make the five-way result mathematically fixed.
    from concurrent.futures import ThreadPoolExecutor
    pending = [spec for spec in _MODELS if spec[0] not in completed]
    width = _extractor_concurrency()
    while pending and not _majority_is_fixed(list(completed.values())):
        batch, pending = pending[:width], pending[width:]
        with ThreadPoolExecutor(max_workers=len(batch),
                                thread_name_prefix="extractor") as pool:
            fresh = list(pool.map(
                lambda spec: one_model(
                    case_dir, run_dir, spec[0], spec[1], timeout),
                batch,
            ))
        for record in fresh:
            tag = record["tag"]
            completed[tag] = record
            # Persist immediately so a later Pod eviction can resume here.
            (run_dir / f"score_{tag}.json").write_text(
                json.dumps(record, ensure_ascii=False, indent=2),
                encoding="utf-8")

    runs = []
    for tag, model in _MODELS:
        record = completed.get(tag)
        if record is None:
            record = {
                "tag": tag,
                "model": model,
                "overall_score": None,
                "payload_files": [],
                "files_normalized": [],
                "extraction_failure": {
                    "category": "not_needed_after_majority",
                    "message": "strict majority already fixed",
                },
                "conservation": None,
            }
        runs.append(record)
        # Keep all five ordered ledgers explicit, including models skipped after
        # majority, so audit tools never confuse an intentional skip with loss.
        (run_dir / f"score_{tag}.json").write_text(
            json.dumps(record, ensure_ascii=False, indent=2), encoding="utf-8")

    return _merge_vote_result(case_dir, run_dir, runs, timeout)


def extractor_needs_rerun(run_dir: Path, tag: str) -> bool:
    """该抽取票是否没有任何 evaluator 可用 payload。

    旧实现会把“没有 payload”送入 evaluator 并伪装成 0 分，不能仅通过
    score_{tag}.json 的分数判断；磁盘上是否有归一化交付文件才是可靠判据。
    """
    return not _normalized_payload_files(run_dir / f"normalized_{tag}")


def reextract_model(case_dir: Path, run_dir: Path, tag: str,
                    timeout: int | None = None) -> dict:
    """只重跑一个抽取模型，再复用其它票重算多模型投票。

    重抽前把该模型的旧 normalized 目录移动到带时间戳的 reextract_backups，
    并复制分数、审计账本和旧 result.json。账本原件保留到新结果完成后再覆盖，
    因此 Job 即使被强制终止也不会留下 score 文件缺口；移动 normalized 则确保
    旧 payload 不会被误认成本次新产物。
    """
    models = dict(_MODELS)
    if tag not in models:
        raise ValueError(f"未知 extractor tag: {tag}；可选：{', '.join(models)}")
    if not (run_dir / "workspace").is_dir():
        raise FileNotFoundError(f"{run_dir} 下没有 workspace")

    # 在付费调用前确认另外两票齐全，避免抽完后才发现无法重新投票。
    loaded: dict[str, dict] = {}
    for peer_tag, _ in _MODELS:
        if peer_tag == tag:
            continue
        score_path = run_dir / f"score_{peer_tag}.json"
        try:
            value = json.loads(score_path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError) as exc:
            raise ValueError(f"无法读取已有票 {score_path}: {exc}") from exc
        if not isinstance(value, dict):
            raise ValueError(f"已有票不是 JSON object: {score_path}")
        loaded[peer_tag] = value

    stamp = time.strftime("%Y%m%d-%H%M%S", time.gmtime())
    backup = run_dir / "reextract_backups" / f"{tag}-{stamp}-{time.time_ns() % 1_000_000_000:09d}"
    backup.mkdir(parents=True, exist_ok=False)
    old_normalized = run_dir / f"normalized_{tag}"
    if old_normalized.exists():
        shutil.move(str(old_normalized), str(backup / old_normalized.name))
    old_ledgers = [
        run_dir / f"score_{tag}.json",
        run_dir / f"extractor_run_{tag}.json",
        *sorted(run_dir.glob(f"extractor_session_{tag}_attempt*.json")),
    ]
    for path in old_ledgers:
        if path.is_file():
            shutil.copy2(path, backup / path.name)
    if (run_dir / "result.json").is_file():
        shutil.copy2(run_dir / "result.json", backup / "result.json")

    fresh = one_model(case_dir, run_dir, tag, models[tag], timeout)
    (run_dir / f"score_{tag}.json").write_text(
        json.dumps(fresh, ensure_ascii=False, indent=2), encoding="utf-8")
    loaded[tag] = fresh
    ordered = [loaded[t] for t, _ in _MODELS]
    merged = _merge_vote_result(
        case_dir, run_dir, ordered, timeout,
        scoring_event="reextract",
        scoring_details={"extractor_tag": tag, "extractor_model": models[tag],
                         "backup": str(backup)},
    )
    # 单模型账本也保留同一条 provenance，不能只在最终 result.json 留痕。
    fresh["scoring_provenance"] = merged["scoring_history"][-1]
    for ledger in (run_dir / f"score_{tag}.json",
                   run_dir / f"extractor_run_{tag}.json"):
        ledger.write_text(json.dumps(fresh, ensure_ascii=False, indent=2), encoding="utf-8")
    return {
        "tag": tag,
        "model": models[tag],
        "score": _score(fresh),
        "extraction_status": fresh.get("extraction_status"),
        "payload_files": fresh.get("payload_files", []),
        "backup": str(backup),
        "scoring_provenance": fresh["scoring_provenance"],
        "vote": merged.get("vote"),
        "overall_score": merged.get("overall_score"),
    }


def summarize(result: dict) -> str:
    if result.get("error"):
        return f"评分失败: {result['error']}"
    v = result.get("final_score", result.get("overall_score"))
    how = result.get("vote", result.get("triple_vote"))
    parts = [f"triple={v} ({how})"]
    if result.get("validity_score") is not None:
        parts.append(f"validity={result.get('validity_score')}")
    if result.get("quality_score") is not None:
        parts.append(f"quality={result.get('quality_score')}")
    return "  ".join(parts)
