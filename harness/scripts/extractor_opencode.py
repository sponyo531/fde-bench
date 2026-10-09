"""opencode 版 extractor：与 extractor_agent.py 同一份 prompt，只换运行时。

为什么不继续用 claude_agent_sdk：它底层是 Claude Code CLI，只说 Anthropic 协议。
glm-5.2 / kimi-k3 会返回 thinking block 但不带 `signature` 字段，CLI 按协议校验
直接拒掉（Missing required field in assistant message: 'signature'），这是协议层的
硬约束，改 prompt 或环境变量都绕不过。实测同样的 SDK 代码：
    ds/deepseek-v4-pro   ✓        siliconflow/glm-5.2  ✗ signature
    glm-5                ✓        kimi-k3              ✗ signature

opencode 的 Chat provider 走 @ai-sdk/openai-compatible（OpenAI 协议），
没有这层校验，而且 839 个 run 的求解 agent 本来就跑在它上面——这几个模型在这条
链路上早已验证可用。

    python extractor_opencode.py --evaluator <p> --workspace <p> --output <p> --model direct/glm-5.2
"""

from __future__ import annotations

import argparse
import asyncio
import json
import os
import re
import shutil
import sys
from datetime import datetime, timezone
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from harness.backends.base import RunnerConfig          # noqa: E402
from harness.backends.registry import runner_from_config  # noqa: E402

REPO = Path(__file__).resolve().parents[2]

# 抽取器必须使用最小配置：只含 provider/model，不得继承 webagent 的
# 项目专用 skills 或插件。此前为了复用 provider
# provider 直接复制整个 webagent/opencode，导致纯文件转换任务被业务 Agent 的
# 强制澄清/规划/委派流程干扰，Kimi 尤其容易以 UnknownError 中止。
EXTRACTOR_OPENCODE = REPO / "harness" / "scoring" / "extractor_opencode"


def _normalized_payload_files(output_dir: Path) -> list[str]:
    """Return payload files written by the extractor, excluding runtime metadata.

    The model is responsible for the actual normalized deliverable, but not for
    reliably writing the bookkeeping ledger.  Keep this scan in sync with the
    scorer's payload detection: do not recurse into OpenCode's node_modules.
    """
    ignored = {"hook.log", "_result.json"}
    files: list[str] = []
    for parent, dirs, names in os.walk(output_dir):
        dirs[:] = [d for d in dirs if d != ".opencode"]
        base = Path(parent)
        for name in names:
            if name in ignored:
                continue
            files.append(str((base / name).relative_to(output_dir)))
    return sorted(files)


def _write_fallback_result(output_dir: Path, payload_files: list[str]) -> dict:
    """Write a truthful ledger when the model omitted ``_result.json``.

    ``partial`` is intentional: the payload can still be evaluated, but the
    model violated the extractor protocol.  The write is atomic so a scorer
    never observes a half-written JSON document.
    """
    result = {
        "status": "partial",
        "notes": (
            "模型已写入归一化产物，但未写出 _result.json；"
            "由 runner 自动补写，未修改决策内容。"
        ),
        "files_found": [],
        "files_normalized": payload_files,
        "protocol_violation": "missing_result_metadata",
    }
    target = output_dir / "_result.json"
    tmp = output_dir / ".result.json.tmp"
    tmp.write_text(json.dumps(result, ensure_ascii=False, indent=2) + "\n",
                   encoding="utf-8")
    os.replace(tmp, target)
    return result


def _prepare_config(workspace: Path, model: str) -> tuple[Path, str]:
    """Build an isolated config from a user-supplied provider registry."""
    runtime = workspace.parent / ".extractor_config" / re.sub(r"[^A-Za-z0-9_.-]", "_", model)
    if runtime.exists():
        shutil.rmtree(runtime)
    shutil.copytree(EXTRACTOR_OPENCODE, runtime,
                    ignore=shutil.ignore_patterns("node_modules", "__pycache__"))
    from harness.installer import _load_jsonc
    source = os.environ.get("FDE_EXTRACTOR_OPENCODE_CONFIG") or os.environ.get("FDE_OPENCODE_CONFIG")
    if source:
        source_path = Path(source).expanduser()
    else:
        source_path = Path(os.environ.get("XDG_CONFIG_HOME", Path.home() / ".config")) / "opencode" / "opencode.json"
    if not source_path.is_file():
        raise RuntimeError(f"Extractor provider config missing: {source_path}")
    private = _load_jsonc(source_path.read_text(encoding="utf-8"))
    provider, sep, model_name = model.partition("/")
    providers = private.get("provider") or {}
    if not sep or model_name not in (providers.get(provider, {}).get("models") or {}):
        raise RuntimeError(f"Extractor model {model} is absent from {source_path}")
    cfg = {"$schema": "https://opencode.ai/config.json",
           "provider": {provider: providers[provider]}, "model": model}
    cfg_path = runtime / "opencode.jsonc"
    src_modules = EXTRACTOR_OPENCODE / "node_modules"
    if (src_modules / "@ai-sdk" / "openai").is_dir():
        shutil.copytree(src_modules, runtime / "node_modules", symlinks=True)
    cfg_path.write_text(json.dumps(cfg, indent=2, ensure_ascii=False), encoding="utf-8")
    return runtime, model


def _safe_text(value: object, limit: int = 2000) -> str:
    """诊断日志脱敏；不把网关凭证带进实验产物。"""
    text = str(value or "")
    text = re.sub(r"(?i)bearer\s+[A-Za-z0-9._~+/=-]+", "Bearer [REDACTED]", text)
    text = re.sub(r"sk-[A-Za-z0-9_-]{12,}", "sk-[REDACTED]", text)
    return text[-limit:]


def build_prompt(evaluator: Path, workspace: Path, output_dir: Path, data_dir: Path) -> str:
    """与 extractor_agent.py 的 prompt 保持一致，仅去掉 CC 专属的工具名说明。

    schema 优先、prose 兜底：submission_schema.json 是从 evaluator 源码自动导出的
    输入契约（字段、别名、每个字段是 identifier/grouping_key/decision），比人手写的
    EXTRACTOR_SPEC 全——手写 prose 漏过 `tasks`/`orders` 这类别名，合法产物会被判无效。
    """
    schema = evaluator.parent / "submission_schema.json"
    spec = ""
    if schema.is_file():
        spec = f"## Target schema (authoritative)\n\n```json\n{schema.read_text(encoding='utf-8')}\n```"
    else:
        src = evaluator.read_text(encoding="utf-8", errors="replace")
        import re
        m = re.search(r'"""(.*?)"""', src, re.DOTALL)
        doc = m.group(1) if m else ""
        sm = re.search(r"(EXTRACTOR_SPEC:.*)", doc, re.DOTALL)
        spec = f"## Target schema\n\n{sm.group(1).strip() if sm else '(infer from evaluator source)'}"

    # The 025 evaluator launches six expensive solver subprocesses. The
    # scoring stage runs it after extraction, so do not run it inside the
    # extractor while the OpenCode server is also resident in the same Pod.
    heavy_code_case = evaluator.parent.parent.name == "025_vessel_stowage_planning_clean"
    verification = (f"""## Heavyweight executable case

Do not run `{evaluator}` or the Agent solver. Copy the selected Agent Python
source byte-for-byte to the required output filename. Do not rewrite its CLI,
output shape, algorithm, or decisions: incompatibility is an Agent result.
The scoring stage evaluates it after you exit.""" if heavy_code_case else
        f"""## Verify by running the evaluator (required)

A schema cannot enumerate every shape an agent might produce, so do not guess whether your
normalization is right — check it:

```bash
python3 {evaluator} --data-dir {data_dir} --submission-dir {output_dir}
```

React to *structural* failures only:

| What you see | What it means | What to do |
|---|---|---|
| `ModuleNotFoundError` | a module the entry file imports is missing | copy that module in too |
| `AttributeError: must define <fn>` | wrong entry file | try the next candidate |
| `缺 <file>` / `KeyError` / missing column | filename or column mapping is off | fix the mapping |
| valid JSON with any score, **including 0** | the evaluator ran to completion | **stop — you are done** |""")

    return f"""You are a data extractor for a benchmark evaluation pipeline.

## Your task

1. **Read `_agent_summary.md` first** (if present in the workspace). It tells you which file
   the agent considers its final answer. The agent's own designation decides which artifact
   gets graded — you do not get to pick a different (better-scoring) one.
2. Read the target schema below to learn what files and fields the evaluator accepts.
3. Find the agent's output in the workspace. Filenames and column names will differ.
4. Normalize it to the schema and write the result to the OUTPUT directory.
5. **If the deliverable is executable code** (the schema asks for a `.py` defining a specific
   function or class): the evaluator imports it from the OUTPUT directory, so copy the entry
   file under the required name AND every workspace-local module it imports, directly or
   transitively. Copying only the entry file leaves it orphaned — ModuleNotFoundError, and a
   working solution scores 0.
6. Write `_result.json` to the OUTPUT directory as your final action.

{spec}

{verification}

**Stop as soon as the evaluator returns a score.** A 0 is a legitimate final answer: the
agent's solution ran and did not meet the constraints. Never edit values, add rows, drop
rows, or swap in a different solution file to raise the number. You normalize
*representation* — filenames, column names, layout, encoding. The decisions are the agent's
and must survive unchanged. If no score can be obtained, the artifact is wrong; say so and
stop rather than making something up.

**The three ways this rule actually gets broken.** Each was observed in production; each
looked defensible to the extractor that did it. All three are forbidden.

1. **Nudging values inside a stated tolerance.** Do not adjust a number "within the ±10mm
   length tolerance" (or any other slack the evaluator grants) so a constraint that was
   failing now passes. One run moved `sub_i_long` by a few mm — every individual edit was
   inside tolerance — and walked 26/35 specs up to 35/35, turning 0.0 into 0.903. Tolerance
   exists to absorb *the agent's* rounding, not to give you a search space. If the agent's
   numbers land outside a limit, that is the answer: report the 0.
2. **Writing an adapter when the agent's interface does not match the evaluator's.** When a
   deliverable is code and its signature or data model disagrees with what the evaluator
   passes in (e.g. the policy expects a custom object, the evaluator hands it a dict), do
   **not** author a shim, a wrapper class, or a re-implementation. One run emitted 669 lines
   of its own code around a 13.5 KB agent file and scored 0.858 where faithful copies scored
   0; that is the extractor's solution being graded, not the agent's. Copy the agent's file
   as-is, let it fail, and record the mismatch in `notes`.
3. **Dropping rows that "look wrong".** Removing rows the agent emitted — even ones that
   seem like annotations, placeholders, or infeasible leftovers — changes the answer. One run
   dropped 3 feasible rows past the 4 obvious annotation rows (515 → 512) and converted 0.0
   into 0.854. If you drop or add **any** row, state the before/after counts and the reason
   in `notes`. Silent row-count changes are the single hardest form of tampering to detect
   downstream, which is exactly why they must be disclosed.

4. **"Fixing" an identifier the agent spelled wrong.** Whether you may repair a bad name
   depends on one question: **does the task specify the naming?**
   - **Task is silent on naming** (e.g. the agent labels a plate `板1` where the evaluator
     keys on `P001`): map it, provided the target is **unique** in the reference data. This
     is ordinary identifier normalization. If the mapping is ambiguous or has no match,
     do not guess — leave it and say so.
   - **Task states the naming rule** (e.g. `information.md` says feature names must be
     real columns of the training data, in `{{trait}}_D{{day}}` form): a name that breaks the
     rule is a **substantive error by the agent**, not a formatting slip. Copy it as-is and
     let it score 0. One run had the agent emit `Anth_Variance_RC_norm_D1` when the data
     only contains `Anth_Variance`; two extractors stripped `_RC_norm` and lifted 0.0 to
     0.9717 — that erased a real violation of a stated requirement.
   Check `information.md` / `instruction.md` for a naming rule **before** you repair any
   name. Also check the schema's `role`: repairing is only ever in scope for `identifier`
   fields, never for `decision` fields — those are the answer itself.

5. **Using the evaluator's own formula to generate a value the evaluator checks.** When the
   schema documents how a field relates to others (`sub_i_long must equal
   mother_cut_weight / (mother_width × 7.65) × 10000`), that formula is there so you can
   understand the field — **not so you can compute it**. Filling the field from the formula
   makes the consistency check pass by construction: the evaluator ends up verifying its own
   arithmetic instead of the agent's. One run did exactly this — the agent had used density
   7.85, the extractor recomputed the whole length column with the documented 7.65, every
   check passed, and 0.0 became 0.871. Its `notes` argued "length is a derived value, no
   decision values were altered," which was *factually true* and still wrong: a derived value
   the evaluator validates is load-bearing.
   The test is not "is this field a decision?" — it is **"does the evaluator look at it?"**
   If yes, the value must come from the agent, even when you can compute a better one, and
   even when the agent's is provably inconsistent. An inconsistency you found is a finding to
   report in `notes`, not a defect to repair.

## When the agent's designated file is incomplete

Rule 1 says the agent's own designation decides what gets graded. That assumes the
designated artifact exists and is coherent. When it is **not**:

- **The designation names a state the agent never finished.** One `_agent_summary.md` was a
  truncated half-sentence — "v4_sector wins … Let me adopt it and re-run stage 2." — and the
  v4 re-run died mid-way: 3 of 5 zones written, the rest left over from v1, mutually
  inconsistent. Two extractors honoured the designation, emitted the 3-zone state and scored
  0; one fell back to the complete, self-validated v1 (11/11 checks) and scored 0.988. The
  fallback was right. **If the designated artifact is incomplete or internally contradictory,
  fall back to the most recent complete state that passes the agent's own validation**, set
  `status` to `"partial"`, and say in `notes` which state you used and why you left the
  designated one. This is not "picking the better-scoring file" — it is picking the only
  file that is a coherent answer.
- **Extra rows beyond the graded set.** Rows the agent produced for keys that are not in the
  evaluation set should be dropped, with the before/after counts disclosed. That is different
  from dropping rows the agent *did* produce for graded keys, which is forbidden.

## Verify order-bearing fields before you write

When a field encodes a sequence (a visit order, a permutation, a concatenated route), its
order **is** the decision. Reconstructing it from a grouped source is easy to get subtly
wrong: one extractor emitted every within-trip order correctly but concatenated the trips
themselves in a scrambled order, across all 25 routes — the stop sets matched, the counts
matched, only the seams were wrong, and the score moved by 0.0014. Its own `notes` claimed
the sequence was "sorted by (trip_id, order_within_trip)"; the file was not.

Before writing an order-bearing field, re-derive it from the source a second way and compare
element by element. If the schema gives you a companion structure (a `trips` array with per-
trip stop counts, say), slice your sequence by those counts and check each slice against the
source group it should match.

Rule of thumb: if an edit would change the score, it is not normalization. Reformatting,
renaming, reshaping, re-encoding never change a score — those are yours to make freely.

## Normalization rules

- **Semantic-first mapping**: the same column name does not guarantee the same meaning.
  Check the schema's field definition (units, scope, total vs. net) before renaming.
- **Column renaming / format conversion**: fuzzy-match agent columns to schema fields;
  convert wide/long layouts as needed.
- **Missing rows**: only fill a missing combination with zero when the schema defines that
  dimension's absence AS zero. Do NOT invent rows for tasks the agent did not produce —
  leave them out and set status to "partial".
- **Never** re-run the agent's solver, re-optimize, or re-compute its decisions.
- **Prefer the highest-precision artifact.** When the agent expresses the same decisions in
  several files — say a minute-truncated CSV and a full-precision JSON — take the lossless
  one, even if `_agent_summary.md` names the other. Precision loss is not a formatting
  choice; it silently changes the answer being graded.
- **Never take anything from the Data directory as the agent's output.** `data/` holds the
  problem input, including any baseline or seed policy shipped with the task. Submitting one
  of those grades the benchmark's own reference instead of the agent — if the workspace has
  no real deliverable, say so and set status to "failed".

## Paths & permissions

| Directory | Path | Read | Write |
|-----------|------|------|-------|
| Workspace | `{workspace}` | YES | **NO** |
| Data      | `{data_dir}` | YES | **NO** |
| Evaluator | `{evaluator}` | YES | **NO** |
| Output    | `{output_dir}` | YES | **YES** |

All writes go to the Output directory only.

## _result.json format

{{
  "status": "success",
  "notes": "what was found and what transformations were applied",
  "files_found": ["relative paths of agent output files discovered"],
  "files_normalized": ["filenames written to the output directory (excluding _result.json)"]
}}

Status: "success" = all required files written; "partial" = found output but it is
incomplete or self-contradictory (describe what is missing); "failed" = no relevant agent
output found. When the agent's output is genuinely incomplete, "partial" is the correct
outcome — do NOT reconstruct or re-solve to reach "success".
"""


async def run(evaluator: Path, workspace: Path, output_dir: Path, model: str,
              audit: dict | None = None) -> dict:
    data_dir = evaluator.parent.parent / "data"
    prompt = build_prompt(evaluator, workspace, output_dir, data_dir)

    config_dir, routed_model = _prepare_config(output_dir, model)
    cfg = RunnerConfig(
        runner_type="opencode",
        agent_config_dir=config_dir,
        model=routed_model,
        allow_external_directory=True,
        max_runtime_s=1800,      # 抽取是分钟级活儿，卡住就该失败而不是拖满默认 4h
        idle_timeout_s=600,
    )
    runner = runner_from_config(cfg)
    session = await runner.start(output_dir)
    audit = audit if audit is not None else {}
    diagnostics: list[dict] = []

    def _on_event(ev) -> None:
        # 只保留错误/诊断，不落完整 prompt、tool input 或模型正文。
        if getattr(ev, "is_error", False) or getattr(ev, "type", "") == "info":
            diagnostics.append({
                "type": getattr(ev, "type", ""),
                "tool": getattr(ev, "tool", ""),
                "content": _safe_text(getattr(ev, "content", "")),
                "is_error": bool(getattr(ev, "is_error", False)),
            })
    session.on_event = _on_event
    try:
        await session.send(prompt)
        # prompt 要求最后一步写 _result.json，但模型有时正常收束却漏写。
        # 给一次补写机会——只补文件，不重跑整个抽取。
        if not (output_dir / "_result.json").exists():
            await session.send(
                f"你已完成分析但没有写出 `_result.json`。请立刻把归一化结果写入 "
                f"`{output_dir}/_result.json`，这是唯一缺失的一步。")
    finally:
        try:
            await session.close()
        finally:
            # send()/close() 抛错时也要留下 session 级诊断，不能只剩 returncode。
            audit.update({
                "session_id": getattr(session, "session_id", None),
                "killed_reason": getattr(session, "killed_reason", None),
                "tool_turns": getattr(session, "tool_turns", None),
                "diagnostics": diagnostics,
            })

    res = output_dir / "_result.json"
    if not res.exists():
        # Do not make the model's bookkeeping file a single point of failure.
        # If a normalized payload exists, preserve it for deterministic
        # evaluator scoring and mark the protocol violation explicitly.
        payload_files = _normalized_payload_files(output_dir)
        if payload_files:
            _write_fallback_result(output_dir, payload_files)
        else:
            print(json.dumps({
                "status": "failed",
                "notes": "extractor completed without writing _result.json",
                "files_found": [], "files_normalized": [],
            }, ensure_ascii=False))
            return audit
    print(res.read_text(encoding="utf-8"))
    return audit


def main() -> None:
    ap = argparse.ArgumentParser(description="opencode 版 extractor")
    ap.add_argument("--evaluator", type=Path, required=True)
    ap.add_argument("--workspace", type=Path, required=True)
    ap.add_argument("--output", type=Path, required=True)
    ap.add_argument("--model", required=True, help="如 direct/glm-5.2")
    ap.add_argument("--audit-output", type=Path,
                    help="可选：写出本次 extractor session 的脱敏诊断 JSON")
    args = ap.parse_args()

    ev, ws, out = args.evaluator.resolve(), args.workspace.resolve(), args.output.resolve()
    if not ev.is_file():
        print(json.dumps({"status": "failed", "notes": f"evaluator not found: {ev}",
                          "files_found": [], "files_normalized": []}, ensure_ascii=False))
        sys.exit(0)
    out.mkdir(parents=True, exist_ok=True)

    audit = {
        "runtime": "opencode", "model": args.model,
        "started_at": datetime.now(timezone.utc).isoformat(),
    }
    try:
        asyncio.run(run(ev, ws, out, args.model, audit))
        audit["status"] = "completed"
    except Exception as exc:
        audit.update({"status": "error", "error_type": type(exc).__name__,
                      "error": _safe_text(exc)})
        raise
    finally:
        audit["finished_at"] = datetime.now(timezone.utc).isoformat()
        if args.audit_output:
            args.audit_output.parent.mkdir(parents=True, exist_ok=True)
            args.audit_output.write_text(
                json.dumps(audit, ensure_ascii=False, indent=2), encoding="utf-8")


if __name__ == "__main__":
    main()
