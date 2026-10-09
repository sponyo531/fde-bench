"""
FDE-Bench 通用 Extractor Agent

读取 evaluator.py 中的 EXTRACTOR_SPEC，在 agent workspace 中发现并标准化产物，
写入 normalized 目录，向 stdout 输出 JSON 状态。

用法：
    python3 benchmark/extractor_agent.py \
        --evaluator path/to/tests/evaluator.py \
        --workspace path/to/agent/working_dir \
        --output    path/to/.eval/normalized/

此脚本同时复制到各 case 的 tests/extractor_agent.py，供 test framework/test.sh 调用。
"""

import argparse
import json
import os
import re
import sys
from datetime import datetime, timezone
from pathlib import Path

import anyio
from claude_agent_sdk import ClaudeAgentOptions, ResultMessage, SystemMessage, query


# ── EXTRACTOR_SPEC 解析 ───────────────────────────────────────────────────────

def parse_extractor_spec(evaluator_path: Path) -> str:
    """从 evaluator.py 开头的 docstring 中提取 EXTRACTOR_SPEC 原文。

    找不到时返回空字符串——extractor 将回退到直接读 evaluator.py 代码推断格式。
    """
    source = evaluator_path.read_text(encoding="utf-8")
    match = re.search(r'"""(.*?)"""', source, re.DOTALL)
    if not match:
        return ""
    docstring = match.group(1)
    spec_match = re.search(r"(EXTRACTOR_SPEC:.*)", docstring, re.DOTALL)
    if not spec_match:
        return ""
    return spec_match.group(1).strip()


# ── Prompt 构建 ───────────────────────────────────────────────────────────────

def build_prompt(
    spec: str,
    evaluator_path: Path,
    workspace: Path,
    output_dir: Path,
    data_dir: Path,
    target_file: Path | None = None,
) -> str:
    if target_file is not None:
        task_section = f"""1. **Use ONLY this designated solution file as your source of truth:**
   `{target_file}`
   This is the highest-scoring valid solution selected by the final evaluator.
   Do NOT scan for or use any other solution file — ignore all other candidates.
2. Read the target schema below to understand what output files and columns are expected.
3. Normalize the designated file to match the required schema and write to the output directory.
4. **If the deliverable is executable code**: the evaluator imports it from the OUTPUT
   directory, so copy every workspace-local module the designated file imports (directly or
   transitively) alongside it, keeping their original filenames. Copying only the entry file
   leaves it orphaned — ModuleNotFoundError, and the run scores 0 despite working code.
5. Write a file called `_result.json` to the output directory as your final action."""
    else:
        task_section = """1. **Read `_agent_summary.md` first** (if present in workspace). This tells you which file the agent considers its final answer and what values it reported. Use this to guide which files to look for in the next step — do not skip this even if you think you already know what to look for.
2. Read the target schema below to understand what output files and columns are expected.
3. Scan the workspace to find the agent's output files (they may have different names or column names). If multiple candidate files exist, use what you learned from `_agent_summary.md` to pick the one the agent intended as its final result.
4. Normalize the chosen file(s) to match the required schema and write to the output directory.
5. **If the deliverable is executable code** (the schema asks for a `.py` file defining a
   specific function, e.g. `solution.py` with `allocation_policy`): the evaluator imports it
   from the OUTPUT directory, so any local module it depends on must be there too. Copy the
   entry file under the required name AND copy every workspace-local module it imports
   (directly or transitively) alongside it, keeping their original filenames. Copying only the
   entry file leaves it orphaned — the import fails with ModuleNotFoundError and the run scores
   0 even though the agent's code was fine. Do not rewrite imports; just bring the files along.
6. Write a file called `_result.json` to the output directory as your final action."""

    return f"""You are a data extractor for a benchmark evaluation pipeline.

## Your task

{task_section}

## Target schema (supplementary hint)

{spec if spec else "(not provided — infer from evaluator.py source code below)"}

The schema uses one of two formats:
- Flat: `plan_file` + `required_columns` (single output file)
- List: `files` with per-file `name`, `required_columns`, and optional `notes` (multiple output files)

If the schema hint above is empty, read `{evaluator_path}` to infer the expected filename(s)
and required columns from the `PLAN_FILE` / `PLAN_FILES` / `REQUIRED_COLUMNS` constants.

## Verify by running the evaluator (required)

A prose schema can never enumerate every output shape an agent might produce, so do not
guess whether your normalization is right — **check it**. Once you have written the output
files, run the evaluator against your output directory:

```bash
python3 {evaluator_path} --data-dir {data_dir} --submission-dir {output_dir}
```

Read the JSON it prints and react to *structural* failures only:

| What you see | What it means | What to do |
|---|---|---|
| `ModuleNotFoundError` | a local module the entry file imports is missing | copy that module in too |
| `AttributeError: must define <fn>` | you picked the wrong entry file | try the next candidate |
| `缺 <file>` / `KeyError` / missing column | filename or column mapping is off | fix the mapping |
| valid JSON with any score, **including 0** | the evaluator ran to completion | **stop — you are done** |

**Stop as soon as the evaluator returns a score.** A score of 0 is a legitimate, final
answer: it means the agent's solution ran and did not meet the constraints. Never edit
values, add rows, drop rows, pick a different (better-scoring) solution file, or otherwise
touch the substance of the agent's answer in order to raise the number. You normalize
*representation* — filenames, column names, layout, encoding. The decisions themselves are
the agent's and must survive unchanged. Re-running the evaluator to chase a higher score
would make this pipeline grade its own homework instead of the agent's.

## Normalization rules

- **Semantic-first mapping**: Same column name does NOT guarantee same meaning. Before renaming, check the schema's `column_formats` definition and reason whether the agent's value matches that definition (units, scope, inclusions/exclusions, total vs. net). If they differ, transform the value first; do not rely on name match alone.
- **Column renaming**: fuzzy-match agent column names to the required names in the schema.
- **Format conversion**: convert between wide/long table formats as needed.
- **Missing rows**: only fill a missing combination with zero when the schema defines that dimension's absence AS zero (e.g. an unlisted cell in a demand/quantity matrix = 0). Do NOT invent rows for tasks/items the agent did not produce — leave them out and set status to "partial" (see Important rules).
- **Value constraints**: clamp negative values to zero for quantity/unit columns.
- **Joins**: if schema notes mention reconstructing an INDEX via a join (a lossless, content-determined key), use reference files in the Data directory. Do NOT use a join to synthesize a value/decision the agent did not make.

## Paths & permissions

| Directory | Path | Read | Write |
|-----------|------|------|-------|
| Workspace | `{workspace}` | YES | **NO** |
| Data      | `{data_dir}` | YES | **NO** |
| Evaluator | `{evaluator_path}` | YES (single file) | **NO** |
| Output    | `{output_dir}` | YES | **YES** |

- **All writes** (normalized files, `_result.json`) go to the Output directory only.
- **Never** create, modify, or delete files outside the Output directory.
- When using Bash to run Python scripts, the same rules apply: read from Workspace / Data, write to Output only.

## _result.json format

Write this file last, after all normalized files are written:

{{
  "status": "success",
  "notes": "brief explanation of what was found and any transformations applied",
  "files_found": ["relative paths of agent output files discovered"],
  "files_normalized": ["filenames written to the output directory (excluding _result.json)"]
}}

Status values:
- "success"  – all required files written to output directory
- "partial"  – found agent output but it cannot be coherently evaluated as-is, for either reason:
               (1) INCOMPLETE — missing rows/tasks or a required decision column left empty; or
               (2) CONFLICT — the agent's own output is self-contradictory (conflicting values
               for the same quantity) with no schema-designated source to resolve it.
               Describe what is missing / in conflict in notes.
- "failed"   – could not find any relevant agent output in the workspace

When the agent's output is genuinely incomplete (missing tasks/rows, or a required
decision column left empty), "partial" is the correct, expected outcome — do NOT fill,
reconstruct, or re-solve to reach "success".

## Important rules

- **Path permissions**: see the table above — all writes go to Output only.
- Use Glob to scan workspace, Read to inspect files, Write to produce output.
- You may use Bash to run `python3` scripts for FORMAT transformation only (parsing, reshaping, renaming columns, wide/long conversion).
  Do NOT use Bash (or any other means) to re-run the agent's solver, re-optimize, or re-compute the agent's decisions.
  Do NOT use: rm, curl, wget, pip install, or any network/system commands.
- **Do NOT fabricate, infer, or re-solve missing content.** This covers missing
  dimensions (aggregate→breakdown), missing rows/tasks the agent did not produce, and
  any required DECISION column the agent left empty. Do NOT distribute/impute from
  aggregates, and do NOT re-run the agent's solver. Instead set status to "partial"
  and describe exactly what is missing.
- **Copy faithfully; do not resolve the agent's contradictions.** Preserve the agent's
  values exactly — do NOT round or truncate (precision can matter for feasibility), and
  do NOT drop values the agent provided. If the agent's own outputs give CONFLICTING
  values for the same quantity (e.g. two different charge amounts for one stop, or a
  summary that disagrees with the detail), that is the agent's inconsistency — it is NOT
  yours to silently resolve. If the schema designates which source to use, use it.
  Otherwise do NOT pick one to make the solution pass: set status to "partial", note the
  reason as CONFLICT, and describe it (which sources, which values). A self-contradictory
  solution must be surfaced, not scored as if coherent.
"""


# ── Main ──────────────────────────────────────────────────────────────────────

async def run(evaluator: Path, workspace: Path, output_dir: Path, target_file: Path | None = None):
    spec = parse_extractor_spec(evaluator)
    data_dir = evaluator.parent.parent / "data"

    prompt = build_prompt(spec, evaluator, workspace, output_dir, data_dir, target_file)

    session_id = None
    stop_reason = None
    started_at = datetime.now(timezone.utc).isoformat()

    # 模型由环境变量 ANTHROPIC_MODEL 控制（评测框架 的 eval.py 从 bench_config.toml 注入）
    _model = os.environ.get("ANTHROPIC_MODEL") or None

    # resume 时默认发空串（仅让 agent 接着用工具）；漏写 _result.json 的那种
    # 情形需要明确指令，见循环末尾。
    resume_prompt = ""
    for attempt in range(6):
        is_resume = attempt > 0
        if is_resume:
            print(f"[extractor auto-resume #{attempt}] stop_reason=tool_use, resuming {session_id}...", file=sys.stderr)

        async for message in query(
            prompt=resume_prompt if is_resume else prompt,
            options=ClaudeAgentOptions(
                cwd=str(output_dir),
                allowed_tools=["Read", "Glob", "Grep", "Write", "Bash"],
                permission_mode="bypassPermissions",
                max_turns=100 if is_resume else 100,
                resume=session_id if is_resume else None,
                model=_model,
            ),
        ):
            if isinstance(message, SystemMessage) and message.subtype == "init":
                if not is_resume:
                    session_id = message.data.get("session_id")
            elif isinstance(message, ResultMessage):
                # SDK 0.1.x：ResultMessage 用 subtype 表示结束原因；
                # 老版本用 stop_reason。兼容两者。
                stop_reason = getattr(message, "stop_reason", None) or getattr(message, "subtype", None)

        # 仅在因 max_turns 中断（还想继续用工具）时 resume；正常结束则停止
        if stop_reason not in ("tool_use", "error_max_turns"):
            # extractor 是 LLM，prompt 要求"最后一步写 _result.json"但它有时正常收束却漏写
            # （实测 stop_reason=success、跑了 83-259s、normalized/ 却是空的）。
            # 代码只认文件在不在，漏写就判 failed，而 agent 的产物其实是好的。
            # 这里给它一次 resume 的机会，只补写文件，不重跑整个抽取。
            if not (output_dir / "_result.json").exists() and attempt < 2:
                print(f"[extractor] stop_reason={stop_reason} 但未写 _result.json，"
                      f"resume 补写 (#{attempt + 1})", file=sys.stderr)
                resume_prompt = ("你已完成分析但没有写出 `_result.json`。请立刻把归一化结果写入 "
                                 f"`{output_dir}/_result.json`，这是唯一缺失的一步。")
                continue
            break

    # 记录 extractor 运行元信息
    extractor_run = {
        "session_id": session_id,
        "started_at": started_at,
        "finished_at": datetime.now(timezone.utc).isoformat(),
        "stop_reason": stop_reason,
    }
    (output_dir.parent / "extractor_run.json").write_text(
        json.dumps(extractor_run, indent=2, ensure_ascii=False)
    )

    # 读 agent 写的 _result.json → 输出到 stdout 供 run_eval.py 解析
    result_path = output_dir / "_result.json"
    if result_path.exists():
        print(result_path.read_text(encoding="utf-8"))
    else:
        print(json.dumps({
            "status": "failed",
            "notes": "extractor agent completed without writing _result.json",
            "files_found": [],
            "files_normalized": [],
        }))


def main():
    parser = argparse.ArgumentParser(description="FDE-Bench 通用 Extractor Agent")
    parser.add_argument("--evaluator", type=Path, required=True)
    parser.add_argument("--workspace", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--target-file", type=Path, default=None,
                        help="指定唯一源解文件（最终评估器最高分解）；提供后 extractor 只处理此文件")
    args = parser.parse_args()

    evaluator = args.evaluator.resolve()
    workspace = args.workspace.resolve()
    output_dir = args.output.resolve()
    target_file = args.target_file.resolve() if args.target_file else None

    if not evaluator.is_file():
        print(json.dumps({
            "status": "failed",
            "notes": f"evaluator not found: {evaluator}",
            "files_found": [],
            "files_normalized": [],
        }))
        sys.exit(0)

    output_dir.mkdir(parents=True, exist_ok=True)

    anyio.run(run, evaluator, workspace, output_dir, target_file)


if __name__ == "__main__":
    main()
