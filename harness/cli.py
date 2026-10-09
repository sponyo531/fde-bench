"""跑一个 case。

    python -m harness.cli --case 03_city_delivery_route_planning --condition Hidden
    python -m harness.cli --case 03_city_delivery_route_planning --condition Hidden,Interact,Full --runs 3
"""

from __future__ import annotations

import argparse
import asyncio
import json
import os
import time
from pathlib import Path

from .backends import RunnerConfig, runner_from_config
from .conditions import CONDITIONS, ClarifyLimits
from .run import (RunSpec, finalize, setup_run, update_manifest,
                  workspace_produced, workspace_snapshot)
from .isolation import sandbox as sb
from .isolation import environment as envmod
from .clarify.loop import run_clarify
from .scoring.solve import score_run, summarize
from .scoring.usage import collect_usage
from .backends.usage import models_mismatch
from .results import update_run_statuses

_ROOT = Path(__file__).resolve().parent.parent


def _record_env(run_dir: Path, info: dict) -> None:
    """把实际生效的 agent 配置写进 manifest，供复现核对。"""
    import json
    mf = run_dir / "manifest.json"
    if not mf.is_file():
        return
    data = json.loads(mf.read_text(encoding="utf-8"))
    data["agent_env"] = info
    mf.write_text(json.dumps(data, indent=2, ensure_ascii=False), encoding="utf-8")


def _completed_despite_nonzero_exit(status: str, produced: list | None, *,
                                   backend_reason: str | None = None) -> bool:
    """后端异常但磁盘上已有交付物 → 该 run 应改判为可评分。

    评分对象是产物，不是退出码。实测 claude-code 跑满一轮、产物齐全、result 里
    terminal_reason=completed，进程仍 exit=1；若保留 status=error，这份 run 会被
    三重丢弃——不评分（gate 只放行 ok/timeout）、被 report/analysis 的
    EXCLUDED_STATUS 剔除。后端层 _stream_completed 只能拦"流里有可解析完成事件"的
    情形，输出被 max_tokens 截断时拦不住，故在此以"产物存在"兜底。
    """
    if status not in {"error", "degraded"} or not produced:
        return False
    if (backend_reason or "").startswith("opencode_"):
        # An empty/abnormal terminal after writing only solver sources is not
        # completion. Keep the existing rescue for candidate payloads (whose
        # validity is still judged by the scorer), and for other backends.
        source_suffixes = {".py", ".pyc", ".sh", ".bash", ".js", ".mjs", ".cjs",
                           ".ts", ".tsx", ".jsx", ".r", ".c", ".cc", ".cpp",
                           ".h", ".hpp", ".java", ".rs", ".go", ".jl"}
        if all(Path(p).suffix.lower() in source_suffixes for p in produced):
            return False
    return True


def _record_backend_diagnostics(run_dir: Path, session) -> None:
    """Recovery evidence survives even when the eventual terminal is normal."""
    diagnostics = getattr(session, "backend_diagnostics", None)
    path = run_dir / "usage.json"
    if diagnostics and path.is_file():
        data = json.loads(path.read_text(encoding="utf-8"))
        data["backend_diagnostics"] = diagnostics
        path.write_text(json.dumps(data, ensure_ascii=False, indent=2), encoding="utf-8")


def _purge_hidden_knowledge_if_thin(spec: RunSpec, run_dir: Path, use_sandbox: bool) -> None:
    """瘦 case 模式：agent 启动前删掉 case 目录里的 information.md。

    pod 用 --subpath 只挂 run 目录，gt.json / tests 不进瘦 case——但 information.md
    必须进来（answerer 要读它作答）。它就躺在 workspace 上面两三层的目录里，
    Hidden / Interact 条件下 agent 一个 `cat ../../<case>/information.md` 就把本该
    隐藏的口径全看了；`--no-sandbox` 下没有任何东西挡它。
    prompt（Full 条件）与 answerer 在调用本函数前都已读完，删掉不影响任何一方；
    评分阶段是另一个运行单元、挂评测目录，不用这份副本。

    只在 DELIVER_THIN_CASE=1 且目录确实是瘦 case（没有
    gt.json）时动手——绝不能删仓库 case/ 里的原件。
    """
    if os.environ.get("DELIVER_THIN_CASE") != "1":
        return
    case = Path(spec.case)
    if (case / "gt.json").exists() or (case / "tests").exists():
        print("  [isolation] case 目录含 gt.json/tests，不是瘦 case，拒绝 purge", flush=True)
        return
    info = case / "information.md"
    if info.is_file():
        info.unlink()
        print("  [isolation] 瘦 case 的 information.md 已在 agent 启动前移除", flush=True)
    update_manifest(run_dir, information_purged=info.exists() is False)


def _sync_clarify_phase_stats(run_dir: Path, token_usage: dict | None) -> dict | None:
    """用脚手架事件时间线校正 native clarify.json 的阶段统计。

    opencode 的 question tool 在一次 ``session.send()`` 内完成提问、回答和后续
    求解，所以 ``run_clarify`` 无法在 send 返回前取得控制权；若直接计时，会把
    整个会话都误记为澄清时间。``collect_usage`` 已根据 SQLite 中最后一条
    question tool 的时间戳切分 clarify/solve，这里将该权威结果回写到
    clarify.json。文本多轮脚手架没有 phase 时间线时保持原记录不变。
    """
    path = run_dir / "clarify.json"
    phase = (token_usage or {}).get("phase") or {}
    if not path.is_file() or phase.get("clarify_secs") is None:
        return None
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, ValueError, json.JSONDecodeError):
        return None
    data["clarify_secs"] = phase["clarify_secs"]
    if phase.get("clarify_tokens") is not None:
        data["total_tokens"] = phase["clarify_tokens"]
    # 来源要写真的：opencode 是 SQLite 时间线；其他后端是各自账本的阶段标签
    # （OpenHands 按最后作答时的调用数切）。曾一律写 opencode_event_timeline。
    src = (token_usage or {}).get("source")
    data["phase_stats_source"] = ("opencode_sqlite" if src in (None, "opencode_sqlite")
                                  else "backend_ledger")
    path.write_text(json.dumps(data, ensure_ascii=False, indent=2), encoding="utf-8")
    return data


def _write_terminal_score_record(run_dir: Path, spec: RunSpec, status: str,
                                 produced: list[str], scored: dict) -> None:
    """Persist an explicit terminal scoring outcome when no score was run.

    Historically a run with no deliverable simply lacked ``result.json``.  That
    made an agent failure indistinguishable from a still-running or crashed
    scoring job.  Keep the numeric score null and record the reason instead of
    manufacturing a zero.  Existing result files are never overwritten.
    """
    path = run_dir / "result.json"
    if path.exists():
        return
    if produced and status in {"ok", "timeout"} and not scored:
        category = "scorer_no_record"
        message = "评分入口未产出 result.json"
        scoring_status = "failed"
        artifact_status = "partial"
    elif produced:
        category = "run_not_scorable"
        message = f"求解阶段状态为 {status}，未进入评分"
        scoring_status = "failed"
        artifact_status = "partial"
    elif status in {"ok", "timeout"}:
        category = "artifact_missing"
        message = "求解结束时未发现 Agent 交付文件，评分不适用"
        scoring_status = "not_applicable"
        artifact_status = "none"
    else:
        category = "run_failed"
        message = f"求解阶段状态为 {status} 且无交付物，评分不适用"
        scoring_status = "not_applicable"
        artifact_status = "none"
    record = {
        "case": spec.case.name,
        "condition": spec.condition,
        "model": spec.model,
        "run_status": status,
        "artifact_status": artifact_status,
        "scoring_status": scoring_status,
        "final_score": None,
        "overall_score": None,
        "combined_score": None,
        "error_info": {"category": category, "message": message},
    }
    path.write_text(json.dumps(record, ensure_ascii=False, indent=2), encoding="utf-8")


_TIMEOUT_GRACE_S = 120   # 后端 killpg 收尾的余量；外层 wait_for 仅兜底


def _effective_idle_timeout(timeout_s: int) -> int:
    """Return an idle budget that cannot pre-empt the requested run budget.

    ``--timeout`` is the public per-run limit.  Historically the CLI passed
    that value to ``max_runtime_s`` but left ``idle_timeout_s`` at the config
    default (43200s), so a run submitted with ``--timeout 28800`` could still
    be killed as idle after twelve hours.  Idle remains a useful watchdog for a
    provider stall, but it must not be shorter than the explicitly requested
    total budget.  Backend-specific callers (for example the extractor) pass
    their own idle timeout directly through ``RunnerConfig``.
    """
    try:
        from .config import get
        configured = int(get("agent", "idle_timeout_s", timeout_s))
    except (TypeError, ValueError, SystemExit):
        configured = timeout_s
    return max(int(timeout_s), configured)


async def _one(spec: RunSpec, results_root: Path, timeout_s: int, skip_eval: bool = False,
               use_sandbox: bool = True, container: object = None) -> dict:
    run_dir, ws, prompt = setup_run(spec, results_root)
    print(f"\n{'─' * 64}\n{spec.run_id}\n  workspace: {ws}")
    workspace_before = workspace_snapshot(ws)

    events: list = []
    killed: str | None = None          # watchdog 击杀原因（timeout 的细分）
    degraded_reason: str | None = None # 后端协议/终态异常（可重试的 degraded）

    answerer = None
    # 显式记录没有 answerer 的条件，避免 manifest 中缺字段与配置缺失混淆。
    update_manifest(run_dir, answerer_config=None, answerer_model=None)
    from .conditions import _INTERACT, canonical
    if canonical(spec.condition) in _INTERACT:
        from .clarify.answerer import ApiAnswerer
        info = spec.case / "information.md"
        # answerer 只吃 instruction + information.md。gt.json 不进来——它是澄清指标的
        # 考点清单，喂给 answerer 会让只写在 gt.json 里的考点变成「问了就能拿到」。
        answerer = ApiAnswerer(
            instruction_text=(spec.case / "instruction.md").read_text(encoding="utf-8"),
            information=info.read_text(encoding="utf-8") if info.is_file() else "",
        )
        answerer_config = answerer.describe()
        update_manifest(
            run_dir,
            answerer_config=answerer_config,
            answerer_model=answerer_config["model"],
        )

    # 中立环境：只继承 provider/model，丢弃宿主的 agent/skill/permission
    import os
    neutral = envmod.build_env(run_dir, spec.model)
    _saved = {k: os.environ.get(k) for k in ("XDG_CONFIG_HOME", "XDG_DATA_HOME", "XDG_CACHE_HOME")}
    os.environ.update({k: neutral[k] for k in _saved})

    sandbox_cfg = None
    if container:
        from .isolation import container as ct
        ct.require(container)          # 镜像缺失即抛错，绝不静默回落到宿主机
        use_sandbox = False           # 容器自带隔离
    if use_sandbox:
        if not sb.available(ws):
            raise SystemExit(
                "[sandbox] 无法建立文件隔离（需 bwrap + user namespace）。"
                "确认环境后重试，或显式 --no-sandbox 承担泄漏风险。"
            )
        sandbox_cfg = sb.build_config(spec.case, ws, results_root, run_dir / '.agent_home')
    # 到这里 prompt（含 Full 条件的 information）与 answerer 都已把 information.md
    # 读进内存；agent 尚未启动。瘦 case 模式下把它从磁盘上拿走。
    _purge_hidden_knowledge_if_thin(spec, run_dir, use_sandbox)

    runner = runner_from_config(RunnerConfig(
        runner_type=spec.scaffold,
        agent_config_dir=None,      # 平台原生形态：不注入任何 agent 配置
        model=spec.model,
        sandbox=sandbox_cfg,
        container=container,
        # 超时必须由持有进程句柄的后端 watchdog 执行。外层 asyncio.wait_for 取消的是
        # await，而 send() 跑在 asyncio.to_thread 里——线程不可中断，子进程会继续跑。
        # 实测：wait_for 3600s 到点写了 status=timeout，agent 又干了 50 分钟并写出
        # NOTES.md，produced 成了过期快照，评分读的是仍在变动的目录。
        max_runtime_s=timeout_s,
        idle_timeout_s=_effective_idle_timeout(timeout_s),
        max_turns=spec.max_turns,
    ))
    # 内层 watchdog 先开火，让超时走 killpg 的干净路径（保留已产出的交付物照常评分）；
    # 外层 wait_for 退化为兜底，只在后端连 kill 都没能收尾时才触发。
    outer_timeout = timeout_s + _TIMEOUT_GRACE_S
    t0 = time.time()
    tool_turns = None          # 工具调用轮数；只有 native 后端数得到，其余为 None
    session = None
    opencode_send_count = [0]
    try:
        session = await runner.start(ws)

        # info 事件必须**打出来**，不能只进 events 列表。
        # 后端用 info 事件报的都是"跑不出结果的原因"：看门狗判因、澄清轮数撞顶、
        # question 轮询异常。而 finalize() 只把 len(events) 写进 usage.json，
        # 内容全丢——实测因此白跑了两个 job：agent 卡在 question 工具上无人应答，
        # 而唯一能指出这一点的诊断事件静默躺在一个没人读的列表里。
        def _on_event(ev) -> None:
            events.append(ev)
            if getattr(ev, "type", "") == "info":
                print(f"  {ev.content}", flush=True)

        session.on_event = _on_event
        # OpenCode backend is intentionally left untouched: count its logical
        # sends at the harness boundary so this metric does not alter the
        # already-running OpenCode implementation.
        if spec.scaffold in {"opencode", "opencode-run"}:
            _original_send = session.send

            async def _counted_send(*args, **kwargs):
                opencode_send_count[0] += 1
                return await _original_send(*args, **kwargs)

            session.send = _counted_send
        async with session:
            if answerer is not None:
                response, _ = await asyncio.wait_for(
                    run_clarify(
                        session, prompt, answerer,
                        max_rounds=spec.limits.max_rounds,
                        max_questions=spec.limits.max_questions,
                        log_path=run_dir / "clarify.json",
                    ),
                    timeout=outer_timeout,
                )
            else:
                # Non-interactive conditions have no clarification loop to set
                # the usage phase.  Mark the direct model call explicitly so
                # DSH (and other ledger-backed scaffolds) expose solve_tokens /
                # solve_secs consistently with Interact-Req.
                set_phase = getattr(session, "set_usage_phase", None)
                if callable(set_phase):
                    set_phase("solve")
                response = await asyncio.wait_for(session.send(prompt), timeout=outer_timeout)
        # watchdog 在后端内部 killpg 后 send() 会正常返回，外层 wait_for 不触发。
        # 不读 killed_reason 就会把被掐死的残缺 run 记成 ok（实测：180s 被杀、只落下
        # solve_routes.py，却报 status=ok）。
        killed = getattr(session, "killed_reason", None)
        degraded_reason = getattr(session, "degraded_reason", None)
        tool_turns = getattr(session, "tool_turns", None)
        status = "timeout" if killed else ("degraded" if degraded_reason else "ok")
    except asyncio.TimeoutError:
        response, status = "", "timeout"
        killed = "outer_wait_for"          # 后端连 killpg 都没收尾，兜底路径
        degraded_reason = None
    except Exception as exc:                      # harness 自身错误，与 agent 失败区分
        response, status = f"{type(exc).__name__}: {exc}", "error"
        degraded_reason = getattr(session, "degraded_reason", None)
        # 必须打出来：之前只写进 response.txt，终端上只剩一行 status=error，
        # 实测 gemini 缺凭据 2 秒退出、排查时无从下手
        print(f"  [error] {response[:600]}", flush=True)

    elapsed = time.time() - t0
    for k, v in _saved.items():
        if v is None:
            os.environ.pop(k, None)
        else:
            os.environ[k] = v

    _record_env(run_dir, envmod.describe(run_dir))
    # 只记录运行后新增或内容变化的文件；data/ 是 harness 的输入副本，不属于
    # Agent 交付物。递归扫描仍覆盖 output/ 等嵌套交付目录。
    produced = workspace_produced(ws, workspace_before)

    # CLI 退出非零 / 后端协议终态异常，但磁盘上已有交付物 → 交付物才是评分对象，
    # 退出码/终态不是。OpenHands 可能写完 solution.json 后没有正确 finish；若仍
    # 排除会让有效率、均分和工具/token 指标都少一个真实可评分样本。
    # 判据与存证详见 _completed_despite_nonzero_exit。改判 ok 让它进评分与统计，
    # 退出码原文落 cli_exit_anomaly + response.txt + .cli_logs/，分析阶段可据此过滤。
    cli_exit_anomaly = None
    if _completed_despite_nonzero_exit(status, produced, backend_reason=degraded_reason):
        cli_exit_anomaly = (response or "")[:600]
        print(f"  [{status}→ok] 后端终态异常，但已产出 {len(produced)} 个交付物 → 按完成评分"
              f"（退出码详情见 cli_exit_anomaly / .cli_logs/）")
        status = "ok"

    # 退化 run：进程正常退出，但 agent 既没说话也没落文件。观察到的成因是 provider
    # 把整轮输出耗在 reasoning 上、从未收束成 text（Full 条件 825s / 3373 个 reasoning
    # part / 0 个 agent text）。若仍记为 ok，会在批量结果里伪装成"agent 能力不足"，
    # 掩盖基础设施故障，故当场判失败。
    if status == "ok" and not response.strip() and not produced:
        status = "degraded"

    # 澄清空转检测：撞满 max_rounds 说明 agent 一直在重发提问、从未进入执行。
    # 实测一次 Interact-Req run 连发 15 轮相同问题、空烧 19 分钟，status 却报 ok——
    # 批量跑几百个 run 时这类空转会被完全淹没，必须显式标记。
    cl = run_dir / "clarify.json"
    if cl.is_file():
        try:
            cj = json.loads(cl.read_text(encoding="utf-8"))
        except Exception:
            cj = {}
        if cj.get("hit_round_limit"):
            killed = killed or "clarify_round_limit"
            if status == "ok":
                status = "degraded"
            print(f"  [clarify] 撞满 {cj.get('total_rounds')} 轮上限 — "
                  f"agent 未进入执行阶段，标记 degraded")

    if spec.scaffold in {"opencode", "opencode-run"}:
        agent_sends = opencode_send_count[0] or None
    else:
        try:
            agent_sends = session.agent_send_count()
        except Exception:
            agent_sends = None
    finalize(run_dir, response, elapsed, events, status=status, produced=produced,
             tool_calls_known=bool(getattr(session, "reports_tool_calls", False)),
             agent_sends=agent_sends)

    # 区分 idle / max_runtime / max_turns / outer_wait_for：都记 timeout，但成因
    # 不同，分析阶段要能分开看（idle 常是 provider 卡住，max_runtime 才是真跑不完，
    # max_turns 是在打转）。tool_turns 无论是否被截断都记——它是"干了多少活"的
    # 直接度量，跨模型对比时比耗时更能说明问题。
    if killed or degraded_reason or tool_turns is not None or cli_exit_anomaly:
        u = run_dir / "usage.json"
        if u.is_file():
            d = json.loads(u.read_text(encoding="utf-8"))
            if killed:
                d["killed_reason"] = killed
            if degraded_reason:
                # 已有产物的异常会在上面抢救为 ok，只留异常证据；无产物才是
                # operational failure，写 failure_reason 供自动归因使用。
                key = "failure_reason" if status in {"degraded", "error"} else "backend_anomaly"
                d[key] = degraded_reason
            if tool_turns is not None:
                d["tool_turns"] = tool_turns
            if cli_exit_anomaly:
                # 进程退出非零但产物齐全——status 已改判 ok 以进入评分/统计，
                # 这里留退出码原文，分析阶段可据此把这类 run 单独摘出来核。
                d["cli_exit_anomaly"] = cli_exit_anomaly
            u.write_text(json.dumps(d, ensure_ascii=False, indent=2), encoding="utf-8")
    _record_backend_diagnostics(run_dir, session)

    # token / cache / 费用：从脚手架自己的记录里采集（opencode 走 SQLite）。
    # 采不到时字段为 None，聚合阶段跳过而非记 0。
    # OpenCode remains authoritative through SQLite.  Other backends may have
    # optional native metadata collected on their session; it is used only
    # when no OpenCode DB exists.
    backend_usage = None
    try:
        backend_usage = session.usage_snapshot()
    except Exception:
        backend_usage = None
    try:
        observed_models = session.effective_models()
    except Exception:
        observed_models = []
    if observed_models:
        requested_model = json.loads(
            (run_dir / "manifest.json").read_text(encoding="utf-8")
        ).get("model")
        update_manifest(run_dir, effective_models=observed_models,
                        model_mismatch=models_mismatch(requested_model, observed_models))
        if backend_usage is not None:
            backend_usage = {**backend_usage, "effective_models": observed_models}
    tok = collect_usage(run_dir, (envmod.describe(run_dir) or {}).get("model"),
                        backend_usage=backend_usage)
    # native question tool 与求解共享一次 send()；run_clarify 的墙钟计时覆盖了整个
    # 会话，不能作为澄清耗时。以 opencode 事件时间线的阶段切分覆盖派生统计。
    synced_clarify = _sync_clarify_phase_stats(run_dir, tok)
    u = run_dir / "usage.json"
    data = json.loads(u.read_text(encoding="utf-8")) if u.is_file() else {}
    data["token_usage"] = tok
    # 从 clarify.json 取澄清阶段统计（native 通路有 clarify_secs，文本通路有 total_tokens）
    cl = run_dir / "clarify.json"
    if cl.is_file():
        try:
            cj = synced_clarify or json.loads(cl.read_text(encoding="utf-8"))
            if cj.get("total_rounds", 0) > 0:
                data["clarify_secs"] = cj.get("clarify_secs", 0)
                data["clarify_tokens"] = cj.get("total_tokens", 0)
                data["clarify_questions"] = cj.get("total_questions", 0)
        except Exception:
            pass
    u.write_text(json.dumps(data, ensure_ascii=False, indent=2), encoding="utf-8")

    _why = f" [{killed}]" if killed else ""
    print(f"  status={status}{_why}  {elapsed:.0f}s  产物: {produced or '(无)'}")
    if status == "degraded":
        print("  [degraded] 无文本输出且无产物 — 疑似 provider/脚手架故障，不计入有效样本")

    scored = {}
    # 超时但已落下产物的 run 仍要评分：它代表"做出来了但违约/没做完"，与 degraded
    # （根本没跑起来）是两种不同结论。不评分会把两者一起记成 0，抹掉区别。
    # status 保留 timeout，供分析阶段按预算受限单独看。
    if status in ("ok", "timeout") and produced and not skip_eval:
        scored = score_run(spec.case, run_dir)
        print(f"  score: {summarize(scored)}")

    # Close the scoring state machine even when there is no deliverable.  This
    # writes a null-score record (never a fake zero), while --skip-eval still
    # intentionally leaves scoring for a later pass.
    if not skip_eval:
        _write_terminal_score_record(run_dir, spec, status, produced, scored)

    # 「白问」检测：Hidden / Full / oracle 条件下无人应答，但 agent 仍可能自发提问、
    # 白耗预算。不记录的话，Hidden 的分数会被这类空转压低，而 `Full − Hidden` 是论文的头号
    # 数字——虚高方向恰好"有利于结论"，审稿人必查（见 docs/FRAMEWORK.md §2）。
    # 只记账、不作答：作答就变成 Interact 条件了。
    if answerer is None and status in ("ok", "timeout"):
        try:
            from .clarify.detect import detect
            asked = detect(response, produced_delta=produced)
            if asked:
                u = run_dir / "usage.json"
                data = json.loads(u.read_text(encoding="utf-8"))
                data["unanswered_asks"] = {
                    "n_questions": len(asked),
                    "questions": asked[:20],
                    "produced_anything": bool(produced),
                }
                u.write_text(json.dumps(data, ensure_ascii=False, indent=2),
                             encoding="utf-8")
                print(f"  [白问] 无人应答的条件下仍提了 {len(asked)} 个问题"
                      f"（产物：{'有' if produced else '无'}）")
        except Exception as exc:                      # noqa: BLE001
            print(f"  [白问] 检测失败（不影响评分）: {type(exc).__name__}: {exc}")

    # 澄清评分（Ask-F1 / recall / precision / KQC / CE-A / CE-B）。
    # 与求解评分分开：产物为空的 run 照样要评澄清——"问得对但没做出来"和
    # "没问也没做出来"是两种不同结论，混在一起就看不出瓶颈在哪一段。
    # 只在有 clarify.json（即 Interact 系列）时评，Hidden/Full 无提问阶段可评。
    if (run_dir / "clarify.json").is_file() and not skip_eval:
        try:
            from .clarify.score import score_clarification
            cs = score_clarification(spec.case, run_dir)
            print(f"  clarify: ask_f1={cs.get('ask_f1')} recall={cs.get('recall')} "
                  f"precision={cs.get('precision')} n_q={cs.get('n_questions')}")
        except Exception as exc:                  # noqa: BLE001
            # judge 是外部 LLM，失败不该让整个 run 记为失败——求解分已经算完了
            print(f"  [clarify] 评分失败（不影响求解分）: {type(exc).__name__}: {exc}")

    # 分别记录进程、产物、评分和可行性状态，避免 timeout/无产物/evaluator 0
    # 在离线分析中被混成同一种失败。
    update_run_statuses(run_dir)

    return {
        "run_id": spec.run_id, "status": status,
        "elapsed_s": elapsed, "score": scored,
    }


async def _main(args) -> None:
    if args.max_rounds < 0 or args.max_questions < 0:
        raise SystemExit(
            "--max-rounds 和 --max-questions 不能为负数（max-questions=0 表示不限）")
    if args.run_index_start < 1:
        raise SystemExit("--run-index-start 必须至少为 1")
    case = _ROOT / "cases" / args.case
    if not case.is_dir():
        raise SystemExit(f"case not found: {case}")

    conditions = [c.strip() for c in args.condition.split(",")]
    from .conditions import valid_condition
    bad = [c for c in conditions if not valid_condition(c)]
    if bad:
        raise SystemExit(f"unknown condition(s): {bad}; expected {CONDITIONS}")

    limits = ClarifyLimits(max_rounds=args.max_rounds, max_questions=args.max_questions)
    # 必须绝对路径：agent 的 cwd 就是 workspace，相对路径会在其内部再建一层
    results_root = Path(args.out).resolve()

    container_spec = None
    if args.container:
        from .isolation.container import agent_binary, build_spec
        if agent_binary(args.scaffold, args.agent_bin) is None:
            raise SystemExit(
                f"找不到 {args.scaffold} 的可执行文件，无法挂入容器；用 --agent-bin 指定"
            )
        container_spec = build_spec(args.container, args.scaffold, agent_bin=args.agent_bin)

    for cond in conditions:
        for i in range(args.run_index_start, args.run_index_start + args.runs):
            spec = RunSpec(
                case=case, condition=cond, model=args.model,
                scaffold=args.scaffold, run_index=i, limits=limits,
                max_turns=args.max_turns,
            )
            await _one(spec, results_root, args.timeout, args.no_eval,
                       not args.no_sandbox, container_spec)


def main() -> None:
    # 行缓冲：run 动辄跑几十分钟，默认块缓冲下进度全卡在缓冲区里，
    # 进程被外部 kill 时缓冲直接丢失——实测一次 25 分钟的 run 日志全空。
    import sys as _sys
    _sys.stdout.reconfigure(line_buffering=True)
    _sys.stderr.reconfigure(line_buffering=True)

    from .config import get
    # 命令行不传时回落到 config.toml，而非写死的默认值
    p = argparse.ArgumentParser(description="FDE-Bench runner")
    p.add_argument("--case", required=True)
    p.add_argument("--condition", default="Hidden",
                   help="Hidden / Interact / Interact-Req / Full，逗号分隔")
    p.add_argument("--model", default=get("agent", "model", "") or None,
                   help="不传则读 config.toml [agent].model")
    p.add_argument("--scaffold", default=get("agent", "scaffold", "opencode"))
    p.add_argument("--runs", type=int, default=1)
    p.add_argument("--run-index-start", type=int, default=1,
                   help="本次 runs 写入的起始 try 编号（默认 1）")
    p.add_argument("--timeout", type=int, default=int(get("agent", "timeout_s", 43200)))
    p.add_argument("--out", default="results")
    p.add_argument("--max-rounds", type=int, default=int(get("clarify", "max_rounds", 15)))
    p.add_argument("--max-questions", type=int, default=int(get("clarify", "max_questions", 4)))
    # 工具调用轮数上限。默认不限制：与时长一样只作失控兜底，而不是让所有模型
    # 跑到同一个轮数——强制对齐会把"高效做完"和"打转到上限"记成同一个数。
    # 只在能数到每次工具调用的后端（opencode serve）生效。
    p.add_argument("--max-turns", type=int,
                   default=(int(get("agent", "max_turns", 0)) or None),
                   help="工具调用次数上限，0/不传=不限制（仅 native 后端生效）")
    p.add_argument("--no-eval", action="store_true", help="只跑不评分")
    p.add_argument("--container", metavar="IMAGE", default=None,
                   help="在指定 docker 镜像内运行 agent（替代 bwrap 沙箱）")
    p.add_argument("--agent-bin", default=None,
                   help="挂进容器的 agent 可执行文件，默认自动探测 opencode")
    p.add_argument("--no-sandbox", action="store_true",
                   default=not get("sandbox", "enabled", True),
                   help="关闭文件隔离（agent 将能读到 gt.json 与 tests/，仅供调试）")
    asyncio.run(_main(p.parse_args()))


if __name__ == "__main__":
    main()
