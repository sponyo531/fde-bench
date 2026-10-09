"""C 条件：澄清 → 求解。

两条通路，按后端能力自动选择——但**不是二选一**：

原生通路（opencode serve，有 question tool）
    框架在旁路拦截 tool 调用、调 answerer 作答、经 API 回注，一次 send 搞定。
    实测（2026-09-02，7 模型）6/7 会主动用这个工具；但少数模型/少数 run 会
    改在回复正文里自然语言提问（006 案例），此时原生通路收不到任何事件。
    所以原生通路拿到回复后**必须再过一遍文本判定**，有问句就落回文本通路
    继续多轮——"原生优先 + 文本兜底"。修复前这里直接 return，正文提问整轮
    丢弃，run 还记成 status=ok / ask_f1=0，与"模型不会问"在数据上不可区分。

文本通路（Claude Code / Codex / Gemini / Kimi 等 headless 脚手架）
    通用原生通道是「多轮对话」——agent 在回复正文里用自然语言提问，框架在
    下一轮发 user message 作答。实测（2026-09-02 逐个复核）headless 模式下：
    Claude Code 不注册 AskUserQuestion、Codex 无 ask 类工具、Gemini 非交互下
    不暴露 ask_user、Kimi -p 即 auto 模式并告知模型 AskUserQuestion 已禁用。
    纯文本提问是这四家唯一可用通道。

    用 ask_detect（语义判定）替代旧的 <clarify> 正则标记：C.md / CF.md
    早已是纯自然语言 prompt，agent 不会输出标记；旧正则一轮就退出，
    文本通路实际上是死的。

设计说明：

轮数上限只作兜底（默认 30），不作约束。同类工作（HIL-Bench、ClarEval、
Ambig-SWE）均不设硬上限，而是用指标惩罚过度提问。
"""

from __future__ import annotations

import json
from datetime import datetime, timezone
import time
from pathlib import Path

from . import detect as ask_detect


_PROCEED = (
    "Understood. Please now carry out the work and write the deliverable files "
    "into the working directory."
)
# 「说完就停」最多推几次。Codex 一次就动；kimi-k3 实测读三遍数据、说一句
# "I will inspect the workspace files…" 就停，推一次再停一次，run 12 秒 status=ok
# 零产物。每次推之间 agent 若开口提问就转入作答，若落了产物就正常结束。
_MAX_PUSHES = 3


def _write_log(log_path: Path | None, rounds: list[dict], *, hit_limit: bool,
               hit_question_limit: bool = False,
               clarify_secs: float = 0.0) -> None:
    if log_path is None:
        return
    log_path.write_text(
        json.dumps({
            "total_rounds": len(rounds),
            "total_questions": sum(len(r["questions"]) for r in rounds),
            "total_tokens": sum((r.get("tokens_this_turn") or 0) for r in rounds),
            "clarify_secs": round(clarify_secs, 1),
            # harness 侧开销（detect + answerer）。原生通路里 agent 阻塞等 answerer，
            # 这段时间混在它的墙钟里；跨脚手架比澄清耗时前要减掉
            "harness_secs": round(sum((r.get("harness_secs") or 0) for r in rounds), 1),
            "hit_round_limit": hit_limit,
            "hit_question_limit": hit_question_limit,
            "rounds": rounds,
        }, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )


async def run_clarify(
    session,
    prompt: str,
    answerer,
    *,
    max_rounds: int = 30,
    max_questions: int = 0,
    log_path: Path | None = None,
) -> tuple[str, list[dict]]:
    """跑澄清 → 求解全程。返回 (agent 最终回复, 澄清轮次记录)。"""
    if getattr(session, "supports_native_clarify", False):
        return await _run_native(session, prompt, answerer, max_rounds,
                                 max_questions, log_path)
    return await _run_text(session, prompt, answerer, max_rounds,
                           max_questions, log_path)


async def _run_native(session, prompt, answerer, max_rounds, max_questions, log_path):
    """原生通路：提问由脚手架自己的 tool 发起，框架只在旁路作答。

    第一次 send() 内部就可能完成提问、作答、继续执行全过程（由后端的
    question 轮询线程处理）。但 agent 也可能根本没调 tool、而是在正文里问——
    所以拿到回复后交给文本循环续跑：它会判定是否在提问、逐条作答回灌、
    推「说完就停」的 agent 去执行、拦重复提问。agent 若在后续轮次又改用
    tool，后端仍会在旁路接住并追加到 session.clarify_rounds。

    两个通道的轮次合并进同一份 clarify.json，各带 channel 标记；文本轮次的
    turn 编号接在原生轮次之后，保证全局顺序。
    """
    session.set_answerer(answerer, max_rounds=max_rounds,
                         max_questions=max_questions)
    t0 = time.time()
    clarify_secs = 0.0

    def _merged(text_rounds: list[dict]) -> list[dict]:
        native_rounds = [{**r, "channel": "native"}
                         for r in getattr(session, "clarify_rounds", [])]
        return native_rounds + text_rounds

    def _flush(text_rounds: list[dict], hit: bool) -> None:
        merged = _merged(text_rounds)
        _write_log(log_path, merged,
                   hit_limit=bool(getattr(session, "hit_round_limit", False)) or hit,
                   hit_question_limit=(
                       bool(getattr(session, "hit_question_limit", False))
                       or any((r.get("dropped_over_budget") or 0) > 0 for r in merged)
                   ),
                   clarify_secs=clarify_secs)

    # 原生路径整个 run 常常是**一次** send（OpenHands 全程、opencode 用 tool 时）：
    # 后端每追加一轮就回调落盘，否则 send 期间被超时掐掉 → clarify.json 不存在。
    # 实测 OpenHands 11 问已作答、被掐后目录里没有 clarify.json。
    session._clarify_flush = lambda: _flush([], False)
    response = await session.send(prompt)
    clarify_secs = time.time() - t0
    native_before = len(getattr(session, "clarify_rounds", []))
    native_questions = sum(
        len(r.get("questions") or []) - int(r.get("dropped_over_budget") or 0)
        for r in getattr(session, "clarify_rounds", [])
    )
    _flush([], False)
    # If native question rounds were handled and the final response contains
    # no question mark, it is already a normal completion.  Avoid sending that
    # text through the semantic judge: it wastes judge calls and can make a
    # detector implementation mistake ordinary completion text for a question.
    if native_before and not any(mark in (response or "") for mark in "?？"):
        text_rounds, text_hit_limit = [], False
    else:
        response, text_rounds, text_hit_limit = await _text_loop(
            session, prompt, answerer,
            max_rounds=max(0, max_rounds - native_before),
            max_questions=max_questions,
            answered_offset=native_questions,
            first_response=response,
            turn_offset=native_before,
            on_round=_flush,
        )
    rounds = _merged(text_rounds)
    _flush(text_rounds, text_hit_limit)
    return response, rounds


def _has_deliverable(session) -> bool:
    """workspace 里有没有 agent 交付的文件（排除输入数据与脚手架自留物）。

    用于区分「说完就停」和「已经交付」——前者必须推它继续，否则整个 run
    零产物；后者是正常收尾。
    """
    # OpenHands 用 session.workspace，其余后端用 _workspace；只认一个会让 OpenHands
    # 永远返回 True（"拿不到"）→ 从不推执行
    ws = getattr(session, "_workspace", None) or getattr(session, "workspace", None)
    if ws is None:
        return True                # 拿不到就别乱推，交给上层超时兜底
    from pathlib import Path
    ws = Path(ws)
    if not ws.is_dir():
        return True
    # 与最终 artifact_status 共用同一份递归扫描规则。否则 agent 已写好
    # output/result.json 时，这里只看根目录仍会误判为零交付并额外催促三轮。
    from ..run import workspace_snapshot
    return bool(workspace_snapshot(ws))


def _dead(session) -> bool:
    """不能再 send 的会话：看门狗已击杀（进程 abort，再发只会挂到外层超时），
    或后端本身一次 send 跑完全程、不支持续接（OpenHands，第二次 send 会 raise）。"""
    return bool(getattr(session, "killed_reason", None)) or \
        not getattr(session, "supports_followup", True)


def _emit_repeat(session) -> None:
    cb = getattr(session, "on_event", None)
    if cb:
        from ..backends.base import AgentEvent
        cb(AgentEvent(type="info",
                      content="[clarify] agent 重复提出同一批问题，停止作答并推它执行"))


async def _run_text(session, prompt, answerer, max_rounds, max_questions, log_path):
    """文本通路：语义判定 agent 是否在提问 + 回灌。

    与旧 <clarify> 正则的区别：
      - agent 不需要学任何格式——自然提问即可（实测 Claude Code / Codex 都是
        纯自然语言问，没有任何标记）
      - 误判代价不对称：把「正在交付」误判成「在提问」会污染上下文（更贵）；
        把「在提问」误判成「在交付」只是少一轮澄清（agent 通常会再问一次）。
        所以 ask_detect 在无问号时短路（不花 LLM），有问号时才判定，坏 JSON
        时保守判「没在提问」。
    """
    def _flush(rounds: list[dict], hit: bool) -> None:
        _write_log(
            log_path, rounds, hit_limit=hit,
            hit_question_limit=any(
                (r.get("dropped_over_budget") or 0) > 0 for r in rounds),
        )

    response, rounds, hit_limit = await _text_loop(session, prompt, answerer,
                                                   max_rounds=max_rounds,
                                                   max_questions=max_questions,
                                                   on_round=_flush)
    _flush(rounds, hit_limit)
    return response, rounds


async def _text_loop(session, prompt, answerer, *, max_rounds: int,
                     max_questions: int = 0,
                     answered_offset: int = 0,
                     first_response: str | None = None, turn_offset: int = 0,
                     on_round=None):
    """文本多轮的核心。返回 (最终回复, 文本轮次记录, 是否撞轮数上限)。

    first_response 非空时表示 prompt 已由调用方发过、这是它的回复（原生通路
    兜底进来的情形），循环从判定这条回复开始而不是重发 prompt。

    on_round(rounds, hit_limit) 在每轮作答后调用，用于**增量落盘** clarify.json：
    澄清之后是可能跑几小时的求解，run 被外层超时掐掉时若只在循环末尾写一次，
    已经发生的澄清记录会整份丢失（实测 1320s 超时的 run 有产物、无 clarify.json）。
    """
    rounds: list[dict] = []
    message = prompt
    response = first_response or ""
    asked_last_turn = False
    seen_signatures: set[tuple] = set()
    set_phase = getattr(session, "set_usage_phase", None)
    relabel_phase = getattr(session, "relabel_last_usage_phase", None)
    if callable(set_phase):
        set_phase("clarify")

    # first_response 占用第 0 轮：它不需要 send，但要走完整的判定/推执行逻辑。
    # 之后每轮先 send 再判定。max_rounds=0 且有 first_response 时也至少判一次。
    turns = range(0 if first_response is not None else 1, max_rounds + 1)
    exhausted = True
    mark = session.usage_call_count() if hasattr(session, "usage_call_count") else 0
    for turn in turns:
        if _dead(session):
            exhausted = False
            break
        if turn > 0:
            mark = session.usage_call_count() if hasattr(session, "usage_call_count") else 0
            response = await session.send(message)

        # 语义判定：agent 在提问吗？不需要 <clarify> 标记
        t_det = time.time()
        questions_text = ask_detect.detect(response)
        detect_secs = time.time() - t_det

        # 阶段归属按"这轮回复是不是提问"给**这次 send 的全部调用**打标签。
        # 此前只在看到非提问回复时 relabel 最后一条：dsh 一次求解 94 步、被超时掐掉
        # 没有回复 → 94 步全挂在 clarify 下（pod 实测 clarify_tokens 226k / solve None）；
        # 首轮不提问直接干活时也只有最后一步是 solve。
        if hasattr(session, "label_usage_from"):
            session.label_usage_from(mark, "clarify" if questions_text else "solve")

        if not questions_text:
            if callable(relabel_phase):
                relabel_phase("solve")
            # 没提问，但也可能没干活——三种情形要分开：
            #   1. 刚回答完提问就停：agent 认为信息够了但尚未动手，推它执行
            #   2. 既没提问也没产物：agent 在"宣告下一步"而非交付（实测 Codex
            #      会先回一句"我先检查数据，看完再列出需确认的口径"就结束本轮），
            #      此时退出会让整个 run 零产物，必须推它继续
            #   3. 有产物：正常交付，结束
            # 只看有没有产物：答完问题 agent 直接交付了就结束，不再多推一轮
            # （原条件 asked_last_turn 会在已交付时白推一次 _PROCEED，多烧一轮）
            if not _has_deliverable(session) and not _dead(session):
                if callable(set_phase):
                    set_phase("solve")
                pushed: list[str] = []
                for _ in range(_MAX_PUSHES):
                    response = await session.send(_PROCEED)
                    pushed = ask_detect.detect(response)
                    if pushed or _has_deliverable(session) or _dead(session):
                        break
                if pushed:
                    # 推了一把之后才开口提问（Codex 的典型节奏）——就地按提问处理。
                    # 不能 `message = response; continue`：那会把 agent 自己的问题
                    # 当成用户消息回灌，等于让它跟自己对话。
                    questions_text = pushed
                else:
                    exhausted = False
                    break
            else:
                exhausted = False
                break

        # 重复提问护栏：同一批问题再次出现，说明 agent 没在消化答案（实测 Codex
        # 因读到过期文件把同一批问题问了 15 轮、194 条）。此时再答一遍没有意义，
        # 直接推它执行并收尾。
        sig = tuple(sorted(q.strip() for q in questions_text))
        if sig in seen_signatures:
            _emit_repeat(session)
            if callable(set_phase):
                set_phase("solve")
            if not _dead(session):
                response = await session.send(_PROCEED)
            exhausted = False
            break
        seen_signatures.add(sig)

        # 逐条作答（计时：answerer 是 harness 侧的开销，不是 agent 的）。
        # max_questions=0 表示不限；非零时按“问题条数”而不是轮数计全局预算。
        answered_so_far = answered_offset + sum(
            len(r.get("answers") or []) for r in rounds)
        remaining = (None if max_questions == 0
                     else max(0, max_questions - answered_so_far))
        allowed_questions = (questions_text if remaining is None
                             else questions_text[:remaining])
        dropped = len(questions_text) - len(allowed_questions)
        answers = []
        t_ans = time.time()
        for q in allowed_questions:
            a = answerer.answer(q)
            answers.append({"question": q, "answer": a})
        harness_secs = round(time.time() - t_ans + detect_secs, 1)

        rounds.append({
            "turn": turn_offset + len(rounds) + 1,
            "channel": "text",
            "harness_secs": harness_secs,      # detect + answerer 用时，不含 agent
            "asked_at": datetime.now(timezone.utc).isoformat(),
            # 澄清阶段的结束锚点（epoch 毫秒，与 opencode SQLite 的 time_created 同钟）：
            # usage._phase_split 用它切 clarify/solve。native 的 question tool 走 DB
            # 时间戳；正文提问没有 tool 事件，只能由这里给锚点，否则 opencode 上
            # 正文提问的 run 澄清 token/秒全被记成 0（实测 9 问 → clarify_secs 0.0）。
            "answered_at_ms": int(time.time() * 1000),
            # 所有实际问句都保留用于 ask 指标；answers 只含预算内得到业务答复的问句。
            "questions": [{"question": q} for q in questions_text],
            "answers": answers,
            "dropped_over_budget": dropped,
            "tokens_this_turn": getattr(session, "_last_call_tokens", None),
        })
        asked_last_turn = True
        if on_round is not None:
            on_round(list(rounds), len(rounds) >= max_rounds)

        body = "\n".join(f"- {a['question']}\n  {a['answer']}" for a in answers)
        question_limit_hit = dropped > 0 or (
            max_questions > 0 and answered_so_far + len(answers) >= max_questions)
        if question_limit_hit:
            body += ("\n\n(No further questions can be answered. "
                     "Proceed with your best judgement.)")
        elif turn >= max_rounds:
            body += "\n\n(No further questions can be answered. Proceed with your best judgement.)"
        message = body
    if exhausted and not _dead(session):
        # 用满全部轮次仍在提问，推它收尾
        if callable(set_phase):
            set_phase("solve")
        response = await session.send(_PROCEED)

    return response, rounds, len(rounds) >= max_rounds and max_rounds > 0
