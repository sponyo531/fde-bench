"""文本澄清通路（非 opencode 脚手架走这条）。

用例全部来自实测踩过的坑——这条通路的失败模式都是**静默空转**，
不报错、只烧预算，跑完才发现 15 轮问了同一批问题。

    python3 -m harness.tests.test_clarify_text
"""

from __future__ import annotations

import asyncio
import json
import sys
import tempfile
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

import harness.clarify.loop as cl                      # noqa: E402


class FakeSession:
    """按脚本回放 agent 的多轮回复。"""
    supports_native_clarify = False

    def __init__(self, replies, workspace=None, writes_at=None):
        self._replies = list(replies)
        self.sent: list[str] = []
        self._workspace = workspace or Path(tempfile.mkdtemp())
        (self._workspace / "data").mkdir(exist_ok=True)
        self._writes_at = writes_at        # 第几次 send 后产出交付物
        self.on_event = None

    async def send(self, message, **kw):
        self.sent.append(message)
        if self._writes_at is not None and len(self.sent) >= self._writes_at:
            (self._workspace / "routes.json").write_text("{}", encoding="utf-8")
        return self._replies.pop(0) if self._replies else "done"


class FakeAnswerer:
    def __init__(self):
        self.asked: list[str] = []

    def answer(self, q):
        self.asked.append(q)
        return f"答:{q[:10]}"


def _patch_detect(mapping):
    """把 ask_detect.detect 换成查表，避免测试打真实 LLM。"""
    cl.ask_detect.detect = lambda text, **kw: mapping.get(text.strip(), [])


def test_repeat_questions_stop_after_one_round():
    """agent 重复问同一批问题时必须止损，而不是答到撞轮数上限。"""
    Q = "速度多少？"
    _patch_detect({"ASK": [Q]})
    s = FakeSession(["ASK"] * 10, writes_at=99)
    a = FakeAnswerer()
    with tempfile.TemporaryDirectory() as td:
        log = Path(td) / "c.json"
        asyncio.run(cl.run_clarify(s, "P", a, max_rounds=15, log_path=log))
        d = json.loads(log.read_text(encoding="utf-8"))
    assert d["total_rounds"] == 1, f"应止于 1 轮，实际 {d['total_rounds']}"
    assert len(a.asked) == 1, f"同一问题只该答一次，实际 {len(a.asked)}"
    assert not d["hit_round_limit"]
    print("✓ 重复提问止损（1 轮而非 15 轮）")


def test_no_echo_of_agent_questions():
    """推它执行后若才提问，应作答而非把它自己的问题回灌。"""
    _patch_detect({"LATER": ["数据在哪？"]})
    s = FakeSession(["等我先看看数据", "LATER", "已完成"], writes_at=99)
    a = FakeAnswerer()
    asyncio.run(cl.run_clarify(s, "P", a, max_rounds=15))
    assert "LATER" not in s.sent, f"把 agent 的回复回灌了: {s.sent}"
    assert any(m.startswith("- 数据在哪？") for m in s.sent), s.sent
    print("✓ 不回灌 agent 自己的问题")


def test_push_when_neither_asks_nor_delivers():
    """既没提问也没产物 → 必须推它继续，否则整个 run 零产物。"""
    _patch_detect({})
    s = FakeSession(["我先检查数据", "已写入 routes.json"], writes_at=2)
    asyncio.run(cl.run_clarify(s, "P", FakeAnswerer(), max_rounds=15))
    assert cl._PROCEED in s.sent, f"未推它执行: {s.sent}"
    print("✓ 说完就停时推它继续")


def test_stop_when_delivered():
    """已有产物且不提问 → 正常结束，不该多推一轮。"""
    _patch_detect({})
    s = FakeSession(["已完成，见 routes.json"], writes_at=1)
    asyncio.run(cl.run_clarify(s, "P", FakeAnswerer(), max_rounds=15))
    assert cl._PROCEED not in s.sent, f"已交付却仍推: {s.sent}"
    print("✓ 已交付则正常收尾")


def test_normal_multi_round():
    """两轮不同的问题应各答一次，正常推进。"""
    _patch_detect({"Q1": ["速度？"], "Q2": ["单位？"]})
    s = FakeSession(["Q1", "Q2", "已完成"], writes_at=3)
    a = FakeAnswerer()
    with tempfile.TemporaryDirectory() as td:
        log = Path(td) / "c.json"
        asyncio.run(cl.run_clarify(s, "P", a, max_rounds=15, log_path=log))
        d = json.loads(log.read_text(encoding="utf-8"))
    assert d["total_rounds"] == 2, d["total_rounds"]
    assert a.asked == ["速度？", "单位？"], a.asked
    print("✓ 多轮不同问题正常推进")


if __name__ == "__main__":
    import harness.clarify.detect
    orig = harness.clarify.detect.detect
    try:
        for fn in [v for k, v in sorted(globals().items()) if k.startswith("test_")]:
            fn()
    finally:
        harness.clarify.detect.detect = orig
    print("\nall clarify-text tests passed")


class FakeNativeSession(FakeSession):
    """模拟 opencode serve：声明原生澄清，但本次 run agent 没调 question 工具。"""
    supports_native_clarify = True

    def __init__(self, *a, native_rounds=None, **kw):
        super().__init__(*a, **kw)
        self.clarify_rounds = list(native_rounds or [])
        self.hit_round_limit = False
        self.killed_reason = None

    def set_answerer(self, answerer, *, max_rounds=30, max_questions=0):
        self._answerer = answerer


def test_native_falls_back_to_text_when_agent_asks_in_prose():
    """006 事故：agent 在正文提问、无 question tool 事件 → 原生分支必须落回文本回路。

    修复前 _run_native 只 send 一次就 return，clarify.json 写空壳、答案从未回注。
    """
    _patch_detect({"请先回答：仓库在哪？我再开始建模": ["仓库在哪？"]})
    s = FakeNativeSession(["请先回答：仓库在哪？我再开始建模", "已完成"], writes_at=2)
    a = FakeAnswerer()
    with tempfile.TemporaryDirectory() as td:
        log = Path(td) / "c.json"
        asyncio.run(cl.run_clarify(s, "P", a, max_rounds=15, log_path=log))
        d = json.loads(log.read_text(encoding="utf-8"))
    assert a.asked == ["仓库在哪？"], a.asked
    assert any(m.startswith("- 仓库在哪？") for m in s.sent), s.sent   # 答案回灌了
    assert d["total_rounds"] == 1 and d["rounds"][0]["channel"] == "text"
    print("✓ 原生分支：正文提问落回文本回路并回注答案")


def test_native_rounds_and_text_rounds_merge_in_order():
    """agent 先用 tool 问了 2 轮，又在正文补问 1 个：三轮合并、编号连续、各带 channel。"""
    _patch_detect({"另外，单位是公斤吗？": ["单位是公斤吗？"]})
    native = [{"turn": 1, "questions": [{"question": "q1"}]},
              {"turn": 2, "questions": [{"question": "q2"}]}]
    s = FakeNativeSession(["另外，单位是公斤吗？", "done"], native_rounds=native, writes_at=2)
    with tempfile.TemporaryDirectory() as td:
        log = Path(td) / "c.json"
        asyncio.run(cl.run_clarify(s, "P", FakeAnswerer(), max_rounds=15, log_path=log))
        d = json.loads(log.read_text(encoding="utf-8"))
    assert [r["turn"] for r in d["rounds"]] == [1, 2, 3]
    assert [r["channel"] for r in d["rounds"]] == ["native", "native", "text"]
    assert d["total_questions"] == 3
    print("✓ 原生 + 文本轮次合并且编号连续")


def test_native_delivered_via_tool_exits_without_extra_sends():
    """正常原生路径（tool 问完就交付）不能被文本兜底多推一把。"""
    _patch_detect({})
    s = FakeNativeSession(["交付完成，见 routes.json"], writes_at=1)
    asyncio.run(cl.run_clarify(s, "P", FakeAnswerer(), max_rounds=15))
    assert s.sent == ["P"], s.sent
    print("✓ 原生正常交付：只发一次 prompt")


def test_killed_session_is_never_sent_again():
    """看门狗击杀后不得再 send（会挂到外层超时）。"""
    _patch_detect({})
    s = FakeNativeSession([""], writes_at=99)
    s.killed_reason = "max_runtime"
    asyncio.run(cl.run_clarify(s, "P", FakeAnswerer(), max_rounds=15))
    assert s.sent == ["P"], s.sent
    print("✓ 被击杀的会话不再收到 PROCEED")


def test_clarify_log_is_written_incrementally():
    """每轮作答后 clarify.json 就要在盘上——求解阶段超时被掐不能丢掉澄清记录。"""
    _patch_detect({"Q1": ["a？"], "Q2": ["b？"]})

    class HangingSession(FakeSession):
        async def send(self, message, **kw):
            if len(self.sent) == 2:          # 第三次 send（求解阶段）模拟被掐
                raise asyncio.CancelledError
            return await super().send(message, **kw)

    s = HangingSession(["Q1", "Q2"], writes_at=99)
    with tempfile.TemporaryDirectory() as td:
        log = Path(td) / "c.json"
        try:
            asyncio.run(cl.run_clarify(s, "P", FakeAnswerer(), max_rounds=15, log_path=log))
        except asyncio.CancelledError:
            pass
        assert log.is_file(), "超时前应已落盘"
        d = json.loads(log.read_text(encoding="utf-8"))
    assert d["total_rounds"] == 2 and d["total_questions"] == 2
    print("✓ clarify.json 增量落盘，超时不丢")


def test_native_session_flushes_log_per_round_before_send_returns():
    """原生路径 send 期间被掐：后端每追加一轮就通过 _clarify_flush 落盘。"""
    _patch_detect({})

    class NativeThatAsksDuringSend(FakeNativeSession):
        async def send(self, message, **kw):
            self.sent.append(message)
            # 模拟 opencode/OpenHands 在同一次 send 内部作答并回调
            self.clarify_rounds.append({"turn": 1, "questions": [{"question": "q?"}]})
            self._clarify_flush()
            raise asyncio.CancelledError          # 然后被掐

    s = NativeThatAsksDuringSend([], writes_at=99)
    with tempfile.TemporaryDirectory() as td:
        log = Path(td) / "c.json"
        try:
            asyncio.run(cl.run_clarify(s, "P", FakeAnswerer(), max_rounds=15, log_path=log))
        except asyncio.CancelledError:
            pass
        d = json.loads(log.read_text(encoding="utf-8"))
    assert d["total_rounds"] == 1 and d["rounds"][0]["channel"] == "native"


def test_pushes_up_to_three_times_when_agent_keeps_announcing():
    """kimi-k3 模式：每轮只说"我将检查…"就停。推 3 次，第 3 次落产物即结束。"""
    _patch_detect({})
    s = FakeSession(["I will inspect the files.", "I will check the data.", "Working on it.", "done"],
                    writes_at=4)               # 第 4 次 send（第 3 次推）后才落产物
    asyncio.run(cl.run_clarify(s, "P", FakeAnswerer(), max_rounds=15))
    assert s.sent == ["P", cl._PROCEED, cl._PROCEED, cl._PROCEED], s.sent


def test_push_stops_early_once_deliverable_appears():
    _patch_detect({})
    s = FakeSession(["I will inspect the files.", "wrote it"], writes_at=2)
    asyncio.run(cl.run_clarify(s, "P", FakeAnswerer(), max_rounds=15))
    assert s.sent == ["P", cl._PROCEED], s.sent


def test_nested_deliverable_is_recognized_without_extra_push():
    """最终扫描承认 output/result.json，澄清循环也必须采用相同口径。"""
    _patch_detect({})
    ws = Path(tempfile.mkdtemp())
    (ws / "output").mkdir()
    (ws / "output" / "result.json").write_text("{}", encoding="utf-8")
    s = FakeSession(["done"], workspace=ws)
    asyncio.run(cl.run_clarify(s, "P", FakeAnswerer(), max_rounds=15))
    assert s.sent == ["P"], s.sent


def test_text_max_questions_caps_answerer_calls_but_keeps_asked_questions():
    _patch_detect({"ASK": ["q1?", "q2?", "q3?"]})
    s = FakeSession(["ASK", "done"], writes_at=2)
    a = FakeAnswerer()
    with tempfile.TemporaryDirectory() as td:
        log = Path(td) / "c.json"
        asyncio.run(cl.run_clarify(
            s, "P", a, max_rounds=15, max_questions=2, log_path=log))
        data = json.loads(log.read_text(encoding="utf-8"))
    assert a.asked == ["q1?", "q2?"]
    assert data["total_questions"] == 3
    assert data["rounds"][0]["dropped_over_budget"] == 1
    assert data["hit_question_limit"] is True
    assert "No further questions" in s.sent[1]


class LedgerSession(FakeSession):
    """带多步账本的文本后端：每次 send 记若干条调用。"""
    def __init__(self, replies, steps_per_send, **kw):
        super().__init__(replies, **kw)
        from harness.backends.usage import UsageLedger
        self._ledger = UsageLedger(); self._steps = list(steps_per_send)
    async def send(self, message, **kw):
        n = self._steps.pop(0) if self._steps else 1
        for _ in range(n):
            self._ledger.record({"input_tokens": 10, "output_tokens": 1})
        return await super().send(message, **kw)
    def usage_call_count(self): return self._ledger.call_count
    def label_usage_from(self, start, phase): self._ledger.label_from(start, phase)
    def set_usage_phase(self, p): self._ledger.set_phase(p)
    def relabel_last_usage_phase(self, p): self._ledger.relabel_last(p)


def test_phase_labels_cover_every_call_of_each_send():
    """首轮 3 步提问 → clarify；第二轮 94 步求解（无回复即被掐）→ solve，全部 94 步。"""
    _patch_detect({"Q": ["q?"]})
    s = LedgerSession(["Q", "done"], steps_per_send=[3, 94], writes_at=2)
    asyncio.run(cl.run_clarify(s, "P", FakeAnswerer(), max_rounds=15))
    ph = s._ledger.snapshot()["phase"]
    assert ph["clarify"]["steps"] == 3 and ph["solve"]["steps"] == 94


def test_first_turn_without_questions_is_all_solve():
    _patch_detect({})
    s = LedgerSession(["working"], steps_per_send=[7], writes_at=1)
    asyncio.run(cl.run_clarify(s, "P", FakeAnswerer(), max_rounds=15))
    ph = s._ledger.snapshot()["phase"]
    assert ph.get("clarify") is None and ph["solve"]["steps"] == 7    # 修前只有最后 1 步是 solve
