"""opencode serve 后端的离线单元测试。

不起真进程、不打真 provider——用一个假的 HTTP 层驱动 `OpenCodeServeSession`，
校验的是**框架侧的逻辑**：提问怎么被拦截、答案怎么回注、轮数上限怎么兜底、
part 怎么转成事件、无 answerer 时权限是不是真关掉了。

这些正是过去只有跑完一次真实验才会暴露的坑（连发 15 轮、事件丢失、
question 通道在 R 条件下没关），放进单测就能秒级发现。

    python3 -m harness.tests.test_serve_backend
"""

from __future__ import annotations

import asyncio
import json
import sys
import urllib.error
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from harness.backends.opencode.serve import OpenCodeServeSession  # noqa: E402
from harness.clarify.loop import run_clarify                          # noqa: E402


class FakeServer:
    """按脚本回放 opencode serve 的 HTTP 行为。

    questions: 待发的提问批次列表，每批一个 QuestionRequest。poller 每轮取一批，
    模拟 agent 在执行途中陆续提问。
    """

    def __init__(self, question_batches, parts=None, reply_text="done"):
        self._batches = list(question_batches)
        self._total = len(self._batches)
        self._parts = parts or []
        self._reply_text = reply_text
        self.replies: list[tuple[str, list]] = []
        self.rejects: list[str] = []
        self.created_permission = None
        self.aborted = False

    @property
    def _handled(self) -> int:
        return len(self.replies) + len(self.rejects)

    def __call__(self, method, path, body=None, timeout=120):
        if path == "/global/health":
            return {"healthy": True}
        if method == "POST" and path == "/session":
            self.created_permission = (body or {}).get("permission")
            return {"id": "ses_test"}
        if method == "GET" and path == "/question":
            return [self._batches.pop(0)] if self._batches else []
        if method == "GET" and path == "/permission":
            return []
        if path.startswith("/question/") and path.endswith("/reply"):
            self.replies.append((path.split("/")[2], (body or {})["answers"]))
            return True
        if path.startswith("/question/") and path.endswith("/reject"):
            self.rejects.append(path.split("/")[2])
            return True
        if path.endswith("/abort"):
            self.aborted = True
            return True
        if method == "GET" and path.endswith("/message"):
            return self._parts
        if method == "POST" and path.endswith("/message"):
            # 阻塞直到所有提问都被**处理完**（reply 或 reject），模拟"agent 边问边干"。
            # 只等 _batches 取空是不够的：最后一批刚被 poller 取走、尚未处理时
            # 就返回，会让断言看到少一次 reply/reject。
            import time
            for _ in range(500):
                if self._handled >= self._total:
                    break
                time.sleep(0.01)
            return {"parts": [{"type": "text", "text": self._reply_text}]}
        raise AssertionError(f"unexpected {method} {path}")


def _session(fake, answerer=None, max_rounds=15):
    s = OpenCodeServeSession(Path("/tmp/does-not-matter"))
    s._api = fake
    # 假装 serve 已就绪：本文件测的是框架侧逻辑，不该起真进程。
    # 漏设这一行时，每跑一次单测就会在机器上留下几个 opencode serve 常驻进程。
    s._launched = True
    s.on_event = lambda e: None
    if answerer is not None:
        s.set_answerer(answerer, max_rounds=max_rounds)
    return s


class ScriptedAnswerer:
    def __init__(self, reply="按 CSV 口径，字段 order_id,amount"):
        self.asked: list[str] = []
        self._reply = reply

    def answer(self, question: str) -> str:
        self.asked.append(question)
        return self._reply


def _q(qid, *questions):
    return {
        "id": qid, "sessionID": "ses_test",
        "questions": [{"question": q, "header": q[:20],
                       "options": [{"label": "A", "description": ""}]}
                      for q in questions],
    }


# ── 用例 ─────────────────────────────────────────────────────────────────────

def test_native_clarify_roundtrip():
    """agent 分两次提问 → 每次都被作答回注 → 记录进 clarify_rounds。"""
    fake = FakeServer([_q("que_1", "数据在哪？", "字段口径？"), _q("que_2", "输出格式？")])
    ans = ScriptedAnswerer()
    sess = _session(fake, ans)
    text, rounds = asyncio.run(run_clarify(sess, "做个报告", ans, max_rounds=15))

    assert text == "done", text
    assert len(rounds) == 2, rounds
    assert [len(r["questions"]) for r in rounds] == [2, 1]
    assert ans.asked == ["数据在哪？", "字段口径？", "输出格式？"], ans.asked
    # 回注格式：answers 与 questions 一一对应，每项是字符串数组
    assert [len(a) for _, a in fake.replies] == [2, 1]
    assert fake.replies[0][1][0] == [ans._reply]
    print("✓ native clarify roundtrip")


def test_round_limit_rejects():
    """撞上限后 reject 而非继续作答——防止空转烧预算。"""
    fake = FakeServer([_q(f"que_{i}", f"问题{i}") for i in range(5)])
    ans = ScriptedAnswerer()
    sess = _session(fake, ans, max_rounds=2)
    asyncio.run(run_clarify(sess, "做个报告", ans, max_rounds=2))

    assert len(fake.replies) == 2, fake.replies
    assert len(fake.rejects) == 3, fake.rejects
    assert sess.hit_round_limit is True
    print("✓ round limit rejects")


def test_question_limit_caps_answers_inside_one_native_request():
    fake = FakeServer([_q("que_1", "q1?", "q2?", "q3?")])
    ans = ScriptedAnswerer()
    sess = _session(fake, ans)
    _, rounds = asyncio.run(run_clarify(
        sess, "做个报告", ans, max_rounds=15, max_questions=2))

    assert ans.asked == ["q1?", "q2?"]
    assert len(fake.replies[0][1]) == 3  # 每题仍回注，避免 question tool 卡死
    assert "No further information" in fake.replies[0][1][2][0]
    assert rounds[0]["dropped_over_budget"] == 1
    assert sess.hit_question_limit is True


def test_question_denied_without_answerer():
    """R / F 条件（无 answerer）必须真的把 question 权限关掉。"""
    fake = FakeServer([])
    sess = _session(fake)
    sess._open_session()          # 只建会话，不起真进程
    perms = {p["permission"]: p["action"] for p in fake.created_permission}
    assert perms["question"] == "deny", perms
    assert perms["webfetch"] == "deny", perms
    assert perms["external_directory"] == "deny", "不能留作 ask 在无头模式永久等待"

    fake2 = FakeServer([])
    sess2 = _session(fake2, ScriptedAnswerer())
    sess2._open_session()
    perms2 = {p["permission"]: p["action"] for p in fake2.created_permission}
    assert perms2["question"] == "ask", perms2
    assert perms2["external_directory"] == "deny", perms2

    fake3 = FakeServer([])
    sess3 = _session(fake3)
    sess3._config = type("C", (), {"allow_external_directory": True})()
    sess3._open_session()
    perms3 = {p["permission"]: p["action"] for p in fake3.created_permission}
    assert perms3["external_directory"] == "allow", perms3
    print("✓ question permission gated by answerer")


class PermissionServer(FakeServer):
    def __init__(self, *, fail_reply=False):
        super().__init__([])
        self.permissions = [
            {"id": "per_030", "sessionID": "ses_test", "permission": "external_directory",
             "patterns": ["/proc/*"]},
            {"id": "per_049", "sessionID": "ses_test", "permission": "external_directory",
             "patterns": ["/wrong-case/try3/*"]},
            {"id": "per_other", "sessionID": "ses_other", "permission": "bash"},
        ]
        self.permission_replies = []
        self.fail_reply = fail_reply

    def __call__(self, method, path, body=None, timeout=120):
        if method == "GET" and path == "/permission":
            return list(self.permissions)
        if method == "POST" and path.startswith("/permission/"):
            if self.fail_reply:
                raise TimeoutError("reply timed out")
            self.permission_replies.append((path, body))
            rid = path.split("/")[2]
            self.permissions = [r for r in self.permissions if r["id"] != rid]
            return True
        return super().__call__(method, path, body, timeout)


def test_permissions_rejected_only_for_own_session():
    fake = PermissionServer()
    ans = ScriptedAnswerer()
    sess = _session(fake, ans)
    sess._open_session()
    assert sess._reject_pending_permissions() is True
    assert fake.permission_replies == [
        ("/permission/per_030/reply", {"reply": "reject"}),
        ("/permission/per_049/reply", {"reply": "reject"}),
    ]
    assert sess._reject_pending_permissions() is False
    assert [r["id"] for r in fake.permissions] == ["per_other"]
    assert ans.asked == []
    assert sess.killed_reason is None


def test_permission_reply_failure_retried_and_diagnostics_deduplicated():
    fake = PermissionServer(fail_reply=True)
    sess = _session(fake)
    sess._open_session()
    events = []
    sess.on_event = events.append
    assert sess._reject_pending_permissions() is True
    count = len(events)
    assert sess._reject_pending_permissions() is True
    assert len(events) == count
    fake.fail_reply = False
    assert sess._reject_pending_permissions() is True
    assert len(fake.permission_replies) == 2


def test_permission_wait_does_not_keep_idle_watchdog_alive():
    import threading
    import time
    fake = PermissionServer(fail_reply=True)
    fake._parts = [{"info": {"role": "assistant"}, "parts": [
        {"id": "p_wait", "type": "tool", "tool": "bash",
         "state": {"status": "running", "input": {"timeout": 180000}}},
    ]}]
    sess = _session(fake)
    sess._open_session()
    done = threading.Event()
    sess._stop_agent = lambda stop: stop.set()
    sess._watch(done, [time.time()], [], None, max_runtime=10, idle_timeout=1)
    assert sess.killed_reason == "idle"


def test_running_tool_without_permission_wait_keeps_heartbeat():
    import threading
    import time
    fake = FakeServer([], parts=[{"info": {"role": "assistant"}, "parts": [
        {"id": "p_work", "type": "tool", "tool": "bash",
         "state": {"status": "running", "input": {}}},
    ]}])
    sess = _session(fake)
    sess._open_session()
    done = threading.Event()
    sess._stop_agent = lambda stop: stop.set()
    sess._watch(done, [time.time()], [], None, max_runtime=2, idle_timeout=1)
    assert sess.killed_reason == "max_runtime", "real work must not be classified idle"


def test_parts_become_events():
    """tool part 的 running/completed 两态各产一个事件，且不重复。"""
    parts = [{
        "info": {"role": "assistant", "id": "m1"},
        "parts": [
            {"id": "p1", "type": "text", "text": "开始"},
            {"id": "p2", "type": "tool", "tool": "bash",
             "state": {"status": "running", "input": {"command": "ls"}}},
        ],
    }]
    fake = FakeServer([], parts=parts)
    sess = _session(fake)
    sess.session_id = "ses_test"
    events = []
    sess.on_event = events.append

    assert sess._drain_parts(set()) is True
    assert sess._drain_parts(set()) is False, "同样的 part 不该重复产事件"
    assert sess._active_tool_ids == {"p2"}, "running tool must keep an idle-watchdog heartbeat"
    kinds = [(e.type, e.tool or e.content) for e in events]
    assert ("text", "开始") in kinds, kinds
    assert ("tool_call", "bash") in kinds, kinds

    # 同一 part 转 completed 后应视为新状态（供 tool_result 使用），不被去重吃掉
    parts[0]["parts"][1]["state"] = {"status": "error", "error": "boom"}
    assert sess._drain_parts(set()) is True
    assert sess._active_tool_ids == set(), "completed tool must stop the heartbeat"
    assert any(e.type == "tool_result" and e.is_error for e in events), events
    print("✓ parts → events, dedup by (id, status)")


def test_provider_error_is_exposed_and_redacted():
    """assistant info.error 不能退化成无信息量 UnknownError，也不能泄露凭证。"""
    parts = [{
        "info": {
            "role": "assistant", "id": "m-error",
            "error": {"name": "APIError", "data": {
                "statusCode": 502,
                "message": "gateway failed with sk-super-secret-token",
            }},
        },
        "parts": [],
    }]
    fake = FakeServer([], parts=parts)
    sess = _session(fake)
    sess.session_id = "ses_test"
    events = []
    sess.on_event = events.append

    assert sess._drain_parts(set()) is True
    assert sess._drain_parts(set()) is False
    messages = [e.content for e in events if e.type == "info"]
    assert any('"status": 502' in m and "APIError" in m for m in messages)
    assert all("sk-super-secret-token" not in m for m in messages)
    assert any("sk-[REDACTED]" in m for m in messages)


def test_stop_on_tools_aborts():
    """stop_on_tools 命中时立即 abort（B2 在执行开始前退出）。"""
    parts = [{
        "info": {"role": "assistant", "id": "m1"},
        "parts": [{"id": "p1", "type": "tool", "tool": "write",
                   "state": {"status": "running", "input": {}}}],
    }]
    fake = FakeServer([], parts=parts)
    sess = _session(fake)
    sess.session_id = "ses_test"
    sess._drain_parts({"write"})
    assert fake.aborted is True
    assert sess.killed_reason == "early_stop"
    print("✓ stop_on_tools aborts")


def test_clarify_log_shape():
    """写出的 clarify.json 字段要与 clarify_score.load_questions 对得上。"""
    import tempfile
    fake = FakeServer([_q("que_1", "数据在哪？")])
    ans = ScriptedAnswerer()
    sess = _session(fake, ans)
    with tempfile.TemporaryDirectory() as td:
        log = Path(td) / "clarify.json"
        asyncio.run(run_clarify(sess, "做个报告", ans, max_rounds=15, log_path=log))
        data = json.loads(log.read_text(encoding="utf-8"))

    assert data["total_rounds"] == 1
    assert data["total_questions"] == 1
    assert data["hit_round_limit"] is False
    q = data["rounds"][0]["questions"][0]
    # clarify_score 读 q["label"] or q["question"]，必须有一个
    assert q.get("question") == "数据在哪？", q
    assert q["answer"] == ans._reply
    print("✓ clarify.json shape matches scorer")


def test_watchdog_enforces_max_runtime():
    """总时长到顶时看门狗必须主动 abort。

    这条曾被漏掉：时长约束原本挂在 urlopen 的 timeout 上，而那是**单次 socket
    操作**的上限、不是总时长——实测 max_runtime=1200 的 run 跑穿到 25 分钟仍
    未触发，最后由外层 shell 的 timeout 兜底，结果目录里什么都没落下。
    """
    class SlowServer(FakeServer):
        def __call__(self, method, path, body=None, timeout=120):
            if method == "POST" and path.endswith("/message") and not path.startswith("/question"):
                import time
                for _ in range(500):          # 一直不返回，等看门狗开火
                    if self.aborted:
                        break
                    time.sleep(0.01)
                return {"parts": [{"type": "text", "text": "partial"}]}
            return super().__call__(method, path, body, timeout)

    fake = SlowServer([])
    sess = _session(fake)
    sess._config = type("C", (), {"max_runtime_s": 1, "idle_timeout_s": 3600})()
    text = sess._send("干活", None)

    assert fake.aborted is True, "看门狗没有 abort"
    assert sess.killed_reason == "max_runtime", sess.killed_reason
    assert text == "partial", text     # 已产出的内容仍要保留供评分
    print("✓ watchdog enforces max_runtime")


def test_watchdog_escalates_to_killpg_when_abort_ignored():
    """abort 不生效时必须升级到 killpg，并保住已产出的文本。

    实测 opencode 的 /abort 只中断推理循环，**不杀**已派生的 bash 子进程：
    一个跑了 12 分钟的求解脚本 abort 后照常算，session 一直 busy、POST 不返回。
    只 abort 就收工的话，harness 会一路等到 socket 超时，而那个满载 CPU 的
    脚本还留在机器上。
    """
    class StubbornServer(FakeServer):
        """收到 abort 也不收手——复现 bash 工具卡住的场景。"""
        def __call__(self, method, path, body=None, timeout=120):
            if method == "POST" and path.endswith("/message") and "question" not in path:
                import time
                for _ in range(1000):
                    if self.killed:
                        raise urllib.error.URLError("connection refused")
                    time.sleep(0.01)
                return {"parts": []}
            if method == "GET" and path.endswith("/message"):
                return [{"info": {"role": "assistant"},
                         "parts": [{"id": "p1", "type": "text", "text": "半成品"}]}]
            return super().__call__(method, path, body, timeout)

    fake = StubbornServer([])
    fake.killed = False
    sess = _session(fake)
    sess._config = type("C", (), {"max_runtime_s": 1, "idle_timeout_s": 3600})()
    sess._ABORT_GRACE_S = 1               # 缩短宽限期，别让单测等一分钟
    sess.terminate = lambda: setattr(fake, "killed", True)

    text = sess._send("干活", None)

    assert fake.aborted is True, "应先礼：abort"
    assert fake.killed is True, "后兵：abort 无效时必须 killpg"
    assert text == "半成品", f"killpg 前抓的快照丢了: {text!r}"
    assert sess.killed_reason == "max_runtime"
    print("✓ watchdog escalates to killpg, keeps partial output")


def test_terminate_is_idempotent_and_deregisters():
    """terminate() 可重复调用（信号处理器 + close() 会各调一次）。"""
    from harness.backends.opencode import serve as mod
    fake = FakeServer([])
    sess = _session(fake)
    mod._LIVE.add(sess)
    sess.terminate()
    assert sess not in mod._LIVE
    sess.terminate()                   # 第二次不应抛
    print("✓ terminate idempotent, deregisters from _LIVE")


def test_tool_turns_counted_once_per_call():
    """工具调用轮数按 part 首次进入 running/pending 计数，不因状态流转重复计。

    计数口径必须与"agent 干了多少活"对齐：一次工具调用会先 running 后
    completed/error 两次出现在 part 列表里，按状态去重键（见 _drain_parts）
    会各产一个事件——若在每个事件都 +1，同一次调用会被记成两次，轮数上限
    实际只有一半，而报表里的 tool_turns 也会系统性翻倍。
    """
    parts = [{
        "info": {"role": "assistant", "id": "m1"},
        "parts": [
            {"id": "p1", "type": "text", "text": "开始"},          # 文本不算轮
            {"id": "p2", "type": "tool", "tool": "bash",
             "state": {"status": "running", "input": {}}},
            {"id": "p3", "type": "tool", "tool": "read",
             "state": {"status": "running", "input": {}}},
        ],
    }]
    fake = FakeServer([], parts=parts)
    sess = _session(fake)
    sess.session_id = "ses_test"

    sess._drain_parts(set())
    assert sess.tool_turns == 2, sess.tool_turns

    # 同一批 part 再轮询一次：全被去重，计数不动
    sess._drain_parts(set())
    assert sess.tool_turns == 2, sess.tool_turns

    # p2 转 error（新状态、产 tool_result 事件），但它不是一次新的工具调用
    parts[0]["parts"][1]["state"] = {"status": "error", "error": "boom"}
    sess._drain_parts(set())
    assert sess.tool_turns == 2, f"completed/error 不该重复计数: {sess.tool_turns}"

    # 新增一次真调用才 +1
    parts[0]["parts"].append({"id": "p4", "type": "tool", "tool": "bash",
                              "state": {"status": "running", "input": {}}})
    sess._drain_parts(set())
    assert sess.tool_turns == 3, sess.tool_turns
    print("✓ tool_turns counted once per call")


def test_watchdog_enforces_max_turns():
    """工具调用数到顶时看门狗必须 abort，理由记 max_turns。

    与 max_runtime 是两个独立的失控形态：agent 反复读同一个文件、每次都很快
    返回，时长和 idle 都不会触发，但它永远不会交付。只有轮数能拦住这种打转。
    """
    class LoopingServer(FakeServer):
        """POST 一直不返回，同时不断产出新的 tool part——模拟无限打转。"""
        def __init__(self):
            super().__init__([], parts=[{"info": {"role": "assistant", "id": "m1"},
                                         "parts": []}])
            self._n = 0

        def __call__(self, method, path, body=None, timeout=120):
            if method == "GET" and path.endswith("/message"):
                self._n += 1
                self._parts[0]["parts"] = [
                    {"id": f"p{i}", "type": "tool", "tool": "bash",
                     "state": {"status": "running", "input": {}}}
                    for i in range(self._n)
                ]
                return self._parts
            if method == "POST" and path.endswith("/message") and "question" not in path:
                import time
                for _ in range(500):
                    if self.aborted:
                        break
                    time.sleep(0.01)
                return {"parts": [{"type": "text", "text": "partial"}]}
            return super().__call__(method, path, body, timeout)

    fake = LoopingServer()
    sess = _session(fake)
    # 时长给足，确保触发的是轮数而不是超时
    sess._config = type("C", (), {"max_runtime_s": 3600, "idle_timeout_s": 3600,
                                  "max_turns": 3})()
    text = sess._send("干活", None)

    assert fake.aborted is True, "轮数到顶没有 abort"
    assert sess.killed_reason == "max_turns", sess.killed_reason
    assert sess.tool_turns >= 3, sess.tool_turns
    assert text == "partial", text          # 已产出内容仍保留供评分
    print("✓ watchdog enforces max_turns")


def test_max_turns_none_never_triggers():
    """不配 max_turns 时轮数无上限——默认行为不能被这个新开关改变。"""
    class BusyServer(FakeServer):
        def __init__(self):
            super().__init__([], parts=[{"info": {"role": "assistant", "id": "m1"},
                                         "parts": [
                                             {"id": f"p{i}", "type": "tool", "tool": "bash",
                                              "state": {"status": "running", "input": {}}}
                                             for i in range(50)]}])

        def __call__(self, method, path, body=None, timeout=120):
            if method == "POST" and path.endswith("/message") and "question" not in path:
                import time
                time.sleep(1.2)          # 轮询间隔 0.5s，留够两轮把 50 次数完
                return {"parts": [{"type": "text", "text": "done"}]}
            return super().__call__(method, path, body, timeout)

    fake = BusyServer()
    sess = _session(fake)
    sess._config = type("C", (), {"max_runtime_s": 3600, "idle_timeout_s": 3600,
                                  "max_turns": None})()
    text = sess._send("干活", None)

    assert sess.tool_turns == 50, sess.tool_turns
    assert fake.aborted is False, "未配上限却 abort 了"
    assert sess.killed_reason is None, sess.killed_reason
    assert text == "done", text
    print("✓ max_turns=None never triggers")


def _serve_pids() -> set[str]:
    import subprocess
    out = subprocess.run(["pgrep", "-f", "opencode serve"],
                         capture_output=True, text=True)
    return {l for l in out.stdout.split() if l}


if __name__ == "__main__":
    # 单测绝不该起真的 serve。曾漏设 _launched 导致每跑一次就留下几个常驻
    # opencode 进程（各占几百 MB），跑几轮机器就被拖垮，且没有任何报错提示。
    before = _serve_pids()
    for fn in [v for k, v in sorted(globals().items()) if k.startswith("test_")]:
        fn()
    leaked = _serve_pids() - before
    assert not leaked, f"单测泄漏了 opencode serve 进程: {sorted(leaked)}"
    print("\nall serve-backend tests passed (no leaked processes)")
