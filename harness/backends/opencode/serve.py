"""OpenCode 后端（HTTP server 形态）——澄清走脚手架**原生** question tool。

为什么不是 `<clarify>` 文本标记
────────────────────────────
评测框架不该发明自己的提问协议。自造标记等于给被测脚手架换了一条它平时
不走的通路：agent 得先学会一个陌生格式，模型输出坏 JSON 时框架还要正则
抢救（实测因此连发 15 轮相同问题、空烧一小时）。opencode 原生就有澄清
通道——`question` tool + `/question/{id}/reply`——用它才是在测脚手架本身。
这也让 opencode 与 OpenHands 在同一个抽象下对齐：两边都是「框架拦截原生
提问通道 → 注入答案 → agent 继续」，差别只在 API 名字。

为什么必须用 serve 而不是 `opencode run`
──────────────────────────────────────
`opencode run` 启动时硬编码把 question 权限置为 deny（CLI 无人可问），
该 tool 因此不可用。只有 `opencode serve` 的 HTTP 接口能把它打开。
顺带解决了另一个老问题：`run --session <id>` 的续接在本环境会挂起，
而 serve 里多轮天然同进程同会话，根本不需要续接。

一次 send 的时序
────────────────
    POST /session/{id}/message        （阻塞，直到本轮 agent 收尾）
        ↑ 后台 poller 线程并行做两件事：
            GET /question             → 有提问就调 answerer 作答并 reply
            GET /session/{id}/message → 新增 part 转成 AgentEvent 抛给上层，
                                        同时刷新 idle 看门狗
"""

from __future__ import annotations

import json
import os
import re
import atexit
import signal
import socket
import subprocess
import threading
import time
import urllib.error
import urllib.parse
import urllib.request
import pathlib
from datetime import datetime, timezone
from pathlib import Path

from ..base import AgentEvent, AgentRunner, AgentSession
from ...defaults import resolve_model

# 权限基线：question 打开（本 benchmark 的核心通路），webfetch 关闭
# （理由见 environment.py：污染公开数据集答案、考点不在检索、结果不可复现）。
# 外部目录权限在建会话时显式设置；其他工具保留脚手架原生权限。
_PERMISSION_BASE = [
    {"permission": "webfetch", "pattern": "*", "action": "deny"},
    {"permission": "websearch", "pattern": "*", "action": "deny"},
]


def _free_port() -> int:
    with socket.socket() as s:
        s.bind(("127.0.0.1", 0))
        return s.getsockname()[1]


# 所有活着的 serve 句柄。serve 是长驻进程，且它派生的 bash 工具（agent 写的
# 求解脚本等）挂在它的进程组下——harness 若非正常退出（外层 timeout、Ctrl-C、
# 崩溃），close() 不会执行，serve 连同一个满载 CPU 的 solve.py 会一直留在机器上。
# 实测一次被外层 timeout 杀掉的 run 就留下了这么一对，只能手工 kill。
_LIVE: "set[OpenCodeServeSession]" = set()


def _cleanup_all() -> None:
    for sess in list(_LIVE):
        sess.terminate()


atexit.register(_cleanup_all)

for _sig in (signal.SIGTERM, signal.SIGINT):
    _prev = signal.getsignal(_sig)

    def _handler(signum, frame, _prev=_prev):
        _cleanup_all()
        if callable(_prev) and _prev not in (signal.SIG_DFL, signal.SIG_IGN):
            _prev(signum, frame)
        else:
            signal.signal(signum, signal.SIG_DFL)
            os.kill(os.getpid(), signum)

    try:
        signal.signal(_sig, _handler)
    except ValueError:
        pass          # 非主线程导入时无法装 handler，atexit 仍然生效


def _children(pid: int) -> list[int]:
    """pid 的直接子进程。读 /proc/<pid>/task/*/children，缺失时退回扫全表。"""
    kids: list[int] = []
    try:
        for task in pathlib.Path(f"/proc/{pid}/task").iterdir():
            kids += [int(x) for x in (task / "children").read_text().split()]
        return kids
    except OSError:
        pass
    for entry in pathlib.Path("/proc").iterdir():          # 内核未开 CONFIG_PROC_CHILDREN
        if not entry.name.isdigit():
            continue
        try:
            if int((entry / "stat").read_text().rsplit(") ", 1)[1].split()[1]) == pid:
                kids.append(int(entry.name))
        except (OSError, IndexError, ValueError):
            continue
    return kids


def _procs_under(directory: pathlib.Path) -> list[int]:
    """cwd 落在 directory 之内的进程。

    进程树遍历对付不了 opencode 的 bash 工具：它派生的子进程会被 setsid /
    提前 reparent 到 init，terminate 时已经不在 serve 的树里也不在它的进程组里
    （实测同一个脚本两次运行，一次抓得到、一次抓不到——纯粹是竞态）。
    workspace 归属是个确定性判据：这次 run 的工作目录里跑着的东西，就是这次
    run 派生的。评测场景下这条规则精确且安全——每个 run 独占一个 workspace。
    """
    root = str(directory.resolve())
    found: list[int] = []
    me = os.getpid()
    for entry in pathlib.Path("/proc").iterdir():
        if not entry.name.isdigit() or int(entry.name) == me:
            continue
        try:
            cwd = os.readlink(entry / "cwd")
        except OSError:
            continue
        if cwd == root or cwd.startswith(root + os.sep):
            found.append(int(entry.name))
    return found


def _kill_tree(pid: int) -> None:
    """自底向上杀掉整棵进程树。

    先收集再杀：边杀边遍历会因为 init 收养孤儿而漏掉一整枝。
    """
    tree: list[int] = []
    frontier = [pid]
    while frontier:
        cur = frontier.pop()
        tree.append(cur)
        frontier += _children(cur)
    for target in reversed(tree):        # 先叶子后根，避免根一死就失去父子关系
        try:
            os.kill(target, signal.SIGKILL)
        except (ProcessLookupError, PermissionError):
            pass


class _Api:
    """opencode serve 的最小 HTTP 客户端。只用标准库，不引第三方依赖。"""

    def __init__(self, base: str, directory: str):
        self._base = base
        self._dir = directory

    def __call__(self, method: str, path: str, body=None, timeout: float = 120):
        sep = "&" if "?" in path else "?"
        url = f"{self._base}{path}{sep}directory={urllib.parse.quote(self._dir, safe='')}"
        data = json.dumps(body).encode() if body is not None else None
        req = urllib.request.Request(
            url, data=data, method=method,
            headers={"Content-Type": "application/json"},
        )

        def _once():
            with urllib.request.urlopen(req, timeout=timeout) as resp:
                raw = resp.read().decode(errors="replace")
            return json.loads(raw) if raw.strip() else None

        if method.upper() != "GET":
            # POST /message 已把 prompt 送进会话、POST reply 已回注答案：盲重会重复，
            # 这些不重试；模型调用层的重试在 opencode 内部
            return _once()
        from .._retry import retry_transient
        return retry_transient(_once, label=f"opencode GET {path.split('?')[0]}")


class OpenCodeServeSession(AgentSession):
    """基于 `opencode serve` 的会话。question tool 可用，多轮同进程。"""
    reports_tool_calls = True
    # Protocol recovery, not another solve attempt: keep the same session and
    # original deadline, and never replay the task or give artifact-specific hints.
    _EMPTY_TERMINAL_CONTINUATIONS = 2
    _CONTINUE_MESSAGE = "Continue from the current session state."

    def __init__(self, workspace: Path, config=None):
        self.session_id: str | None = None
        self.on_event = None
        self.killed_reason: str | None = None
        self.degraded_reason: str | None = None
        self.backend_diagnostics: dict = {}
        self._terminal_continuations = 0
        self.clarify_rounds: list[dict] = []      # 原生 question tool 的提问记录

        self._workspace = workspace
        self._config = config
        self._answerer = None
        self._max_clarify_rounds = 30
        self._max_clarify_questions = 0
        self._answered_clarify_questions = 0
        self.hit_round_limit = False
        self.hit_question_limit = False
        self._port = _free_port()
        self._api = _Api(f"http://127.0.0.1:{self._port}", str(workspace))
        self._proc: subprocess.Popen | None = None
        self._launched = False
        self._container_name: str | None = None
        self._seen_parts: set[str] = set()
        # Tool parts can remain in ``running`` for hours while a shell solver
        # is doing real work. Keep their ids separately so the watchdog can
        # distinguish active work from a genuinely idle session.
        self._active_tool_ids: set[str] = set()
        self._fallback_text = ""
        # 已报过的诊断信息，用于「同类只报一次」——轮询是 0.5 秒一轮，
        # 不去重会把事件流刷满，反而把真正的信息埋掉。
        self._diag: set[str] = set()
        # 工具调用轮数：每个 tool part 首次转入 running/pending 记一次。
        # 与澄清轮数（_max_clarify_rounds）是两个维度——前者数"干了多少活"，
        # 后者数"问了多少轮"。
        self.tool_turns = 0

    # ── answerer 注入 ────────────────────────────────────────────────────────

    def set_answerer(self, answerer, *, max_rounds: int = 30,
                     max_questions: int = 0) -> None:
        """挂上澄清回答者。未挂时 question 权限直接置 deny。

        max_rounds 只作失控兜底：正常情况下 agent 问够了自己就会开工，
        真撞上限说明它在空转，此时 reject 后续提问逼它收尾——继续无限作答
        只会白烧预算（旧实现实测连问 15 轮相同问题、空烧 19 分钟）。
        """
        self._answerer = answerer
        self._max_clarify_rounds = max_rounds
        self._max_clarify_questions = max_questions

    @property
    def supports_native_clarify(self) -> bool:
        return True

    # ── 生命周期 ────────────────────────────────────────────────────────────

    def start(self) -> None:
        self._launch()
        self._open_session()

    def _launch(self) -> None:
        """起 serve 进程（或容器）并等就绪。"""
        self._workspace.mkdir(parents=True, exist_ok=True)
        cmd = ["opencode", "serve", "--port", str(self._port), "--hostname", "127.0.0.1"]

        container = getattr(self._config, "container", None)
        if container:
            self._start_container(cmd, container)
        else:
            self._proc = subprocess.Popen(
                cmd, cwd=str(self._workspace), env=dict(os.environ),
                stdout=subprocess.DEVNULL, stderr=subprocess.PIPE,
                start_new_session=True,
            )
        self._wait_ready()
        self._launched = True
        _LIVE.add(self)

    def _open_session(self) -> None:
        """建会话并定稿权限。

        question 权限：有 answerer 才开。R / F 条件下「无人可问」是条件定义的
        一部分，必须真的把通道关掉——只在 prompt 里说一句而 tool 仍可调用，
        agent 一旦调用就会挂在无人应答的请求上。
        """
        perms = list(_PERMISSION_BASE)
        # Native default is ask, NOT deny. A headless solver cannot answer that
        # approval, and bash's execution timeout does not cover the wait before
        # execution. Fail closed so the model can correct its command instead
        # of consuming the entire run budget. Preserve the extractor opt-in.
        perms.append({
            "permission": "external_directory", "pattern": "*",
            "action": "allow" if getattr(self._config, "allow_external_directory", False) else "deny",
        })
        perms.append({
            "permission": "question", "pattern": "*",
            "action": "ask" if self._answerer is not None else "deny",
        })
        created = self._api("POST", "/session", {"title": "FDE-bench",
                                                 "permission": perms})
        self.session_id = created["id"]
        self._emit(AgentEvent(type="info", content=f"session_id: {self.session_id}"))

    def _start_container(self, cmd: list[str], spec) -> None:
        """容器内 detached 起 serve。

        与 opencode_run 的 `docker run --rm -i` 不同：serve 是长驻进程，
        必须 -d 常驻，由 close() 负责 rm -f。--network host 让宿主直接用
        127.0.0.1:port 访问，无需 -p 映射。
        """
        from ...isolation.container import wrap_command
        env = dict(os.environ)
        name = f"FDE-bench-serve-{os.getpid()}-{self._port}"
        full = wrap_command(cmd, spec, self._workspace, env)
        # wrap_command 产出的是 `docker run --rm -i ...`；serve 要常驻且不占 stdin
        assert full[:4] == ["docker", "run", "--rm", "-i"], full[:4]
        full = ["docker", "run", "-d", "--name", name] + full[4:]
        out = subprocess.run(full, capture_output=True, text=True, timeout=120)
        if out.returncode != 0:
            raise RuntimeError(f"启动 serve 容器失败: {out.stderr.strip()[:500]}")
        self._container_name = name

    def _wait_ready(self, timeout: float = 60) -> None:
        deadline = time.time() + timeout
        last = ""
        while time.time() < deadline:
            try:
                if self._api("GET", "/global/health", timeout=5).get("healthy"):
                    return
            except Exception as exc:
                last = f"{type(exc).__name__}: {exc}"
            if self._proc is not None and self._proc.poll() is not None:
                err = self._proc.stderr.read().decode(errors="replace")[-800:]
                raise RuntimeError(f"opencode serve 启动即退出:\n{err}")
            time.sleep(0.5)
        raise RuntimeError(f"opencode serve 未在 {timeout}s 内就绪（最后错误：{last}）")

    async def close(self) -> None:
        self.terminate()

    def terminate(self) -> None:
        """停掉 serve 及其全部子孙。幂等，可从信号处理器里调。

        为什么不能只 killpg：opencode 的 bash 工具把子进程放进**自己的**进程组
        （实测 abort 之后一个 `time.sleep(600)` 仍然活着，serve 的 pgid 里找不到
        它）。只杀进程组会把 agent 派生的求解脚本留成孤儿，继续满载 CPU——真实
        跑批时这类残留会一直堆积。故按 /proc 的父子关系整棵树杀。
        """
        _LIVE.discard(self)
        if self._container_name:
            try:
                subprocess.run(["docker", "rm", "-f", self._container_name],
                               capture_output=True, timeout=60)
            except Exception:
                pass
            self._container_name = None
        if self._proc is not None:
            _kill_tree(self._proc.pid)
            self._proc = None
        # 兜底扫一遍：树遍历漏掉的、已被 init 收养的 agent 子进程
        for pid in _procs_under(self._workspace):
            try:
                os.kill(pid, signal.SIGKILL)
            except (ProcessLookupError, PermissionError):
                pass

    # ── 发消息 ──────────────────────────────────────────────────────────────

    async def send(self, message: str, *, stop_on_tools: set[str] | None = None,
                   fork: bool = False) -> str:
        import asyncio
        return await asyncio.to_thread(self._send, message, stop_on_tools)

    def _send(self, message: str, stop_on_tools: set[str] | None) -> str:
        # 延迟到首次发消息才起 serve：question 权限在建 session 时定稿，
        # 而 answerer 由调用方在 runner.start() 之后才挂上。
        if self.session_id is None:
            if not self._launched:
                self._launch()
            self._open_session()
        max_runtime = (getattr(self._config, "max_runtime_s", None) or 43200)
        idle_timeout = (getattr(self._config, "idle_timeout_s", None) or 43200)
        max_turns = getattr(self._config, "max_turns", None)
        # A session may contain several sends (clarification followed by
        # solving). The runtime budget is for the whole run, not per send.
        if not hasattr(self, "_run_started_at"):
            self._run_started_at = time.time()
        run_started_at = self._run_started_at

        stop = threading.Event()          # 整个 send（含有界续接）结束后才停止看门狗
        last_activity = [time.time()]
        poll_err: list[str] = []
        self._fallback_text = ""
        watcher = threading.Thread(
            target=self._watch,
            args=(stop, last_activity, poll_err, stop_on_tools,
                  max_runtime, idle_timeout, max_turns, run_started_at),
            daemon=True,
        )
        watcher.start()

        result = None
        try:
            # timeout 给足余量：真正的时长约束由 _watch 的看门狗执行（见下）。
            # 不能指望这个 timeout —— urlopen 的 timeout 是**单次 socket 操作**
            # 的上限，不是总时长；实测 max_runtime=1200 的 run 跑到 25 分钟仍未
            # 触发，最后是外层 shell 的 timeout 兜的底。
            next_message = message
            while True:
                remaining = max_runtime - (time.time() - run_started_at)
                if remaining <= 0:
                    self.killed_reason = self.killed_reason or "max_runtime"
                if max_turns and self.tool_turns >= max_turns:
                    self.killed_reason = self.killed_reason or "max_turns"
                if self.killed_reason:
                    break
                result = self._api(
                    "POST", f"/session/{self.session_id}/message",
                    {"parts": [{"type": "text", "text": next_message}]},
                    timeout=remaining + 600,
                )
                if self.killed_reason:
                    break  # watchdog / intentional early stop must never resume
                info = (result or {}).get("info") or {}
                if info.get("error"):
                    self.degraded_reason = "opencode_message_error"
                    error = info["error"]
                    self.backend_diagnostics["terminal_error"] = {
                        "message_id": info.get("id"),
                        "name": error.get("name") if isinstance(error, dict) else type(error).__name__,
                    }
                    break  # includes auth/provider errors; never replay a POST
                if info.get("finish") != "unknown":
                    if result is None:
                        self.degraded_reason = "opencode_empty_http_response"
                    elif self._terminal_continuations and info.get("finish") == "stop":
                        self.backend_diagnostics["terminal_recovery"]["outcome"] = "recovered"
                    break

                audit = self.backend_diagnostics.setdefault("terminal_recovery", {
                    "limit": self._EMPTY_TERMINAL_CONTINUATIONS,
                    "continuation_message": self._CONTINUE_MESSAGE,
                    "session_id": self.session_id,
                    "run_started_at": run_started_at,
                    "max_runtime_s": max_runtime,
                    "events": [],
                })
                event = {"message_id": info.get("id"), "finish": info.get("finish"),
                         "part_types": [p.get("type") for p in result.get("parts", [])],
                         "elapsed_s": round(time.time() - run_started_at, 3)}
                audit["events"].append(event)
                blocker = self._empty_terminal_blocker(result)
                if self._terminal_continuations >= self._EMPTY_TERMINAL_CONTINUATIONS:
                    blocker = blocker or "continuation_limit"
                # Recheck after read-only safety checks: they also consume budget.
                if time.time() - run_started_at >= max_runtime:
                    self.killed_reason = self.killed_reason or "max_runtime"
                if max_turns and self.tool_turns >= max_turns:
                    self.killed_reason = self.killed_reason or "max_turns"
                if time.time() - last_activity[0] >= idle_timeout:
                    self.killed_reason = self.killed_reason or "idle"
                blocker = self.killed_reason or blocker
                if blocker:
                    event.update(action="not_continued", reason=blocker)
                    audit["outcome"] = "not_recovered"
                    self.degraded_reason = "opencode_unknown_terminal"
                    self._emit(AgentEvent(type="info", content=
                        f"[terminal] finish=unknown; no continuation: {blocker}"))
                    break
                self._terminal_continuations += 1
                event.update(action="continued", continuation=self._terminal_continuations)
                audit["outcome"] = "continuing"
                self._emit(AgentEvent(type="info", content=
                    f"[terminal] empty finish=unknown; same-session continuation "
                    f"{self._terminal_continuations}/{self._EMPTY_TERMINAL_CONTINUATIONS} "
                    f"within original {max_runtime}s budget"))
                next_message = self._CONTINUE_MESSAGE
                last_activity[0] = time.time()
        except urllib.error.HTTPError as exc:
            # HTTPError is also a URLError: do not mislabel an HTTP 401 as timeout.
            self.degraded_reason = "opencode_message_http_error"
            raise RuntimeError(f"HTTP {exc.code} on message") from exc
        except (urllib.error.URLError, TimeoutError, socket.timeout):
            # 连 socket 层都断了。看门狗多半已判过因，没判就记 max_runtime。
            self.killed_reason = self.killed_reason or "max_runtime"
            self._abort()
        finally:
            stop.set()
            watcher.join(timeout=10)
            audit = self.backend_diagnostics.get("terminal_recovery")
            if audit and (self.killed_reason or self.degraded_reason):
                audit["outcome"] = "not_recovered"
                audit["stop_reason"] = self.killed_reason or self.degraded_reason

        if result is None:
            # serve 可能已被 killpg，_collect_text_since 会连不上；
            # 看门狗在动手前抓的快照才是唯一还拿得到的文本。
            return (self._fallback_text or self._collect_text_since()) if self.killed_reason else ""
        texts = [p.get("text", "") for p in result.get("parts", [])
                 if p.get("type") == "text" and p.get("text")]
        if self.killed_reason:
            return "\n".join(texts) or self._fallback_text or self._collect_text_since()
        # An empty terminal response is not the older assistant's "Let me fix".
        # Keep old text only as a watchdog snapshot, never as successful output.
        return "\n".join(texts)

    def _empty_terminal_blocker(self, result: dict) -> str | None:
        """Only resume an observed, completed, empty unknown turn in an idle session.

        A failed safety GET, a different latest message, or any pending tool
        fails closed. Never resend a task after an ambiguous network failure.
        """
        info = result.get("info") or {}
        if (info.get("role") != "assistant" or not info.get("id") or
                not (info.get("time") or {}).get("completed")):
            return "unconfirmed_terminal"
        for part in result.get("parts", []):
            if part.get("type") in {"text", "reasoning"} and not part.get("text", "").strip():
                continue
            if part.get("type") not in {"step-start", "step-finish"}:
                return "nonempty_unknown_turn"
        try:
            statuses = self._api("GET", "/session/status", timeout=10)
            # OpenCode removes idle sessions from this map.
            if not isinstance(statuses, dict):
                return "unconfirmed_session_status"
            if statuses.get(self.session_id, {"type": "idle"}).get("type") != "idle":
                return "session_not_idle"
            msgs = self._api("GET", f"/session/{self.session_id}/message", timeout=10)
            if not msgs or msgs[-1].get("info", {}).get("id") != info["id"]:
                return "terminal_message_changed"
            for msg in msgs:
                for part in msg.get("parts", []):
                    if (part.get("type") == "tool" and
                            (part.get("state") or {}).get("status") in {"pending", "running"}):
                        return "unfinished_tool"
        except Exception as exc:
            return f"safety_check_failed:{type(exc).__name__}"
        return None

    def _abort(self) -> None:
        try:
            self._api("POST", f"/session/{self.session_id}/abort", timeout=30)
        except Exception:
            pass

    # 超时后等 abort 生效的宽限期。abort 只中断 agent 的推理循环，**不会**杀掉
    # 已经派生出去的 bash 子进程——实测一个跑了 12 分钟的求解脚本在 abort 后
    # 照常算，session 一直是 busy，POST 不返回。宽限期一过就直接 killpg。
    _ABORT_GRACE_S = 60

    def _stop_agent(self, done: threading.Event) -> None:
        """看门狗判超时后的收尾：先礼后兵。

        1. abort —— 让 agent 干净收手，POST 正常返回，产物齐全；
        2. 宽限期内 POST 仍不返回（多半卡在一个长跑的 bash 工具上）→ 先把
           已产出的文本抓下来，再 killpg 整个 serve 进程组。不这么做的话，
           harness 会一路等到 POST 的 socket 超时，而那个满载 CPU 的求解
           脚本还在机器上继续跑。
        """
        self._abort()
        if done.wait(self._ABORT_GRACE_S):
            return
        self._emit(AgentEvent(
            type="info",
            content=f"[watchdog] abort 后 {self._ABORT_GRACE_S}s 未收手，killpg serve 进程组",
        ))
        self._fallback_text = self._collect_text_since()
        self.terminate()

    def _collect_text_since(self) -> str:
        """POST 未正常返回时的兜底：从会话消息里捞最后一条 assistant 文本。"""
        try:
            msgs = self._api("GET", f"/session/{self.session_id}/message", timeout=60) or []
        except Exception:
            return ""
        for msg in reversed(msgs):
            if msg.get("info", {}).get("role") != "assistant":
                continue
            texts = [p.get("text", "") for p in msg.get("parts", [])
                     if p.get("type") == "text" and p.get("text")]
            if texts:
                return "\n".join(texts)
        return ""

    # ── 后台轮询：答问 + 事件 + idle 看门狗 ────────────────────────────────

    def _watch(self, stop: threading.Event, last_activity: list,
               errors: list, stop_on_tools: set[str] | None,
               max_runtime: float = 43200, idle_timeout: float = 43200,
               max_turns: int | None = None,
               run_started_at: float | None = None) -> None:
        """轮询线程：答问 + 转事件 + **执行时长/轮数约束**。

        时长必须在这里判、并主动 abort。放在 POST 的 socket timeout 上不管用
        （那是单次操作上限，不是总时长），实测会一路跑穿。abort 之后 POST 会
        很快返回，已写出的产物照常保留供评分——超时的 run 代表"做出来了但没
        做完"，与压根没跑起来是两种结论。

        轮数（max_turns）同理：数的是工具调用次数，判定点必须在能看到每次
        tool part 的这里。它与时长是两个独立的失控形态——"一直在打转"未必
        跑得久（每次调用都很快），"跑得久"也未必轮数多（单次求解脚本跑很久）。
        """
        answered: set[str] = set()
        stoppers = {t.lower() for t in (stop_on_tools or set())}
        started = run_started_at if run_started_at is not None else time.time()
        while not stop.wait(0.5):
            if self.killed_reason is None:
                elapsed = time.time() - started
                idle = time.time() - last_activity[0]
                reason = ("max_runtime" if elapsed > max_runtime else
                          "idle" if idle > idle_timeout else
                          "max_turns" if (max_turns and self.tool_turns >= max_turns) else None)
                if reason:
                    self.killed_reason = reason
                    self._emit(AgentEvent(
                        type="info",
                        content=f"[watchdog] {reason} (elapsed={elapsed:.0f}s "
                                f"idle={idle:.0f}s turns={self.tool_turns}) — abort session",
                    ))
                    self._stop_agent(stop)
                    return
            # Permission approvals are distinct from benchmark clarification
            # questions. Never send them to the answerer or auto-approve them.
            permission_waiting = False
            try:
                permission_waiting = self._reject_pending_permissions()
            except Exception as exc:
                msg = f"permission poll: {type(exc).__name__}: {exc}"
                if msg not in self._diag:
                    self._diag.add(msg)
                    self._emit(AgentEvent(type="info", content=f"[permission] {msg}"))
            try:
                for q, gen in self._poll_questions():
                    if q["id"] in answered:
                        continue
                    # sessionID 不匹配就跳过 —— 但**必须留痕**。
                    # 一个提问被静默丢掉的后果是：agent 卡在 question 工具上等到
                    # max_runtime，produced 为空、solve_secs=0，最终记一条 timeout。
                    # 从分数上看像"模型不会做题"，实则一行求解代码都没轮到写。
                    # 实测踩过：4 个提问全程无人应答，2270 秒空等。
                    if q.get("sessionID") != self.session_id:
                        key = (f"question 被丢弃: sessionID={q.get('sessionID')} "
                               f"≠ 本会话 {self.session_id}（id={q['id']} api={gen}）")
                        if key not in self._diag:
                            self._diag.add(key)
                            self._emit(AgentEvent(type="info", content=f"[clarify] {key}"))
                        continue
                    answered.add(q["id"])
                    last_activity[0] = time.time()
                    self._handle_question(q, gen)
            except Exception as exc:
                msg = f"question poll: {type(exc).__name__}: {exc}"
                errors.append(msg)
                # 轮询失败会让所有澄清提问无人应答，后果同上，必须能看见。
                # 只报第一次同类错误，避免 0.5 秒一次刷满事件流。
                if msg not in self._diag:
                    self._diag.add(msg)
                    self._emit(AgentEvent(type="info", content=f"[clarify] {msg}"))

            try:
                fresh = self._drain_parts(stoppers)
                # A running/pending tool is real activity even when its state
                # does not emit another event (for example, a long CP-SAT or
                # MIP subprocess). Without this heartbeat the idle watchdog
                # kills valid work after idle_timeout_s.
                if fresh or (self._active_tool_ids and not permission_waiting):
                    last_activity[0] = time.time()
            except Exception as exc:
                msg = f"part poll: {type(exc).__name__}: {exc}"
                errors.append(msg)
                if msg not in self._diag:
                    self._diag.add(msg)
                    self._emit(AgentEvent(type="info", content=f"[drain] {msg}"))

    def _reject_pending_permissions(self) -> bool:
        """Reject unexpected headless approvals; return whether this session waits.

        GET /permission is server-wide. Filter by sessionID before replying so
        another session's requests are never touched. Retry failed replies on
        the next poll; do not mark them handled before the server accepts them.
        Even after a successful reply, suppress the running-tool heartbeat for
        this poll, since its message state may not yet reflect the rejection.
        """
        requests = self._api("GET", "/permission", timeout=5) or []
        waiting = False
        for req in requests:
            if req.get("sessionID") != self.session_id:
                continue
            waiting = True
            rid = req["id"]
            key = f"permission-wait:{rid}"
            if key not in self._diag:
                self._diag.add(key)
                self._emit(AgentEvent(
                    type="info",
                    content=f"[permission] unattended approval: id={rid} "
                            f"permission={req.get('permission', '?')} — reject",
                ))
            try:
                self._api("POST", f"/permission/{urllib.parse.quote(rid, safe='')}/reply",
                          {"reply": "reject"}, timeout=5)
            except Exception as exc:
                key = f"permission-reply:{rid}:{type(exc).__name__}"
                if key not in self._diag:
                    self._diag.add(key)
                    self._emit(AgentEvent(
                        type="info", content=f"[permission] reject failed: id={rid} "
                                             f"{type(exc).__name__}; will retry",
                    ))
        return waiting

    def _poll_questions(self) -> list[tuple[dict, str]]:
        """把两代 question API 都查一遍，返回 [(请求, 代号)]。

        opencode 1.16.2 的 OpenAPI 里同时有两套 question 接口，各带一份独立的
        schema 与路由，**而且它们是两个互不相通的存储**：

            v1   GET  /question
                 POST /question/{requestID}/reply
            v2   GET  /api/question/request                     → {location, data:[…]}
                 POST /api/session/{sid}/question/request/{rid}/reply

        实测在同一个 serve 实例上手工触发一次提问，v1 立刻返回该请求，而 v2 的
        `data` 是空的 —— 说明同一次提问只会落进其中一边。只轮询一边的风险是：
        另一边来的提问永远无人应答，agent 卡在 question 工具上直到 max_runtime，
        产出为空。这种失败在分数上表现为"模型不会做题"，极难归因（实测两次白跑）。

        所以两边都查，谁有就答谁，并记下是哪一代 —— reply 的路由不一样，答错
        地址等于没答。
        """
        out: list[tuple[dict, str]] = []
        try:
            for q in self._api("GET", "/question", timeout=30) or []:
                out.append((q, "v1"))
        except Exception:
            raise            # v1 是主路径，出错要冒到上层记诊断
        try:
            v2 = self._api("GET", "/api/question/request", timeout=30)
            # v2 把列表包在 data 里，且带一个 location 字段；v1 是裸列表。
            data = (v2 or {}).get("data") if isinstance(v2, dict) else (v2 or [])
            for q in data or []:
                out.append((q, "v2"))
        except Exception as exc:
            # v2 在某些版本上不存在，404 属正常，只报一次不打断 v1。
            key = f"v2 question 路由不可用（不影响 v1）: {type(exc).__name__}"
            if key not in self._diag:
                self._diag.add(key)
        if out:
            key = "seen:" + ",".join(sorted({g for _, g in out}))
            if key not in self._diag:
                self._diag.add(key)
                self._emit(AgentEvent(
                    type="info",
                    content=f"[clarify] 提问来自 {key.split(':')[1]} 代接口",
                ))
        return out

    def _handle_question(self, req: dict, gen: str = "v1") -> None:
        """原生 question 请求 → answerer 作答 → reply 回注。

        answers 是「每个问题一个字符串数组」。opencode 的 question tool 允许
        自定义答案（custom 默认 true），因此直接回填 answerer 的自由文本，
        不强行往给定选项上靠——把业务方的真实答复压成一个选项标签会丢信息，
        而选项本来就是 agent 自己拟的、未必覆盖真值。
        """
        questions = req.get("questions") or []
        if len(self.clarify_rounds) >= self._max_clarify_rounds:
            self.hit_round_limit = True
            self._emit(AgentEvent(
                type="info",
                content=f"[clarify] 已达 {self._max_clarify_rounds} 轮上限，reject 后续提问",
            ))
            self._api("POST", self._q_route(req, gen, "reject"), timeout=60)
            return
        answers: list[list[str]] = []
        record: list[dict] = []
        failed: str | None = None
        dropped = 0
        t_ans = time.time()
        for q in questions:
            text = (q.get("question") or q.get("header") or "").strip()
            # answerer 是外部 LLM，会失败（网关 400 / 限流 / 超时）。**绝不能让
            # 异常冒出去** —— 冒出去这个 question 就永远不被 reply，agent 卡在
            # question 工具上直到 max_runtime，产出为空。实测一次 400 白烧 2270 秒。
            # 失败就回一句"问不到"，agent 至少能带着不确定继续做，这比整格记 0 好。
            over_budget = (
                self._max_clarify_questions > 0
                and self._answered_clarify_questions >= self._max_clarify_questions
            )
            if over_budget:
                self.hit_question_limit = True
                dropped += 1
                reply = ("No further information is available. "
                         "Proceed with your best judgement.")
            else:
                try:
                    reply = self._answerer.answer(text) if self._answerer else ""
                except Exception as exc:                  # noqa: BLE001
                    failed = f"{type(exc).__name__}: {exc}"
                    reply = "（业务方暂时无法答复这一条，请按你的专业判断处理并在交付说明里注明假设。）"
                self._answered_clarify_questions += 1
            answers.append([reply])
            record.append({
                "question": text,
                "header": q.get("header", ""),
                "options": [o.get("label", "") for o in (q.get("options") or [])],
                "answer": reply,
                **({"dropped_over_budget": True} if over_budget else {}),
                **({"answerer_error": failed} if failed else {}),
            })
            self._emit(AgentEvent(type="info", content=f"[clarify] Q: {text[:160]}"))
        if failed:
            # 必须显式报出来：这类 run 的澄清覆盖率天然偏低，若不标记，
            # 分析阶段会把"基础设施故障"读成"这个模型不会问问题"。
            self._emit(AgentEvent(
                type="info",
                content=f"[clarify] ⚠ answerer 失败，已用占位答复放行：{failed}",
            ))

        self.clarify_rounds.append({
            "turn": len(self.clarify_rounds) + 1,
            "asked_at": datetime.now(timezone.utc).isoformat(),
            "request_id": req.get("id"),
            "questions": record,
            "dropped_over_budget": dropped,
            "harness_secs": round(time.time() - t_ans, 1),   # agent 在 question 工具上等 answerer 的时间
        })
        flush = getattr(self, "_clarify_flush", None)
        if callable(flush):
            try:
                flush()
            except Exception:                          # noqa: BLE001
                pass
        self._api("POST", self._q_route(req, gen, "reply"),
                  {"answers": answers}, timeout=60)

    def _q_route(self, req: dict, gen: str, verb: str) -> str:
        """按提问来自哪一代接口，选对应的 reply/reject 路由。

        两代的路径形状完全不同 —— v2 把 sessionID 编进路径里。答到错的地址上
        opencode 会返回 QuestionNotFoundError，而调用方只看到一次异常、提问
        依旧挂着，等于没答。
        """
        if gen == "v2":
            return f"/api/session/{req.get('sessionID')}/question/request/{req['id']}/{verb}"
        return f"/question/{req['id']}/{verb}"

    def _drain_parts(self, stoppers: set[str]) -> bool:
        """把新增的 message part 转成 AgentEvent。返回是否有新增。"""
        msgs = self._api("GET", f"/session/{self.session_id}/message", timeout=60) or []
        fresh = False
        active_tool_ids: set[str] = set()
        for msg in msgs:
            info = msg.get("info", {})
            if info.get("role") != "assistant":
                continue
            # OpenCode 的 provider/协议错误挂在 assistant message.info.error，而不是
            # parts 中。以前这里只扫 parts，外层只能看到无信息量的 UnknownError；
            # 保留脱敏后的 name/status/message，才能区分网关 4xx/5xx 与协议错误。
            error = info.get("error")
            if error:
                raw_data = error.get("data") if isinstance(error, dict) else None
                data = raw_data if isinstance(raw_data, dict) else {}
                summary = {
                    "name": error.get("name") if isinstance(error, dict) else type(error).__name__,
                    "status": data.get("statusCode") or data.get("status"),
                    "message": (data.get("message") or
                                (error.get("message") if isinstance(error, dict) else str(error))),
                    "response": data.get("responseBody") or data.get("response"),
                }
                raw = json.dumps({k: v for k, v in summary.items() if v is not None},
                                 ensure_ascii=False)[:2000]
                raw = re.sub(r"(?i)bearer\s+[A-Za-z0-9._~+/=-]+",
                             "Bearer [REDACTED]", raw)
                raw = re.sub(r"sk-[A-Za-z0-9_-]{12,}", "sk-[REDACTED]", raw)
                key = f"message-error:{info.get('id', '')}:{raw}"
                if key not in self._seen_parts:
                    self._seen_parts.add(key)
                    fresh = True
                    self._emit(AgentEvent(type="info",
                                          content=f"[provider] session error: {raw}"))
            for part in msg.get("parts", []):
                pid = part.get("id")
                ptype = part.get("type")
                state = part.get("state") or {}
                if (pid and ptype == "tool" and
                        state.get("status") in {"running", "pending"}):
                    active_tool_ids.add(pid)
                # tool part 会先以 running 出现、后转 completed；用 status 参与
                # 去重键，才能同时拿到 tool_call 与 tool_result 两个事件。
                key = f"{pid}:{state.get('status', '')}"
                if not pid or key in self._seen_parts:
                    continue
                self._seen_parts.add(key)
                fresh = True
                if ptype == "text" and part.get("text"):
                    self._emit(AgentEvent(type="text", content=part["text"]))
                elif ptype == "tool":
                    tool = part.get("tool", "?")
                    status = state.get("status", "")
                    if status == "running" or status == "pending":
                        self.tool_turns += 1
                        self._emit(AgentEvent(
                            type="tool_call", tool=tool,
                            tool_input=json.dumps(state.get("input", {}), ensure_ascii=False)[:2000],
                        ))
                        if tool.lower() in stoppers:
                            self.killed_reason = "early_stop"
                            self._abort()
                    elif status == "error":
                        self._emit(AgentEvent(type="tool_result", tool=tool,
                                              content=str(state.get("error", ""))[:2000],
                                              is_error=True))
        self._active_tool_ids = active_tool_ids
        return fresh


# ── Runner ───────────────────────────────────────────────────────────────────

class OpenCodeServeRunner(AgentRunner):
    """启动一个 `opencode serve`，返回可交互的 session。"""

    async def start(self, workspace: Path) -> OpenCodeServeSession:
        workspace.mkdir(parents=True, exist_ok=True)
        if self.config.agent_config_dir:
            from ...installer import install_agent_config
            install_agent_config(self.config.agent_config_dir, workspace, ".opencode")
        _inject_model(workspace, self.config.model, self.config.runner_type)
        return OpenCodeServeSession(workspace, config=self.config)


def _inject_model(workspace: Path, model: str | None, runner_type: str) -> None:
    """把 model 写进 workspace/.opencode/opencode.jsonc（与 opencode_run 一致）。"""
    import re
    resolved = resolve_model(model, runner_type)
    if resolved is None:
        return
    config_dir = workspace / ".opencode"
    config_dir.mkdir(parents=True, exist_ok=True)
    path = config_dir / "opencode.jsonc"
    if path.exists():
        text = re.sub(r"^\s*//.*$", "", path.read_text(encoding="utf-8"), flags=re.M)
        data = json.loads(re.sub(r",\s*([}\]])", r"\1", text))
    else:
        data = {"$schema": "https://opencode.ai/config.json"}
    data["model"] = resolved
    path.write_text(json.dumps(data, indent=2, ensure_ascii=False), encoding="utf-8")
