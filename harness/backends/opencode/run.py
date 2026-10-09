"""
OpenCode runners — 通过 opencode run --format json 运行 OpenCode。

替代原 ACP 实现，使用子进程 + stdout JSON 事件流，
解决 ACP 的事件竞争问题，并提供更丰富的事件信息（工具 input/output、token 统计）。

多轮交互通过 --session <ID> 续接 session（注意：不能同时带 --continue，
opencode 的 --continue 是"使用最近一个 session"，与 --session 互斥；
两个一起传会导致 --session 被忽略，消息发到全局最新 session 里）。

Runner 类型：
  OpenCodeRunner       — 注入指定配置（opencode.jsonc + agents/），路由由 default_agent 控制
  OpenCodeNativeRunner — 不注入配置，走 OpenCode 内置 build agent
"""

import asyncio
import json
import os
import re
import signal
import subprocess
import sys
import threading
import time
from pathlib import Path

from ..base import AgentEvent, AgentRunner, AgentSession
from ...defaults import resolve_model

os.environ.setdefault("OPENCODE_EXPERIMENTAL_OUTPUT_TOKEN_MAX", "64000")
# 关掉 tools/lib/log.ts 的 HTTP sink:runners 离线没有日志服务,
# 不设的话 plugin 每 60s 会在 stderr 打一行 endpoint unreachable warning。
# 显式设了别的值(http URL)就尊重 caller 的选择。
os.environ.setdefault("WEBAGENT_LOG_ENDPOINT_URL", "")


def _killpg_safe(proc: subprocess.Popen) -> None:
    """杀整个进程组（opencode + 派生的 bash/工具子孙），幂等。

    依赖 Popen(start_new_session=True)：子进程会自成一个 session/group，
    killpg 不会误伤本 runner 自己。
    """
    try:
        os.killpg(os.getpgid(proc.pid), signal.SIGKILL)
    except (ProcessLookupError, PermissionError):
        pass


# ── Session ──────────────────────────────────────────────────────────────────

class OpenCodeRunSession(AgentSession):
    """
    基于 opencode run --format json 的 session。

    每次 send() 启动一个子进程，通过 stdout JSON 事件流收集结果。
    多轮对话通过 --session <ID> 复用 session。
    """

    reports_tool_calls = True

    def __init__(self, workspace: Path, config=None):
        self.session_id: str | None = None
        self.on_event = None
        self._workspace = workspace
        self._config = config
        self.killed_reason: str | None = None   # watchdog 击杀原因，见 _watchdog()

    async def send(self, message: str, *, stop_on_tools: set[str] | None = None, fork: bool = False) -> str:
        # DB 兜底必须只读「本轮新增」的 text：记录启动前的最新 part 时间戳作游标，
        # 否则会读到历史 part（实测：agent 本轮只调工具不说话时，兜底读到上一轮的
        # <clarify>，run_clarify 误判"又提问"→ 连发 15 轮相同问题、空烧 19 分钟）。
        marker = self._last_part_time() if self.session_id else 0
        text, events = await asyncio.to_thread(
            self._run, message, stop_on_tools, fork,
        )
        # 正常路径：stdout 里有 step_finish(reason=end_turn)，text_parts 已包含完整最终文本。
        # 异常路径（高延迟 provider）：进程在 end_turn 前退出，最后一轮文本
        # （含 <clarify>）只写进了 DB，没来得及刷到 stdout。此时从 sqlite3 直读 DB 取
        # 本轮新增的最后一条 assistant text part，绕开 opencode export 子进程的
        # WAL checkpoint 时序问题。early-stop(stop_on_tools)是有意中断，不回退。
        got_end_turn = any(
            e.get("type") == "step_finish"
            and e.get("part", {}).get("reason") == "end_turn"
            for e in events
        )
        # DB 里本轮新增的 assistant text 才是权威：stdout 事件流可能漏掉最末一段
        # 文本（实测 reason=stop 正常收尾时，含 <clarify> 的那段就没进 text_parts，
        # 澄清循环因此当成"未提问"直接退出、run 无产物）。
        # 仅当 DB 明显更长时替换，避免把 stdout 已聚合的多段文本换成单段。
        if self.session_id and not stop_on_tools:
            fb = await asyncio.to_thread(self._export_assistant_text_after, marker)
            if fb.strip() and (not got_end_turn or len(fb) > len(text)):
                text = fb
        return text

    def _last_part_time(self) -> int:
        """当前 session 在 DB 里的最新 part 时间戳（毫秒），作为本轮新增的起点。"""
        db_path = Path(os.environ.get("XDG_DATA_HOME", "")) / "opencode" / "opencode.db"
        if not db_path.exists():
            return 0
        try:
            import sqlite3
            conn = sqlite3.connect(str(db_path), timeout=10)
            row = conn.execute(
                "SELECT MAX(time_created) FROM part WHERE session_id=?", (self.session_id,)
            ).fetchone()
            conn.close()
            return row[0] if row and row[0] else 0
        except Exception:
            return 0

    def _export_assistant_text_after(self, marker: int) -> str:
        """读 DB 里时间戳晚于 marker 的最后一条 assistant text part。

        只取本轮新增的文本：agent 本轮若只调工具、无新文本，返回空串；
        若发了新 <clarify>，则返回它。marker=0（首轮）退化为取全会话最后一条。
        """
        if not self.session_id:
            return ""
        data_home = os.environ.get("XDG_DATA_HOME")
        if not data_home:
            return ""
        db_path = Path(data_home) / "opencode" / "opencode.db"
        if not db_path.exists():
            return ""
        try:
            import sqlite3
            conn = sqlite3.connect(str(db_path), timeout=10)
            rows = conn.execute(
                """
                SELECT p.data FROM part p
                JOIN message m ON p.message_id = m.id
                WHERE m.session_id = ?
                  AND json_extract(m.data, '$.role') = 'assistant'
                  AND json_extract(p.data, '$.type') = 'text'
                  AND json_extract(p.data, '$.text') IS NOT NULL
                  AND json_extract(p.data, '$.text') != ''
                  AND p.time_created > ?
                ORDER BY p.time_created DESC LIMIT 1
                """,
                (self.session_id, marker),
            ).fetchall()
            conn.close()
            if rows:
                return json.loads(rows[0][0]).get("text", "")
        except Exception:
            pass
        return ""


    def _run(self, message: str, stop_on_tools: set[str] | None, fork: bool = False) -> tuple[str, list[dict]]:
        """同步执行 opencode run，解析 stdout JSON 事件流。"""
        cmd = [
            "opencode", "run",
            "--format", "json",
            "--dir", str(self._workspace),
            "--dangerously-skip-permissions",
        ]

        if self.session_id:
            # 注意：opencode 的 --session 已经隐含"继续这个 session"，**不要**再带 --continue,
            # 否则两者一起会让 opencode 静默 fail 并新建一个 session（实测过）。
            cmd.extend(["--session", self.session_id])
            # fork=True 时带 --fork：本次 send 不污染原 session，
            # 输出事件里 sessionID 会变成新 fork 的 id。
            if fork:
                cmd.append("--fork")

        # opencode run 不像 ACP 那样自动检测 / 前缀路由 command，
        # 需要显式传 --command 才能走 sdk.session.command()
        stripped = message.strip()
        if stripped.startswith("/"):
            parts = stripped[1:].split(None, 1)
            cmd.extend(["--command", parts[0]])
            cmd.append(parts[1] if len(parts) > 1 else "")
        else:
            cmd.append(message)

        # DB 隔离由 environment.py 的私有 XDG_DATA_HOME（run_dir/.agent_home/data）提供，
        # 每个 run 一份，天然没有并发写锁竞争。不要把它改指到 workspace 内部——
        # 那会让 agent 的 SQLite 文件和交付物混在一起，污染产物识别。
        env = dict(os.environ)

        # 容器 / bwrap 包裹。二者互斥：容器已隔离文件系统与进程，不再叠 bwrap。
        # 必须在 env 定稿之后，否则传进容器的 XDG_DATA_HOME 与宿主侧不一致，
        # SQLite 会落到两个不同位置。
        if self._config and getattr(self._config, "container", None):
            from ...isolation.container import wrap_command as container_wrap
            cmd = container_wrap(
                cmd, self._config.container, self._workspace, env
            )
        elif self._config and self._config.sandbox:
            from ...isolation.sandbox import wrap_command
            cmd = wrap_command(cmd, self._config.sandbox, self._workspace)

        proc = subprocess.Popen(
            cmd,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            cwd=str(self._workspace),
            env=env,
            start_new_session=True,  # Fix 1：让 proc 自成 process group，便于 killpg 全部子孙
        )

        # 看门狗：双重时长防护（死循环检测归 opencode 自身的 doom_loop / step-limiter）
        #  - idle：卡死时进程仍活但不再产出 stdout 事件，空闲超时触发 killpg；
        #    正常长跑（evolve 等）只要有事件刷新就不会被误杀
        #  - total：总墙钟时长硬上限，活跃但跑太久同样止损
        idle_timeout = (self._config.idle_timeout_s if self._config else None) or 43200
        max_runtime = (self._config.max_runtime_s if self._config else None) or 43200
        started_at = time.time()
        last_activity = [time.time()]
        watchdog_stop = threading.Event()

        def _watchdog():
            while not watchdog_stop.wait(timeout=30):
                if proc.poll() is not None:
                    return
                elapsed = time.time() - started_at
                if elapsed > max_runtime:
                    print(
                        f"[opencode] total runtime {elapsed:.0f}s > {max_runtime}s, killpg pid={proc.pid}",
                        file=sys.stderr,
                    )
                    # 记录击杀原因：killpg 后 send() 会正常返回，调用方无从区分
                    # "agent 自己收尾" 和 "被掐死"，会把残缺 run 记成 status=ok。
                    self.killed_reason = "max_runtime"
                    _killpg_safe(proc)
                    return
                idle = time.time() - last_activity[0]
                if idle > idle_timeout:
                    print(
                        f"[opencode] idle {idle:.0f}s > {idle_timeout}s, killpg pid={proc.pid}",
                        file=sys.stderr,
                    )
                    self.killed_reason = "idle"
                    _killpg_safe(proc)
                    return

        watchdog_thread = threading.Thread(target=_watchdog, daemon=True)
        watchdog_thread.start()

        events: list[dict] = []
        text_parts: list[str] = []
        early_stopped = False

        try:
            # opencode run --format json 的输出理论上每行一个 JSON，
            # 但 tool output 含换行时会导致 JSON 跨行，需要用 buffer 拼接。
            buf = ""
            for raw in proc.stdout:
                last_activity[0] = time.time()  # 收到任何数据就刷新 watchdog
                buf += raw.decode(errors="replace")
                # 尝试从 buffer 中解析完整的 JSON 对象
                while buf:
                    buf = buf.lstrip()
                    if not buf or buf[0] != "{":
                        # 跳过非 JSON 行（如 opencode 的日志输出）
                        nl = buf.find("\n")
                        if nl == -1:
                            break
                        skipped = buf[:nl].strip()
                        if skipped:
                            print(f"[opencode] {skipped[:200]}", file=sys.stderr)
                        buf = buf[nl + 1:]
                        continue
                    try:
                        event, end = json.JSONDecoder().raw_decode(buf)
                    except json.JSONDecodeError:
                        break  # 不完整，等更多数据
                    buf = buf[end:]

                    # ── 处理已解析的事件 ──
                    events.append(event)

                    new_sid = event.get("sessionID")
                    if new_sid and self.session_id is None:
                        self.session_id = new_sid
                        self._emit(AgentEvent(type="info", content=f"session_id: {self.session_id}"))
                    elif new_sid and fork and new_sid != self.session_id:
                        # --fork 产出全新 session_id，切换过去
                        self.session_id = new_sid
                        self._emit(AgentEvent(type="info", content=f"forked session_id: {new_sid}"))

                    etype = event.get("type", "")

                    if etype == "text":
                        text = event.get("part", {}).get("text", "")
                        if text:
                            text_parts.append(text)
                            self._emit(AgentEvent(type="text", content=text))

                    elif etype == "tool_use":
                        part = event.get("part", {})
                        tool = part.get("tool", "?")
                        state = part.get("state", {})
                        status = state.get("status", "?")
                        tool_input = json.dumps(state.get("input", {}), ensure_ascii=False)

                        if stop_on_tools and tool.lower() in {t.lower() for t in stop_on_tools}:
                            early_stopped = True
                            self._emit(AgentEvent(
                                type="info",
                                content=f"[early stop] 检测到 {tool}，终止 session",
                            ))
                            # 立即 killpg 并关闭 stdout，防止 for 循环阻塞在读上
                            _killpg_safe(proc)
                            proc.stdout.close()
                            break

                        self._emit(AgentEvent(
                            type="tool_call", tool=tool, tool_input=tool_input,
                        ))
                        if status == "error":
                            self._emit(AgentEvent(
                                type="tool_result", tool=tool,
                                content=state.get("error", ""), is_error=True,
                            ))

                    elif etype == "step_finish":
                        part = event.get("part", {})
                        tokens = part.get("tokens", {})
                        reason = part.get("reason", "")
                        if tokens:
                            self._emit(AgentEvent(
                                type="info",
                                content=f"[step] reason={reason} tokens: in={tokens.get('input', 0)} out={tokens.get('output', 0)} cache_read={tokens.get('cache', {}).get('read', 0)}",
                            ))
                        if reason == "end_turn":
                            # 完整响应已捕获（end_turn 前所有 text 事件已到达）。
                            # 主动终止进程，不等 EOF，避免高延迟 provider 导致 stdout 延迟关闭。
                            _killpg_safe(proc)
                            proc.stdout.close()
                            break

                    elif etype == "error":
                        error = event.get("error", {})
                        msg = error.get("data", {}).get("message", "") if isinstance(error, dict) else str(error)
                        self._emit(AgentEvent(type="info", content=f"[error] {msg}"))

                if early_stopped:
                    break

        except ValueError:
            pass  # stdout.close() 后 for 循环抛出 I/O on closed file，正常退出

        # 停掉 watchdog（proc 即将 wait/kill 收尾，watchdog 不需要再监测）
        watchdog_stop.set()

        try:
            # 确保进程退出
            if early_stopped:
                # _killpg_safe(proc) 已在 early stop 时调用
                proc.wait(timeout=5)
            else:
                proc.wait(timeout=30)
        except subprocess.TimeoutExpired:
            _killpg_safe(proc)
            proc.wait(timeout=5)

        # stderr（early stop 时进程已 kill，不再读取避免阻塞）
        if not early_stopped:
            stderr = proc.stderr.read().decode(errors="replace").strip()
            if stderr and proc.returncode != 0:
                for line in stderr.split("\n")[:5]:
                    print(f"[opencode] {line}", file=sys.stderr)
        else:
            proc.stderr.close()

        if early_stopped:
            self._emit(AgentEvent(
                type="info", content="[early stop] 执行工具被拦截，已停止",
            ))

        return "\n".join(text_parts), events

    async def close(self) -> None:
        pass  # 无状态，无需关闭


# ── Runner ───────────────────────────────────────────────────────────────────

def _inject_model(workspace: Path, model: str | None, runner_type: str) -> None:
    """将 model 写入 workspace/.opencode/opencode.jsonc（已有则合并，没有则新建）。"""
    resolved = resolve_model(model, runner_type)
    if resolved is None:
        return
    config_dir = workspace / ".opencode"
    config_dir.mkdir(parents=True, exist_ok=True)
    config_path = config_dir / "opencode.jsonc"
    if config_path.exists():
        text = config_path.read_text(encoding="utf-8")
        stripped = re.sub(r"^\s*//.*$", "", text, flags=re.MULTILINE)
        stripped = re.sub(r",\s*([}\]])", r"\1", stripped)
        data = json.loads(stripped)
    else:
        data = {"$schema": "https://opencode.ai/config.json"}
    data["model"] = resolved
    config_path.write_text(json.dumps(data, indent=2, ensure_ascii=False), encoding="utf-8")


class OpenCodeRunner(AgentRunner):
    """
    通过 opencode run 运行 agent，注入指定配置。
    路由由 opencode.jsonc 中的 default_agent 控制。
    """

    async def start(self, workspace: Path) -> OpenCodeRunSession:
        workspace.mkdir(parents=True, exist_ok=True)

        if self.config.agent_config_dir:
            from ...installer import install_agent_config
            install_agent_config(self.config.agent_config_dir, workspace, ".opencode")

        _inject_model(workspace, self.config.model, self.config.runner_type)
        return OpenCodeRunSession(workspace, config=self.config)
