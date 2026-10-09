"""Claude Code（Anthropic 第一方 CLI）。

    claude -p --session-id <uuid> ...        首轮
    claude -p --resume <uuid> ...            续接

实测的两个坑：
  - root 下 `--dangerously-skip-permissions` 被硬拒，须设 IS_SANDBOX=1
  - `AskUserQuestion` 在二进制里存在但 headless 的 25 个工具**不注册它**（2.1.224 实测），
    `allowed_tools` / `--brief` / CLAUDE_CODE_AUTO_MODE_CLASSIFY_ASK_USER_QUESTION
    都解锁不了。所以澄清只能走文本多轮。

认证走 ANTHROPIC_BASE_URL + ANTHROPIC_AUTH_TOKEN（可指向任意兼容网关）。
"""

from .._cli_spec import _Spec, make_runner

# --output-format stream-json 必须配 --verbose（-p 下否则拒绝）。不开 JSON 时
# stdout 是纯文本：tokens / cache / cost / 工具调用全部采不到（2026-09-02 实测
# 真实 run tokens=None），且 tool_calls 会被记成假零。
SPEC = _Spec(
    name="claude-code", bin="claude",
    first=("-p", "--session-id", "{sid}", "--model", "{model}",
           "--output-format", "stream-json", "--verbose",
           "--dangerously-skip-permissions"),
    resume=("-p", "--resume", "{sid}", "--model", "{model}",
            "--output-format", "stream-json", "--verbose",
            "--dangerously-skip-permissions"),
    env={"IS_SANDBOX": "1"},
    noise=("is not a model this version of Claude Code recognizes",),
    output="claude-stream-json",
)

ClaudeCodeRunner = make_runner(SPEC)
AGENTS = {"claude-code": ClaudeCodeRunner}
