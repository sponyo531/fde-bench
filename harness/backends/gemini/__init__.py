"""Gemini CLI（Google 第一方）。

    gemini -m <model> --approval-mode yolo --skip-trust --session-id <uuid>
    gemini ... --resume latest               续接

实测的两个坑：
  - 不加 `--skip-trust` 会把 approval-mode 降级并拒绝执行
    （"not running in a trusted directory"）
  - **消息必须走 stdin**：`-p` 后跟 `-` 开头的值会被 yargs 当成新 flag
    （报 "Not enough arguments following: p"），而澄清答案正是
    "- 问题\\n  答案" 的形式。gemini 没有 `--` 转义，只能用 stdin。

认证走 GEMINI_API_KEY + GOOGLE_GEMINI_BASE_URL。注意它用的是 Gemini **原生
协议**端点（/v1beta/models），不是 OpenAI 协议——网关需同时提供该端点。
"""

from .._cli_spec import _Spec, make_runner

SPEC = _Spec(
    name="gemini", bin="gemini",
    first=("-m", "{model}", "--approval-mode", "yolo", "--skip-trust",
           "--output-format", "stream-json", "--session-id", "{sid}"),
    # --resume 用显式 session id（实测可用），不用 latest：latest 按 cwd 取最近会话，
    # 同一 workspace 内没有并发问题，但显式 id 不依赖这个假设
    resume=("-m", "{model}", "--approval-mode", "yolo", "--skip-trust",
            "--output-format", "stream-json", "--resume", "{sid}"),
    via_stdin=True,
    noise=("Ripgrep is not available", "YOLO mode is enabled"),
    output="gemini-stream-json",   # result.stats 给整轮 token（input 不含缓存 / cached）+ tool_use 事件
)

GeminiRunner = make_runner(SPEC)
AGENTS = {"gemini": GeminiRunner}
