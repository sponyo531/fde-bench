"""Codex（OpenAI 第一方 CLI）。

    codex exec -C <dir> ... -- <message>          首轮
    codex exec resume <sid> ... -- <message>      续接

实测的三个坑：
  - `resume` 子命令**不接受 `-C/--cd`**（exit=2 "unexpected argument"），
    工作目录只能靠 cwd 传
  - 消息须用 `--` 分隔：澄清答案以 "- 问题" 形式回灌，开头的 `-` 会被 clap
    当成命令行参数
  - `-o` 写的是"最后一条 agent 消息"，本轮失败时**不覆盖**它 → 会读到上一轮的
    内容。实测因此把同一批问题反复问了 15 轮（194 条），故每轮先删该文件

认证走 ~/.codex/config.toml 的 model_providers（可指向任意兼容网关）。
"""

from .session import CodexRunner, CodexSession   # noqa: F401

AGENTS = {"codex": CodexRunner}
