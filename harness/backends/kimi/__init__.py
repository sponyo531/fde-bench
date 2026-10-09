"""Kimi CLI（Moonshot 第一方，装在 ~/.kimi-code）。

    kimi --print -p <message> --model <model>             首轮
    kimi --print -p <message> --model <model> --continue  续接（按 cwd 关联会话，无显式 id）

实测的坑：模型必须先在 `~/.kimi-code/config.toml` 注册，否则报
`Model "X" is not configured`。形如：

    [models."responses/kimi-k3"]
    provider = "responses"
    model = "kimi-k3"
    max_context_size = 128000

因此模型名要**保留 provider 前缀**（与 claude/gemini 相反），见 _strip_provider。

**max_context_size 不能设小**。kimi 的压缩触发是 `used ≥ max×0.85` 或
`used + 50k(保留) ≥ max`；写 128000 时有效阈值只有 78k，harness prompt + 系统提示 +
读一遍千行 CSV 就到 ~89k → **每一步之后都 full compaction**，工作记忆清空，模型每轮
从"I will inspect the files…"重来，run 68 秒 status=ok 零产物零提问（2026-09-02 实测）。
现统一为 262144（与 config.toml [agent].default_context_tokens、dsh 一致；opencode 侧
给 kimi-k3 声明的是 1M）。
"""

from .._cli_spec import _Spec, make_runner

SPEC = _Spec(
    name="kimi", bin="kimi",
    # kimi-cli 1.49.0 起，-p/--prompt 不再隐含非交互 print UI；JSON 输出必须
    # 显式配 --print。小写 -c 也是 --command（prompt）的别名，续接是大写 -C /
    # --continue。两处都写长选项，避免大小写再次被误读。
    first=("--print", "-p", "{message}", "--model", "{model}",
           "--output-format", "stream-json"),
    resume=("--print", "-p", "{message}", "--model", "{model}",
            "--continue", "--output-format", "stream-json"),
    needs_session_id=False,
    noise=("To resume this session:",),
    output="kimi-stream-json",     # stdout 无 usage；逐 step 用量从独立 KIMI_SHARE_DIR 的 wire.jsonl 取
)

KimiRunner = make_runner(SPEC)
AGENTS = {"kimi": KimiRunner}
