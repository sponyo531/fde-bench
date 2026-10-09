"""opencode。两种形态，差别在**澄清通道**：

    opencode（默认）  serve 常驻 + HTTP，原生 `question` tool 可用 → C/CF 能跑
    opencode-run      CLI 单发，question 权限被硬编码 deny → 只能跑 R/F

serve 形态是本 benchmark 里唯一走「原生提问 tool」的脚手架（其余都走文本多轮），
因此也是「结构化提问 vs 文本提问」对照的唯一来源。

另：`run --session <id>` 续接在本环境会挂起，所以多轮只能用 serve。
"""

from .run import OpenCodeRunner              # noqa: F401
from .serve import OpenCodeServeRunner, OpenCodeServeSession   # noqa: F401

AGENTS = {
    "opencode": OpenCodeServeRunner,
    "opencode-run": OpenCodeRunner,
}
