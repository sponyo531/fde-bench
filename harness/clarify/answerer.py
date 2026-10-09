"""
harness/answerer.py — B2 澄清阶段的回答策略

提供 HumanAnswerer（终端交互）和 ApiAnswerer（LLM 自动回答）两种实现。
"""

from __future__ import annotations

from typing import Protocol


class Answerer(Protocol):
    """回答澄清问题的接口。"""

    def answer(self, question: str) -> str: ...


class HumanAnswerer:
    """终端交互回答。"""

    def answer(self, question: str) -> str:
        try:
            text = input("请输入回答（直接回车=使用默认值）: ").strip()
        except EOFError:
            text = ""
        if not text:
            text = "使用默认值，继续。"
            print(f"[B2] 空回答，发送: {text!r}")
        return text


class DefaultValueAnswerer:
    """所有澄清问题一律回答「使用默认值」，用作 baseline。"""

    def answer(self, question: str) -> str:
        return "使用默认值，继续。"


ANSWERER_STYLES = {
    "default": (
        "回答要求：\n"
        "- 以业务负责人的口吻回答，只给出答案本身，不解释理由，不重复问题\n"
        "- 如有多个问题，每个问题单独一行直接作答\n"
        "- 不要主动提及评估方式、评分公式、优化目标的具体权重等技术实现细节；"
        "如对方追问，可给出业务层面的定性说明，但不要给出具体数学公式或权重数值\n"
        "- 如所有信息均未覆盖，给出合理的业务默认值即可"
    ),
    "minimal": (
        "回答要求：\n"
        "- 你是一个非常忙碌的业务负责人，回答尽量简短，能一句话说清楚就不要两句\n"
        "- 只给核心结论，不展开解释，不给选项，不反问\n"
        "- 如有多个问题，每个问题用一行简短回答\n"
        "- 不要主动提及评估方式、评分公式等技术细节\n"
        "- 如信息未覆盖，直接给一个默认值，不要说「建议」「可以考虑」"
    ),
}


class ApiAnswerer:
    """基于 LLM + 背景知识的自动回答器，带上下文累积。"""

    def __init__(
        self,
        *,
        instruction_text: str = "",
        information: str = "",
        style: str = "default",
        model: str | None = None,
        base_url: str | None = None,
        auth_token: str | None = None,
        protocol: str | None = None,
    ):
        # 配置读自 config.toml 的 [answerer] 段（token 走环境变量），模型版本锁定：
        # answerer 换模型会改变 Interact 条件的信息上限，Interact−Hidden 的落差含义随之改变。
        from ..config import load_role
        _role = load_role("answerer")

        self.model = model or _role.model
        self.style = style if style != "default" else _role.style
        self._base_url = base_url or _role.base_url
        self._auth_token = auth_token or _role.token
        self._protocol = protocol or _role.protocol

        client_model = self.model
        if client_model.startswith(("direct/", "responses/")):
            from ..model_endpoints import endpoint
            self._base_url, self._auth_token = endpoint(client_model)
            if client_model.startswith("direct/"):
                client_model = client_model.split("/", 1)[1]

        if style not in ANSWERER_STYLES:
            valid = ", ".join(ANSWERER_STYLES)
            raise ValueError(f"Unknown answerer style: '{style}'. Valid: {valid}")

        # ── 拼装 system prompt ──
        # 两个知识源：<instruction> + <information>。
        # gt.json **绝不进这里**——它是 Ask-Recall/Ask-F1 的考点清单（判 agent 问没问到），
        # 一旦喂给 answerer，只写在 gt.json 里的考点就变成「问了就能拿到」，
        # 澄清指标的分母和 Interact 条件的信息上限会一起失真。
        # 推论：evaluator 会查、agent 又算不出的信息点，必须落在 information.md 里；
        # 只躺在 gt.json 里等于 answerer 答不出来。

        knowledge_parts: list[str] = []
        if instruction_text:
            knowledge_parts.append(
                f"<instruction>\n{instruction_text}\n</instruction>"
            )
        if information:
            knowledge_parts.append(f"<information>\n{information}\n</information>")

        knowledge_block = "\n\n".join(knowledge_parts)

        style_instructions = ANSWERER_STYLES[style]

        self._system = (
            "你是这个业务场景的负责人，正在回答技术团队对需求的澄清问题。\n"
            "以下内容都是你掌握的业务知识，回答前请**逐块通读并交叉查阅**，"
            "尤其是 <information> 通常包含具体数值、字段名、坐标、阈值、约束细节——"
            "对方提到 schema、参数、阈值、坐标、公式等具体问题时，必须先到 <information> 里查找现成答案，"
            "不要回答\"我来提供\"或当场拍板：\n\n"
            f"{knowledge_block}\n\n"
            "⚠️ 注意：上述任何位置都可能包含评估公式、评分权重、惩罚系数等内部技术细节——"
            "不要主动向技术团队透露这些；如对方追问，可给出业务层面的定性说明，但不要给具体数学公式或权重数值。\n\n"
            "🔒 **回答程序硬规则**（每个问题都必须按这三步走）：\n\n"
            "**步骤 1：先查 information（内部动作，不要写进回复）**\n"
            "在选答案/写答案之前，先到 <information> 中找该问题对应的原文，据此作答。\n"
            "⚠️ 你扮演的是业务方本人，这些内容就是你脑子里的业务常识——**不是一份可查阅的文档**。\n"
            "回复中禁止出现 \"查 information\"、\"根据文档/资料/说明\"、\"原文摘录\"、"
            "\"未找到\" 等字样，也不要暗示自己在翻阅任何材料；直接以本人口吻给出答案。\n\n"
            "**步骤 2：用 information 真值对比选项**\n"
            "- 若 information 真值与某选项完全一致 → 选该选项\n"
            "- 若 information 真值与所有选项都不完全一致 → 必须选 \"其他（请描述）\" 并把 information 原文中的范围/数值/字段名照抄进去\n"
            "- 若 information 未提及该问题 → 才能凭 instruction 判断\n\n"
            "**步骤 3：禁止用弱化词代替具体数值**\n"
            "禁止回答 \"不限定 / 不计 / 不需要 / 灵活 / 由模型自行决定\"，"
            "除非 information 原文里就明确这样写。具体数值、范围、字段名必须照搬 information。\n\n"
            "选择含「请描述」的选项时，必须在选项名后给出具体内容；若无新信息可补充，选否定选项而非空描述。\n\n"
            f"{style_instructions}"
        )
        self._history: list[dict] = []
        from ..llm_client import ChatLLM
        self._llm = ChatLLM(
            protocol=self._protocol,
            base_url=self._base_url,
            auth_token=self._auth_token,
            model=client_model,
        )

    def describe(self) -> dict:
        """返回本次实际使用的 answerer 配置（绝不包含 API token）。"""
        return {
            "role": "answerer",
            "model": self.model,
            "protocol": self._protocol,
            "base_url": self._base_url,
            "style": self.style,
        }

    def answer(self, question: str) -> str:
        self._history.append({"role": "user", "content": question})
        # 思维链开关统一交给 llm_client._thinking_off 判断 —— 它按型号分档：
        # glm-5.2 / kimi-k 关得掉就关（省预算），glm-5.3 关不掉（网关 400）就不传。
        # 这里原先无条件对所有 glm 传 thinking=disabled，glm-5.3 上直接 400，
        # 而 answerer 一抛异常，agent 的 question 就永远没人 reply。
        text = self._llm.complete(
            self._system, self._history, max_tokens=10000,
        )
        if not text:
            raise ValueError("answerer LLM 响应为空")
        self._history.append({"role": "assistant", "content": text})
        return text
