"""文本提问预筛使用与最终 judge 相同的五模型多数投票。"""

from __future__ import annotations

import importlib

import pytest

from harness.clarify import detect


class _Client:
    def __init__(self, model: str):
        self.model = model


@pytest.fixture(autouse=True)
def _restore_detect_module():
    # test_clarify_text 兼容旧式写法会直接替换模块函数；重载确保本文件
    # 验证的是生产实现，而不是前一个测试留下的 lambda。
    importlib.reload(detect)
    yield


def test_detect_uses_strict_majority(monkeypatch):
    clients = [(f"m{i}", _Client(f"m{i}")) for i in range(5)]

    def fake_once(llm, model, text, hint):
        return ["参数是多少？"] if model in {"m0", "m1", "m2"} else []

    monkeypatch.setattr(detect, "_llms", lambda: clients)
    monkeypatch.setattr(detect, "_detect_once", fake_once)
    assert detect.detect("请确认？") == ["参数是多少？"]


def test_detect_tie_or_minority_is_not_asking(monkeypatch):
    clients = [(f"m{i}", _Client(f"m{i}")) for i in range(5)]

    def fake_once(llm, model, text, hint):
        return ["参数是多少？"] if model in {"m0", "m1"} else []

    monkeypatch.setattr(detect, "_llms", lambda: clients)
    monkeypatch.setattr(detect, "_detect_once", fake_once)
    assert detect.detect("请确认？") == []


def test_detect_without_question_mark_does_not_initialize_judges(monkeypatch):
    monkeypatch.setattr(detect, "_llms", lambda: (_ for _ in ()).throw(
        AssertionError("无问号不应创建 judge client")))
    assert detect.detect("已完成工作。") == []


def test_detect_takes_one_ballot_not_union(monkeypatch):
    """四票措辞各异时不得取并集：12 问不能膨胀成 46 条。取问句数居中的那份。"""
    clients = [(f"m{i}", _Client(f"m{i}")) for i in range(5)]
    ballots = {
        "m0": ["A？", "B？"],                       # 切得最粗
        "m1": ["A 呢？", "B 呢？", "C 呢？"],         # 居中（3 条）
        "m2": ["A?", "B?", "C?", "D?", "E?"],       # 切得最碎
        "m3": [],                                   # 认为没在提问
        "m4": None,                                 # 调用失败
    }

    def fake_once(llm, model, text, hint):
        if ballots[model] is None:
            raise RuntimeError("503")
        return ballots[model]

    monkeypatch.setattr(detect, "_llms", lambda: clients)
    monkeypatch.setattr(detect, "_detect_once", fake_once)
    # 有效 4 票，3 票在提问 → 通过；问句取 3 条那份，而不是 2+3+5 的并集
    assert detect.detect("请确认？") == ["A 呢？", "B 呢？", "C 呢？"]


def test_detect_all_judges_failing_is_loud(monkeypatch, capsys):
    """五票全失败要在 stderr 留痕——它和"没在提问"长得一样，静默过就查不到。"""
    clients = [(f"m{i}", _Client(f"m{i}")) for i in range(5)]

    def fake_once(llm, model, text, hint):
        raise RuntimeError("503 model_not_found")

    monkeypatch.setattr(detect, "_llms", lambda: clients)
    monkeypatch.setattr(detect, "_detect_once", fake_once)
    assert detect.detect("请确认？") == []
    assert "judge 调用失败" in capsys.readouterr().err


def test_detect_single_judge_retries_malformed_json():
    """单个 detector judge 的格式错误应重试，而不是静默丢掉本轮提问。"""
    class Client:
        model = "chat/test"

        def __init__(self):
            self.calls = 0

        def complete(self, *args, **kwargs):
            self.calls += 1
            if self.calls == 1:
                return '{"asking": true, "questions":'
            return '{"asking": true, "questions": ["参数是多少？"]}'

    client = Client()
    assert detect._detect_once(client, client.model, "参数？", "") == ["参数是多少？"]
    assert client.calls == 2


def test_detect_parser_accepts_markdown_and_trailing_text():
    raw = '```json\n{"asking": true, "questions": ["参数是多少？"]}\n```\n完成。'
    assert detect._parse(raw) == ["参数是多少？"]


def test_llms_keep_provider_route_for_client_selection(monkeypatch):
    """完整 provider ID 交给 make_client 选择端点，审计标签保持一致。"""
    from dataclasses import dataclass

    @dataclass
    class _Role:
        model: str = "x"
        protocol: str = "openai"
        base_url: str = "http://gw"
        token: str = "t"

    built = []
    monkeypatch.setattr("harness.config.load_role", lambda role: _Role())
    monkeypatch.setattr("harness.config.make_client",
                        lambda role: built.append(role.model) or role.model)
    labels = [m for m, _ in detect._llms()]
    assert labels == [m for _, m in detect.JUDGE_MODELS]         # 标签保留前缀
    assert built == labels
    assert "direct/glm-5.2" in built


# ── 软标记快捷通道 ────────────────────────────────────────────────────────────

_NO_LLM = lambda: (_ for _ in ()).throw(AssertionError("软标记命中时不应调 judge"))  # noqa: E731


def test_marked_block_is_taken_verbatim_without_llm(monkeypatch):
    monkeypatch.setattr(detect, "_llms", _NO_LLM)
    text = """我看了数据，1334 个网点，总重 60903 kg。

## Questions for the client
1. 网点编码=0 的那一行是仓库吗？
2. 「总重量」单位是公斤吗？
- 12 辆车必须全部出动吗
"""
    assert detect.detect(text) == [
        "网点编码=0 的那一行是仓库吗？",
        "「总重量」单位是公斤吗？",
        "12 辆车必须全部出动吗",          # 没问号也算：agent 用了标记就是在问
    ]


def test_marked_block_stops_at_next_heading(monkeypatch):
    monkeypatch.setattr(detect, "_llms", _NO_LLM)
    text = """## Questions for the client
- 仓库在哪？

## Plan
1. 读数据 — 这一行不是问题？
"""
    assert detect.detect(text) == ["仓库在哪？"]


def test_heading_variants(monkeypatch):
    monkeypatch.setattr(detect, "_llms", _NO_LLM)
    for h in ("## Questions for the client", "### QUESTION FOR THE CLIENT:",
              "# questions for the client："):
        assert detect.detect(f"{h}\n* 单位？\n") == ["单位？"], h


def test_generic_questions_heading_does_not_count(monkeypatch):
    """交付报告里的 'Open questions' 小节不能被当成提问；应落回语义判定。"""
    clients = [(f"m{i}", _Client(f"m{i}")) for i in range(5)]
    monkeypatch.setattr(detect, "_llms", lambda: clients)
    monkeypatch.setattr(detect, "_detect_once", lambda *a: [])     # judge 说没在问
    text = "交付完成。\n\n## Open Questions\n- 未来可否用真实路网？\n"
    assert detect.extract_marked(text) == []
    assert detect.detect(text) == []


def test_empty_marked_block_falls_back_to_semantic(monkeypatch):
    clients = [(f"m{i}", _Client(f"m{i}")) for i in range(5)]
    monkeypatch.setattr(detect, "_llms", lambda: clients)
    monkeypatch.setattr(detect, "_detect_once", lambda *a: ["速度多少？"])
    text = "我需要先确认：速度多少？\n\n## Questions for the client\n\n"   # 标题下是空的
    assert detect.extract_marked(text) == []
    assert detect.detect(text) == ["速度多少？"]
