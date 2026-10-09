"""answerer 配置必须可复现地写入 manifest，且不得包含密钥。"""

from harness.clarify.answerer import ApiAnswerer
from harness.config import LLMRole


def test_api_answerer_describe_excludes_token(monkeypatch):
    role = LLMRole(
        role="answerer",
        model="chat/test-answerer",
        protocol="openai",
        base_url="https://example.invalid/v1",
        token="secret-token",
        style="default",
    )

    class StubLLM:
        def __init__(self, **kwargs):
            self.kwargs = kwargs

    monkeypatch.setattr("harness.config.load_role", lambda name: role)
    monkeypatch.setattr("harness.llm_client.ChatLLM", StubLLM)

    answerer = ApiAnswerer(instruction_text="需求", information="背景")
    assert answerer.describe() == {
        "role": "answerer",
        "model": "chat/test-answerer",
        "protocol": "openai",
        "base_url": "https://example.invalid/v1",
        "style": "default",
    }
    assert "secret-token" not in repr(answerer.describe())


def test_update_manifest_keeps_answerer_fields(tmp_path):
    import json

    from harness.run import update_manifest

    path = tmp_path / "manifest.json"
    path.write_text(json.dumps({"case": "demo"}), encoding="utf-8")
    update_manifest(
        tmp_path,
        answerer_model="chat/test-answerer",
        answerer_config={
            "role": "answerer",
            "model": "chat/test-answerer",
            "protocol": "openai",
            "base_url": "https://example.invalid/v1",
            "style": "default",
        },
    )
    data = json.loads(path.read_text(encoding="utf-8"))
    assert data["case"] == "demo"
    assert data["answerer_model"] == "chat/test-answerer"
    assert data["answerer_config"]["role"] == "answerer"
    assert "token" not in data["answerer_config"]
