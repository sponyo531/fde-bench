import os
from types import SimpleNamespace

from harness.llm_client import ChatLLM, _thinking_off


def test_deepseek_flash_disables_thinking_for_auxiliary_calls():
    assert _thinking_off("ds/deepseek-v4-flash") == {
        "thinking": {"type": "disabled"}
    }


def test_deepseek_flash_request_contains_thinking_switch(monkeypatch):
    class Client:
        def __init__(self):
            self.kwargs = None

        def chat_completions_create(self, **kwargs):
            self.kwargs = kwargs
            return SimpleNamespace(
                choices=[SimpleNamespace(message=SimpleNamespace(content='{"ok":true}'))]
            )

        class _Chat:
            def __init__(self, outer):
                self.completions = SimpleNamespace(create=outer.chat_completions_create)

    client = Client()
    client.chat = client._Chat(client)
    llm = ChatLLM(protocol="openai", base_url="http://unused",
                   auth_token="token", model="ds/deepseek-v4-flash")
    monkeypatch.setattr(llm, "_build_client", lambda: client)

    assert llm._complete_once("system", [{"role": "user", "content": "x"}]) == '{"ok":true}'
    assert client.kwargs["extra_body"] == {"thinking": {"type": "disabled"}}


def test_responses_model_uses_responses_and_strips_provider(monkeypatch):
    class Client:
        def __init__(self):
            self.kwargs = None
            self.responses = SimpleNamespace(create=self.create)

        def create(self, **kwargs):
            self.kwargs = kwargs
            return SimpleNamespace(
                output_text="",
                output=[
                    SimpleNamespace(type="reasoning", content=[]),
                    SimpleNamespace(type="message", content=[
                        SimpleNamespace(type="output_text", text="OK")
                    ]),
                ],
            )

    client = Client()
    llm = ChatLLM(protocol="openai", base_url="http://old.invalid/v1",
                  auth_token="old", model="responses/gpt-6-astra")
    monkeypatch.setattr(llm, "_build_client", lambda: client)

    assert llm._complete_once(
        "system", [{"role": "user", "content": "x"}], max_tokens=12,
        temperature=0.2,
    ) == "OK"
    assert llm.model == "gpt-6-astra"
    assert llm.base_url == os.environ.get("DELIVER_RESPONSES_BASE_URL", "")
    assert client.kwargs == {
        "model": "gpt-6-astra",
        "instructions": "system",
        "input": [{"role": "user", "content": "x"}],
        "max_output_tokens": 256,
    }
