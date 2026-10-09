"""Public route IDs select private endpoints without embedding provider details."""

import json
import sys
from pathlib import Path

import pytest

from harness.backends._retry import gateway
from harness.config import LLMRole, make_client
from harness.defaults import resolve_model
from harness.model_endpoints import endpoint, is_responses_model
from harness.scripts.stage_opencode_config import main as stage_opencode_main, strip_jsonc


@pytest.mark.parametrize("route,prefix", [
    ("chat/grok-4.6", "AGENT"),
    ("responses/custom-model", "RESPONSES"),
    ("direct/glm-5.3", "DIRECT"),
])
def test_explicit_route_selects_its_own_endpoint(monkeypatch, route, prefix):
    for name in ("AGENT", "RESPONSES", "DIRECT"):
        monkeypatch.setenv(f"DELIVER_{name}_BASE_URL", f"https://{name.lower()}.invalid/v1/")
        monkeypatch.setenv(f"DELIVER_{name}_API_KEY", f"{name.lower()}-key")
    expected = (f"https://{prefix.lower()}.invalid/v1", f"{prefix.lower()}-key")
    assert endpoint(route) == expected
    assert gateway(route) == expected


def test_explicit_chat_route_does_not_switch_to_responses():
    assert not is_responses_model("chat/grok-4.6")
    assert is_responses_model("responses/custom-model")
    assert resolve_model("grok-4.6", "opencode") == "responses/grok-4.6"
    assert resolve_model("chat/grok-4.6", "opencode") == "chat/grok-4.6"


def test_direct_client_uses_direct_endpoint_and_bare_wire_model(monkeypatch):
    monkeypatch.setenv("DELIVER_DIRECT_BASE_URL", "https://direct.invalid/v1")
    monkeypatch.setenv("DELIVER_DIRECT_API_KEY", "direct-key")
    role = LLMRole("judge", "direct/glm-5.3", "openai", "https://old.invalid", "old-key")
    client = make_client(role)
    assert (client.base_url, client.auth_token, client.model) == (
        "https://direct.invalid/v1", "direct-key", "glm-5.3")
    assert not client._use_responses


def test_responses_client_preserves_protocol_for_unlisted_model(monkeypatch):
    monkeypatch.setenv("DELIVER_RESPONSES_BASE_URL", "https://responses.invalid/v1")
    monkeypatch.setenv("DELIVER_RESPONSES_API_KEY", "responses-key")
    role = LLMRole("judge", "responses/custom-model", "openai", "https://old.invalid", "old-key")
    client = make_client(role)
    assert (client.base_url, client.auth_token, client.model) == (
        "https://responses.invalid/v1", "responses-key", "custom-model")
    assert client._use_responses


def test_chat_client_keeps_chat_protocol_for_named_responses_model():
    role = LLMRole("judge", "chat/grok-4.6", "openai", "https://chat.invalid/v1", "chat-key")
    client = make_client(role)
    assert (client.base_url, client.auth_token, client.model) == (
        "https://chat.invalid/v1", "chat-key", "grok-4.6")
    assert not client._use_responses


def test_published_registry_is_empty_and_example_is_complete():
    root = Path(__file__).resolve().parents[2]
    for path in (root / "opencode/opencode.jsonc",
                 root / "harness/scoring/extractor_opencode/opencode.jsonc"):
        cfg = json.loads(strip_jsonc(path.read_text()))
        assert cfg.get("provider") == {}
        assert "model" not in cfg
    example = json.loads(strip_jsonc((root / "opencode-routes.example.jsonc").read_text()))
    assert set(example["provider"]) == {"chat", "responses", "direct"}
    for provider in example["provider"].values():
        assert provider["models"]
        assert provider["options"]["apiKey"].startswith("{env:DELIVER_")


def _stage(monkeypatch, source: Path, dest: Path, model: str) -> int:
    monkeypatch.setattr(sys, "argv", ["stage_opencode_config.py", "--src", str(source),
                                  "--dst", str(dest), "--require-model", model])
    return stage_opencode_main()


def test_stage_checks_model_in_private_registry(monkeypatch, tmp_path, capsys):
    source = tmp_path / "private.jsonc"
    source.write_text(json.dumps({"provider": {"direct": {
        "npm": "@ai-sdk/openai-compatible", "models": {"glm-5.3": {}}
    }}}))
    assert _stage(monkeypatch, source, tmp_path / "missing", "chat/glm-5.3") == 65
    assert "没有 provider" in capsys.readouterr().err
    assert _stage(monkeypatch, source, tmp_path / "valid", "direct/glm-5.3") == 0
    staged = json.loads((tmp_path / "valid/opencode/opencode.json").read_text())
    assert "glm-5.3" in staged["provider"]["direct"]["models"]
    assert staged.get("model") is None


def test_stage_requires_nonempty_private_registry(monkeypatch, tmp_path, capsys):
    source = tmp_path / "empty.jsonc"
    source.write_text('{"provider": {}}')
    assert _stage(monkeypatch, source, tmp_path / "staged", "direct/glm-5.3") == 65
    assert "provider 为空" in capsys.readouterr().err


def test_responses_provider_sdk_is_seeded_without_image_rebuild(tmp_path):
    from harness.isolation.environment import _seed_responses_provider_sdk

    source = Path(__file__).resolve().parents[2] / "harness/scoring/extractor_opencode/node_modules/@ai-sdk/openai"
    _seed_responses_provider_sdk(tmp_path)
    sdk = tmp_path / "opencode" / "node_modules" / "@ai-sdk" / "openai"
    assert sdk.is_symlink() if source.is_dir() else not sdk.exists()
