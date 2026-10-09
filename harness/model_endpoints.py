"""Public model-route names and private endpoint lookup."""

from __future__ import annotations

import os

RESPONSES_MODELS = frozenset({"grok-4.6", "gpt-6-astra", "gpt-5.6-sol", "kimi-k3"})


def bare_model(model: str) -> str:
    return model.split("/", 1)[-1]


def is_responses_model(model: str | None) -> bool:
    if model and "/" in model:
        return model.startswith("responses/")
    return bare_model(model or "") in RESPONSES_MODELS


def endpoint(model: str) -> tuple[str, str]:
    """Return the URL and key for a model's configured wire protocol."""
    if model.startswith("direct/"):
        prefix = "DELIVER_DIRECT"
    elif is_responses_model(model):
        prefix = "DELIVER_RESPONSES"
    else:
        prefix = "DELIVER_AGENT"
    return (os.environ.get(f"{prefix}_BASE_URL", "").strip().rstrip("/"),
            os.environ.get(f"{prefix}_API_KEY", "").strip())
