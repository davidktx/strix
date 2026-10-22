"""Tests for the built-in model catalog and OpenAI-compatible routing."""

from __future__ import annotations

import pytest
from agents.models.openai_chatcompletions import OpenAIChatCompletionsModel
from agents.models.openai_responses import OpenAIResponsesModel

from strix.config.model_catalog import (
    catalog_entry,
    resolve_pricing_model,
)
from strix.config.models import StrixProvider, _NonStreamingModel, _TurnGuardModel
from strix.llm import context_budget
from strix.llm.request_log import RequestLoggingModel
from strix.report.pricing import resolve_litellm_model


def test_resolve_pricing_model_known_bare_name() -> None:
    resolve_pricing_model.cache_clear()
    assert resolve_pricing_model("deepseek-v4-flash") == "deepseek/deepseek-v4-flash"
    assert resolve_litellm_model("openai/deepseek-v4-flash") == "deepseek/deepseek-v4-flash"


def test_resolve_pricing_model_provider_qualified() -> None:
    resolve_pricing_model.cache_clear()
    assert resolve_pricing_model("deepseek/deepseek-v4-flash") == "deepseek/deepseek-v4-flash"


def test_context_window_from_catalog() -> None:
    context_budget._model_info.cache_clear()
    assert context_budget.context_window("gpt-4o") == 128_000
    assert context_budget.context_window("openai/gpt-4o") == 128_000


@pytest.mark.parametrize(
    "model_name",
    [
        "anthropic/claude-sonnet-4-5",
        "openrouter/z-ai/glm-5.3",
        "litellm/openai/gw-model",
    ],
)
def test_strix_provider_passes_prefixed_names_as_model_id(
    monkeypatch: pytest.MonkeyPatch, model_name: str
) -> None:
    """``unknown_prefix_mode=model_id`` keeps provider prefixes on the wire model id."""
    monkeypatch.setenv("OPENAI_API_KEY", "test-key")
    monkeypatch.setenv("LLM_API_BASE", "https://gateway.example/v1")
    model = StrixProvider().get_model(model_name)
    while isinstance(model, _NonStreamingModel | _TurnGuardModel | RequestLoggingModel):
        model = model._inner
    assert isinstance(model, OpenAIChatCompletionsModel | OpenAIResponsesModel)
    if model_name.startswith("litellm/"):
        assert model.model == model_name.removeprefix("litellm/")
    else:
        assert model.model == model_name


def test_catalog_entry_strips_legacy_prefixes() -> None:
    entry = catalog_entry("litellm/openai/gpt-4o")
    assert entry is not None
    assert entry.get("max_input_tokens") == 128_000
