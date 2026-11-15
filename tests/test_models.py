"""Tests for LLM model configuration helpers."""

from __future__ import annotations

import pytest
from agents.model_settings import ModelSettings
from agents.models import _openai_shared
from agents.models.openai_chatcompletions import OpenAIChatCompletionsModel
from agents.models.openai_responses import OpenAIResponsesModel

from strix.config import models
from strix.config.model_catalog import catalog_entries
from strix.config.models import (
    StrixProvider,
    _NonStreamingModel,
    _TurnGuardModel,
    configure_sdk_model_defaults,
    model_supports_images,
    request_timeout_extra_args,
    resolve_api_type,
    routes_through_litellm,
    supports_strict_tool_schemas,
    uses_chat_completions_tool_schema,
)
from strix.config.settings import Settings
from strix.llm.request_log import RequestLoggingModel


def test_request_timeout_extra_args_positive() -> None:
    assert request_timeout_extra_args(300) == {"timeout": 300}
    assert request_timeout_extra_args(10) == {"timeout": 10}


def test_request_timeout_extra_args_survives_model_settings_json_dump() -> None:
    settings = ModelSettings(extra_args=request_timeout_extra_args(300))
    assert settings.to_json_dict()["extra_args"] == {"timeout": 300}


@pytest.mark.parametrize("value", [None, 0, -1])
def test_request_timeout_extra_args_disabled(value: float | None) -> None:
    assert request_timeout_extra_args(value) is None


@pytest.mark.parametrize(
    "model_name",
    [
        "anthropic/claude-sonnet-4-6",
        "bedrock/anthropic.claude-opus-4-8-v1:0",
        "vertex_ai/claude-sonnet-5",
        "Sonnet-5",
    ],
)
def test_claude_routes_reject_strict_tool_schemas(model_name: str) -> None:
    assert not supports_strict_tool_schemas(model_name)


@pytest.mark.parametrize(
    "model_name",
    ["openai/gpt-5.4", "gpt-5.4", "gemini/gemini-3.1-pro-preview", "deepseek/deepseek-v4"],
)
def test_other_routes_keep_strict_tool_schemas(model_name: str) -> None:
    assert supports_strict_tool_schemas(model_name)


@pytest.mark.parametrize(
    "model_name",
    [
        "claude-sonnet-4-5",
        "openai/claude-sonnet-4-5",
        "any-llm/anthropic/claude-sonnet-4-5",
        "anthropic/claude-sonnet-4-5",
        "litellm/anthropic/claude-sonnet-4-5",
        "bedrock/anthropic.claude-sonnet-4-5-20250929-v1:0",
        "ollama/llama3",
    ],
)
def test_routes_through_litellm_is_always_false(model_name: str) -> None:
    assert routes_through_litellm(model_name) is False


@pytest.mark.parametrize(
    "model_name",
    [
        "anthropic/claude-sonnet-4-5",
        "openrouter/z-ai/glm-5.3",
        "litellm/openai/gpt-4o",
    ],
)
def test_strix_provider_uses_openai_compatible_sdk(
    monkeypatch: pytest.MonkeyPatch, model_name: str
) -> None:
    monkeypatch.setenv("OPENAI_API_KEY", "test-key")
    monkeypatch.setenv("LLM_API_BASE", "https://gateway.example/v1")
    model = StrixProvider().get_model(model_name)
    while isinstance(model, _NonStreamingModel | _TurnGuardModel | RequestLoggingModel):
        model = model._inner
    assert isinstance(model, OpenAIChatCompletionsModel | OpenAIResponsesModel)


def test_api_type_override_settings(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("STRIX_LLM", "gpt-4")
    monkeypatch.setenv("STRIX_API_TYPE", "chat_completions")
    assert uses_chat_completions_tool_schema("gpt-4", Settings()) is True
    monkeypatch.setenv("STRIX_LLM", "openai/gpt-4")
    monkeypatch.setenv("STRIX_API_TYPE", "responses")
    assert uses_chat_completions_tool_schema("openai/gpt-4", Settings()) is False
    monkeypatch.setenv("STRIX_LLM", "anthropic/claude-sonnet-4-5")
    assert uses_chat_completions_tool_schema("anthropic/claude-sonnet-4-5", Settings()) is True


@pytest.mark.parametrize(
    ("api_type", "expected"),
    [
        (None, OpenAIResponsesModel),
        ("chat_completions", OpenAIChatCompletionsModel),
        ("responses", OpenAIResponsesModel),
    ],
)
def test_api_type_overrides_the_api_base_route(
    monkeypatch: pytest.MonkeyPatch, api_type: str | None, expected: type
) -> None:
    """gpt-5 is catalogued on /v1/responses, so a base URL alone changes nothing."""
    monkeypatch.setattr(_openai_shared, "_use_responses_by_default", True)
    monkeypatch.setattr(_openai_shared, "_default_openai_client", None)
    monkeypatch.setattr(_openai_shared, "_default_openai_key", None)
    monkeypatch.setenv("OPENAI_API_KEY", "test-key")
    monkeypatch.setenv("OPENAI_BASE_URL", "")
    monkeypatch.setenv("STRIX_LLM", "gpt-5")
    monkeypatch.setenv("LLM_API_KEY", "test-key")
    monkeypatch.setenv("LLM_API_BASE", "https://gateway.example/v1")
    monkeypatch.delenv("STRIX_API_TYPE", raising=False)
    if api_type is not None:
        monkeypatch.setenv("STRIX_API_TYPE", api_type)
    configure_sdk_model_defaults(Settings())
    model = StrixProvider().get_model("gpt-5")
    while isinstance(model, _NonStreamingModel | _TurnGuardModel | RequestLoggingModel):
        model = model._inner
    assert isinstance(model, expected)


def _settings(monkeypatch: pytest.MonkeyPatch, model: str, api_base: str | None) -> Settings:
    monkeypatch.setenv("STRIX_LLM", model)
    monkeypatch.delenv("STRIX_API_TYPE", raising=False)
    for name in ("LLM_API_BASE", "OPENAI_API_BASE", "OPENAI_BASE_URL"):
        monkeypatch.delenv(name, raising=False)
    if api_base is not None:
        monkeypatch.setenv("LLM_API_BASE", api_base)
    return Settings()


@pytest.mark.parametrize(
    "api_base", [None, "https://api.openai.com/v1", "https://gateway.example/v1"]
)
def test_resolve_api_type_follows_the_catalog_not_the_base_url(
    monkeypatch: pytest.MonkeyPatch, api_base: str | None
) -> None:
    settings = _settings(monkeypatch, "gpt-5", api_base)
    for model in ("gpt-5", "gpt-5.6-sol", "openai/gpt-5.4", "gpt-daybreak-blue-latest"):
        assert resolve_api_type(model, settings) == "responses", model
        assert uses_chat_completions_tool_schema(model, settings) is False, model
    for model in ("gpt-4o", "my-private-model"):
        assert resolve_api_type(model, settings) == "chat_completions", model
        assert uses_chat_completions_tool_schema(model, settings) is True, model


def test_resolve_api_type_explicit_override_wins(monkeypatch: pytest.MonkeyPatch) -> None:
    _settings(monkeypatch, "gpt-daybreak-blue-latest", "https://gateway.example/v1")
    monkeypatch.setenv("STRIX_API_TYPE", "chat_completions")
    assert resolve_api_type("gpt-daybreak-blue-latest", Settings()) == "chat_completions"
    monkeypatch.setenv("STRIX_API_TYPE", "Responses")
    assert resolve_api_type("gpt-5", Settings()) == "responses"


@pytest.mark.parametrize(
    ("raw", "expected"),
    [("None", "none"), ("HIGH", "high"), (" xhigh ", "xhigh"), ("Max", "max")],
)
def test_reasoning_effort_is_case_insensitive(
    monkeypatch: pytest.MonkeyPatch, raw: str, expected: str
) -> None:
    monkeypatch.setenv("STRIX_REASONING_EFFORT", raw)
    monkeypatch.setenv("STRIX_DEDUPE_REASONING_EFFORT", raw)
    settings = Settings()
    assert settings.llm.reasoning_effort == expected
    assert settings.dedupe.reasoning_effort == expected


def test_configure_sdk_api_route_follows_the_given_model(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    routes: list[str] = []
    monkeypatch.setattr(models, "set_default_openai_api", routes.append)
    settings = _settings(monkeypatch, "gpt-5", "https://gateway.example/v1")

    models.configure_sdk_api_route("gpt-5", settings)
    models.configure_sdk_api_route("my-private-model", settings)

    assert routes == ["responses", "chat_completions"]


def test_image_support_follows_the_catalog() -> None:
    entries = catalog_entries()
    entries["acme-text"] = {"supports_vision": False}
    entries["acme-vision"] = {"supports_vision": True}
    try:
        assert model_supports_images("acme-text") is False
        assert model_supports_images("litellm/openai/acme-vision") is True
        assert model_supports_images("acme-unknown-model") is True
    finally:
        entries.pop("acme-text", None)
        entries.pop("acme-vision", None)
