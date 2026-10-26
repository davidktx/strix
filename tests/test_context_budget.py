"""Tests for model-aware token budgets."""

from __future__ import annotations

from typing import TYPE_CHECKING

from strix.config import load_settings
from strix.llm import context_budget


if TYPE_CHECKING:
    import pytest


def test_context_window_known_model() -> None:
    context_budget._model_info.cache_clear()
    assert context_budget.context_window("gpt-4o") == 128_000


def test_context_window_strips_provider_prefix() -> None:
    context_budget._model_info.cache_clear()
    assert context_budget.context_window("openai/gpt-4o") == 128_000


def test_context_window_chatgpt_prefix_skips_provider_auth(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    context_budget._model_info.cache_clear()
    calls: list[str] = []

    def _model_info(model: str) -> dict[str, int]:
        calls.append(model)
        return {"max_input_tokens": 1_050_000, "max_output_tokens": 128_000}

    monkeypatch.setattr(context_budget, "catalog_model_info", _model_info)
    try:
        assert context_budget.context_window("chatgpt/gpt-5.6-luna") == 1_050_000
        assert calls == ["gpt-5.6-luna"]
    finally:
        context_budget._model_info.cache_clear()


def test_context_window_unmapped_uses_fallback(monkeypatch: pytest.MonkeyPatch) -> None:
    context_budget._model_info.cache_clear()

    def _empty(_model: str) -> dict[str, int]:
        return {"max_input_tokens": 0, "max_output_tokens": 0}

    monkeypatch.setattr(context_budget, "catalog_model_info", _empty)
    expected = load_settings().context.fallback_context_tokens
    assert context_budget.context_window("totally-made-up-model") == expected
    context_budget._model_info.cache_clear()


def test_count_tokens_fallback_on_tokenizer_error(monkeypatch: pytest.MonkeyPatch) -> None:
    def _raise(_name: str) -> object:
        raise RuntimeError("no tokenizer")

    monkeypatch.setattr("tiktoken.get_encoding", _raise)
    assert context_budget.count_tokens("weird-model", "x" * 400) == 400
    assert context_budget.count_tokens("weird-model", "😀" * 10) == 40


def test_count_tokens_empty_is_zero() -> None:
    assert context_budget.count_tokens("gpt-4o", "") == 0
