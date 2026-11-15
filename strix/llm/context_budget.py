"""Model-aware token budgets from the built-in catalog with safe fallbacks."""

from __future__ import annotations

import logging
from functools import lru_cache

from strix.config import load_settings
from strix.config.model_catalog import count_tokens as catalog_count_tokens
from strix.config.model_catalog import model_info as catalog_model_info


logger = logging.getLogger(__name__)

_DEFAULT_OUTPUT_TOKENS = 8_192


@lru_cache(maxsize=128)
def _model_info(model: str) -> dict[str, int]:
    lookup_key = model
    if model.startswith("chatgpt/"):
        lookup_key = model[len("chatgpt/") :]
    info = catalog_model_info(lookup_key)
    if info["max_input_tokens"] or info["max_output_tokens"]:
        return info
    if lookup_key != model:
        info = catalog_model_info(model)
    if info["max_input_tokens"] or info["max_output_tokens"]:
        return info
    logger.debug("No catalog entry for %r; using configured fallbacks", model)
    return {"max_input_tokens": 0, "max_output_tokens": 0}


def context_window(model: str) -> int:
    """Input token capacity for ``model`` (configured fallback when unmapped)."""
    resolved = _model_info(model)["max_input_tokens"]
    return resolved or load_settings().context.fallback_context_tokens


def output_limit(model: str) -> int:
    """Max output tokens for ``model`` (a conservative default when unmapped)."""
    return _model_info(model)["max_output_tokens"] or _DEFAULT_OUTPUT_TOKENS


def count_tokens(model: str, text: str) -> int:
    """Token count for ``text`` under ``model`` (catalog tokenizer with byte fallback)."""
    return catalog_count_tokens(model, text)
