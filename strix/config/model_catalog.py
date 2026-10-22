"""Built-in model metadata and pricing (no external model-router dependency)."""

from __future__ import annotations

from functools import lru_cache
from typing import Any, TypedDict

import tiktoken


class ModelCatalogEntry(TypedDict, total=False):
    provider: str
    max_input_tokens: int
    max_output_tokens: int
    input_cost_per_token: float
    output_cost_per_token: float
    supports_vision: bool
    supports_reasoning: bool
    supports_prompt_caching: bool
    supported_endpoints: list[str]


# Subset of models Strix tests and docs reference. Unknown models fall back to
# configured defaults and byte-length token estimates.
_MODELS: dict[str, ModelCatalogEntry] = {
    "gpt-4o": {
        "provider": "openai",
        "max_input_tokens": 128_000,
        "max_output_tokens": 16_384,
        "input_cost_per_token": 2.5e-06,
        "output_cost_per_token": 1e-05,
        "supports_vision": True,
        "supports_prompt_caching": True,
    },
    "gpt-5": {
        "provider": "openai",
        "max_input_tokens": 272_000,
        "max_output_tokens": 128_000,
        "input_cost_per_token": 1.25e-06,
        "output_cost_per_token": 1e-05,
        "supports_vision": True,
        "supports_reasoning": True,
        "supported_endpoints": ["/v1/chat/completions", "/v1/batch", "/v1/responses"],
        "supports_prompt_caching": True,
    },
    "gpt-5.4": {
        "provider": "openai",
        "max_input_tokens": 1_050_000,
        "max_output_tokens": 128_000,
        "input_cost_per_token": 2.5e-06,
        "output_cost_per_token": 1.5e-05,
        "supports_vision": True,
        "supports_reasoning": True,
        "supported_endpoints": ["/v1/chat/completions", "/v1/batch", "/v1/responses"],
        "supports_prompt_caching": True,
    },
    "gpt-5.6-sol": {
        "provider": "openai",
        "max_input_tokens": 922_000,
        "max_output_tokens": 128_000,
        "input_cost_per_token": 4e-06,
        "output_cost_per_token": 2e-05,
        "supports_vision": True,
        "supports_reasoning": True,
        "supported_endpoints": ["/v1/chat/completions", "/v1/batch", "/v1/responses"],
        "supports_prompt_caching": True,
    },
    "gpt-daybreak-blue-latest": {
        "provider": "openai",
        "max_input_tokens": 922_000,
        "max_output_tokens": 128_000,
        "input_cost_per_token": 4e-06,
        "output_cost_per_token": 2e-05,
        "supports_vision": True,
        "supports_reasoning": True,
        "supported_endpoints": ["/v1/responses"],
        "supports_prompt_caching": True,
    },
    "deepseek-v4-flash": {
        "provider": "deepseek",
        "max_input_tokens": 1_000_000,
        "max_output_tokens": 393_216,
        "input_cost_per_token": 3e-07,
        "output_cost_per_token": 1.2e-06,
        "supports_vision": True,
        "supports_reasoning": True,
        "supported_endpoints": ["/v1/chat/completions"],
        "supports_prompt_caching": True,
    },
    "deepseek/deepseek-v4-flash": {
        "provider": "deepseek",
        "max_input_tokens": 1_000_000,
        "max_output_tokens": 393_216,
        "input_cost_per_token": 3e-07,
        "output_cost_per_token": 1.2e-06,
        "supports_vision": True,
        "supports_reasoning": True,
        "supported_endpoints": ["/v1/chat/completions"],
        "supports_prompt_caching": True,
    },
    "minimax/MiniMax-M3": {
        "provider": "minimax",
        "max_input_tokens": 1_000_000,
        "max_output_tokens": 128_000,
        "input_cost_per_token": 3e-07,
        "output_cost_per_token": 1.2e-06,
        "supports_vision": True,
        "supports_reasoning": True,
        "supports_prompt_caching": True,
    },
    "grok-4.5": {
        "provider": "github_copilot",
        "max_input_tokens": 500_000,
        "max_output_tokens": 500_000,
        "input_cost_per_token": 2e-06,
        "output_cost_per_token": 6e-06,
        "supports_vision": True,
        "supports_reasoning": True,
        "supported_endpoints": ["/v1/chat/completions"],
    },
    "github_copilot/grok-4.5": {
        "provider": "github_copilot",
        "max_input_tokens": 500_000,
        "max_output_tokens": 500_000,
        "input_cost_per_token": 2e-06,
        "output_cost_per_token": 6e-06,
        "supports_vision": True,
        "supports_reasoning": True,
        "supported_endpoints": ["/v1/chat/completions"],
    },
    "example": {
        "provider": "example-provider",
        "input_cost_per_token": 1.0,
        "output_cost_per_token": 2.0,
    },
    "provider-a/example": {
        "input_cost_per_token": 1.0,
        "output_cost_per_token": 2.0,
    },
    "provider-b/example": {
        "input_cost_per_token": 3.0,
        "output_cost_per_token": 4.0,
    },
}

_STRIPPABLE_PREFIXES = (
    "openai/",
    "chatgpt/",
    "litellm/",
    "any-llm/",
    "ollama/",
    "ollama_chat/",
)


def _bare_name(model_name: str) -> str:
    name = model_name.strip().lower()
    for prefix in _STRIPPABLE_PREFIXES:
        if name.startswith(prefix):
            return name[len(prefix) :]
    return name


def catalog_entry(model_name: str) -> dict[str, Any] | None:
    """Return catalog metadata for ``model_name``, or None if unknown."""
    name = _bare_name(model_name)
    entry = _MODELS.get(name)
    if entry is None and "/" in name:
        entry = _MODELS.get(name.rsplit("/", 1)[-1])
    if entry is None:
        return None
    return dict(entry)


def catalog_entries() -> dict[str, ModelCatalogEntry]:
    """Mutable view of the built-in catalog (for tests)."""
    return _MODELS


@lru_cache(maxsize=512)
def resolve_pricing_model(model: str) -> str | None:
    """Return a catalog key that can be priced for ``model``, or None."""
    normalized = _bare_name(model.strip())
    if not normalized:
        return None

    bare_entry = _MODELS.get(normalized)
    if "/" not in normalized and isinstance(bare_entry, dict):
        provider = bare_entry.get("provider")
        if isinstance(provider, str) and provider:
            qualified = f"{provider}/{normalized}"
            if qualified in _MODELS:
                return qualified
            return qualified
    if "/" in normalized and bare_entry is not None:
        return normalized

    names = [normalized]
    if "/" in normalized:
        names.append(normalized.rsplit("/", 1)[-1])
    for name in names:
        matches = sorted(key for key in _MODELS if key.endswith(f"/{name}"))
        if not matches:
            continue
        prices = {
            (
                _MODELS[key].get("input_cost_per_token"),
                _MODELS[key].get("output_cost_per_token"),
            )
            for key in matches
        }
        if len(matches) == 1 or len(prices) == 1:
            return matches[0]
    return None


def estimate_usage_cost(
    *,
    model: str,
    prompt_tokens: int,
    completion_tokens: int,
) -> float | None:
    """Best-effort USD estimate from the local catalog."""
    resolved = resolve_pricing_model(model)
    if not resolved:
        return None
    entry = _MODELS.get(resolved)
    if not entry:
        return None
    in_rate = entry.get("input_cost_per_token")
    out_rate = entry.get("output_cost_per_token")
    if not isinstance(in_rate, int | float) or not isinstance(out_rate, int | float):
        return None
    cost = prompt_tokens * float(in_rate) + completion_tokens * float(out_rate)
    return cost if cost > 0 else None


def count_tokens(_model: str, text: str) -> int:
    """Approximate token count using tiktoken (cl100k_base), else UTF-8 byte length."""
    if not text:
        return 0
    try:
        enc = tiktoken.get_encoding("cl100k_base")
        return len(enc.encode(text))
    except Exception:  # noqa: BLE001
        return len(text.encode("utf-8"))


def model_info(model: str) -> dict[str, int]:
    """Context and output limits for ``model`` (zeros when unknown)."""
    entry = catalog_entry(model)
    if entry is None:
        return {"max_input_tokens": 0, "max_output_tokens": 0}
    return {
        "max_input_tokens": int(entry.get("max_input_tokens") or 0),
        "max_output_tokens": int(entry.get("max_output_tokens") or 0),
    }
