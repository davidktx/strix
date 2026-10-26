"""Model-name resolution for local cost estimates."""

from __future__ import annotations

from strix.config.model_catalog import resolve_pricing_model


def resolve_litellm_model(model: str) -> str | None:
    """Return a catalog key that can be priced for ``model`` (legacy name)."""
    return resolve_pricing_model(model)
