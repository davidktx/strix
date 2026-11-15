"""OpenRouter helpers for the OpenAI-compatible client route."""

from __future__ import annotations

import uuid
from typing import Any

from strix.config.loader import load_settings


_SESSION_IDS: dict[str, str] = {}


def sticky_session_extra_body(agent_id: str | None) -> dict[str, Any] | None:
    """OpenRouter ``session_id`` for per-agent provider stickiness."""
    if not agent_id or not load_settings().llm.openrouter_sticky_sessions:
        return None
    session_id = _SESSION_IDS.setdefault(agent_id, str(uuid.uuid4()))
    return {"session_id": session_id}
