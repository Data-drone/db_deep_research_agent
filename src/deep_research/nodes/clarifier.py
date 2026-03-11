"""Clarifier node — refines and focuses user queries for research."""

from __future__ import annotations

import json
import logging
from typing import Any

from deep_research.prompts import CLARIFIER_SYSTEM
from deep_research.state import ResearchState

logger = logging.getLogger(__name__)


async def clarifier_node(state: ResearchState, *, model: Any) -> dict:
    """Refine and focus the user query. Always produces a clarified version.

    When conversation history is present (follow-up query), includes prior
    turns so the model can resolve references like "Tell me more about X".
    """
    messages: list[dict[str, str]] = [{"role": "system", "content": CLARIFIER_SYSTEM}]

    # Include recent prior turns for context (filtered to user/assistant only, capped)
    _MAX_HISTORY_TURNS = 10
    for msg in state.get("conversation_history", [])[-_MAX_HISTORY_TURNS:]:
        role = msg.get("role")
        content = msg.get("content")
        if role in ("user", "assistant") and isinstance(content, str):
            messages.append({"role": role, "content": content})

    messages.append({"role": "user", "content": state["user_query"]})

    response = await model.ainvoke(messages)

    try:
        result = json.loads(response.content)
    except (json.JSONDecodeError, AttributeError):
        return {"clarified_query": state["user_query"]}

    return {"clarified_query": result.get("clarified_query", state["user_query"])}
