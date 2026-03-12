"""Clarifier node — refines and focuses user queries for research."""

from __future__ import annotations

import json
import logging
from typing import Any

from deep_research.prompts import CLARIFIER_SYSTEM
from deep_research.state import ResearchState

logger = logging.getLogger(__name__)


async def clarifier_node(state: ResearchState, *, model: Any) -> dict:
    """Refine and focus the user query. May request clarification for ambiguous queries.

    When called with clarified_query already set (pre-graph clarification resolved it),
    short-circuits and returns the existing value.

    Returns either:
    - {"clarified_query": "..."} for clear queries or pre-resolved
    - {"clarified_query": best_guess, "needs_clarification": True,
       "clarification_question": "...", "clarification_options": [...]}
    """
    # Short-circuit if pre-graph clarification already resolved this
    if state.get("clarified_query"):
        return {"clarified_query": state["clarified_query"]}

    messages: list[dict[str, str]] = [{"role": "system", "content": CLARIFIER_SYSTEM}]

    _MAX_HISTORY_TURNS = 10
    for msg in state.get("conversation_history", [])[-_MAX_HISTORY_TURNS:]:
        role = msg.get("role")
        content = msg.get("content")
        if role in ("user", "assistant") and isinstance(content, str):
            messages.append({"role": role, "content": content})

    messages.append({"role": "user", "content": state["user_query"]})

    response = await model.ainvoke(messages)

    content = getattr(response, "content", "")
    if not isinstance(content, str):
        content = str(content)

    try:
        result = json.loads(content)
    except (json.JSONDecodeError, AttributeError):
        return {"clarified_query": state["user_query"]}

    if result.get("needs_clarification"):
        options = result.get("options")
        if not isinstance(options, list):
            options = []
        options = [str(o) for o in options[:3]]

        return {
            "clarified_query": result.get("best_guess", state["user_query"]),
            "needs_clarification": True,
            "clarification_question": str(result.get("question", "Could you clarify your question?")),
            "clarification_options": options,
        }

    return {"clarified_query": result.get("clarified_query", state["user_query"])}
