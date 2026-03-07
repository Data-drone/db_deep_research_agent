"""Clarifier node — detects ambiguity in user queries."""

from __future__ import annotations

import json
import logging
from typing import Any

from deep_research.prompts import CLARIFIER_SYSTEM
from deep_research.state import ResearchState

logger = logging.getLogger(__name__)


async def clarifier_node(state: ResearchState, *, model: Any) -> dict:
    """Analyze query for ambiguity. Returns state update."""
    response = await model.ainvoke([
        {"role": "system", "content": CLARIFIER_SYSTEM},
        {"role": "user", "content": state["user_query"]},
    ])

    try:
        result = json.loads(response.content)
    except (json.JSONDecodeError, AttributeError):
        return {
            "clarification_needed": False,
            "clarified_query": state["user_query"],
        }

    return {
        "clarification_needed": result.get("clarification_needed", False),
        "clarified_query": result.get("clarified_query", state["user_query"]),
    }
