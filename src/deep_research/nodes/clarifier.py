"""Clarifier node — refines and focuses user queries for research."""

from __future__ import annotations

import json
import logging
from typing import Any

from deep_research.prompts import CLARIFIER_SYSTEM
from deep_research.state import ResearchState

logger = logging.getLogger(__name__)


async def clarifier_node(state: ResearchState, *, model: Any) -> dict:
    """Refine and focus the user query. Always produces a clarified version."""
    response = await model.ainvoke([
        {"role": "system", "content": CLARIFIER_SYSTEM},
        {"role": "user", "content": state["user_query"]},
    ])

    try:
        result = json.loads(response.content)
    except (json.JSONDecodeError, AttributeError):
        return {"clarified_query": state["user_query"]}

    return {"clarified_query": result.get("clarified_query", state["user_query"])}
