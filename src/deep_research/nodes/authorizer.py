"""Authorizer node — validates tool access based on risk tiers and policies."""

from __future__ import annotations

import logging
from typing import Any

from deep_research.state import ResearchState

logger = logging.getLogger(__name__)


async def authorizer_node(state: ResearchState, *, model: Any) -> dict:
    """Validate planned tool calls against policies. Passthrough for now.

    Future: check risk tiers, user permissions, rate limits.
    """
    plan = state.get("research_plan", [])

    # Collect all tools referenced in the plan
    tool_assignments: dict[str, list[str]] = {}
    for sq in plan:
        for tool in sq.assigned_tools:
            if tool not in tool_assignments:
                tool_assignments[tool] = []
            tool_assignments[tool].append(sq.subquestion_id)

    return {"tool_assignments": tool_assignments}
