"""Authorizer node — validates tool access based on risk tiers and error rates."""

from __future__ import annotations

import logging
from typing import Any

from deep_research.state import ResearchState

logger = logging.getLogger(__name__)

# Risk tier mapping: tool name substring -> tier
_RISK_TIERS: list[tuple[str, str]] = [
    ("genie", "safe"),
    ("vector_search", "safe"),
    ("vs", "safe"),
    ("knowledge", "restricted"),
    ("ka", "restricted"),
]


def _get_risk_tier(tool_name: str) -> str:
    """Classify a tool into a risk tier based on its name."""
    lower = tool_name.lower()
    for keyword, tier in _RISK_TIERS:
        if keyword in lower:
            return tier
    return "safe"  # unknown tools default to safe


def _tool_error_rate(tool_name: str, tool_call_log: list) -> float:
    """Calculate the error rate for a specific tool from the call log."""
    calls = [tc for tc in tool_call_log if tc.tool_name == tool_name]
    if not calls:
        return 0.0
    errors = sum(1 for tc in calls if tc.status == "error")
    return errors / len(calls)


async def authorizer_node(state: ResearchState, *, model: Any) -> dict:
    """Validate planned tool calls against risk tiers and error rates.

    - safe: always approved
    - restricted: approved with logging
    - privileged: rejected if error rate > 50%
    """
    plan = state.get("research_plan", [])
    tool_call_log = state.get("tool_call_log", [])

    tool_assignments: dict[str, list[str]] = {}
    for sq in plan:
        approved_tools = []
        for tool in sq.assigned_tools:
            tier = _get_risk_tier(tool)
            if tier == "safe":
                approved_tools.append(tool)
            elif tier == "restricted":
                logger.info("Restricted tool access: %s", tool)
                approved_tools.append(tool)
            elif tier == "privileged":
                if _tool_error_rate(tool, tool_call_log) > 0.5:
                    logger.warning("Downgrading %s — high error rate", tool)
                    continue
                approved_tools.append(tool)
            else:
                approved_tools.append(tool)

            if tool not in tool_assignments:
                tool_assignments[tool] = []
            tool_assignments[tool].append(sq.subquestion_id)

        sq.assigned_tools = approved_tools

    return {"research_plan": plan, "tool_assignments": tool_assignments}
