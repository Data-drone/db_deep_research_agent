"""Query adapter node — reformulates sub-questions for each assigned tool."""

from __future__ import annotations

import logging
from typing import Any

from deep_research.prompts import QUERY_ADAPTER_SYSTEM
from deep_research.state import ResearchState

logger = logging.getLogger(__name__)

# Tool-type detection and guidance mapping
_TOOL_TYPE_GUIDANCE = [
    (
        "genie",
        "data_query",
        "Rephrase as a natural-language data question suitable for SQL analysis",
    ),
    (
        "vector_search",
        "semantic_search",
        "Extract key terms and rephrase as a semantic search query",
    ),
    (
        "vs",
        "semantic_search",
        "Extract key terms and rephrase as a semantic search query",
    ),
    (
        "knowledge",
        "knowledge_retrieval",
        "Rephrase as a focused knowledge retrieval question",
    ),
    (
        "ka",
        "knowledge_retrieval",
        "Rephrase as a focused knowledge retrieval question",
    ),
]


def _classify_tool(tool_name: str) -> tuple[str, str]:
    """Return (tool_type, guidance) for a tool name."""
    lower = tool_name.lower()
    for keyword, tool_type, guidance in _TOOL_TYPE_GUIDANCE:
        if keyword in lower:
            return tool_type, guidance
    return "default", ""


async def _reformulate(model: Any, question: str, tool_name: str) -> str:
    """Reformulate a question for a specific tool type.

    For unknown/default tool types, returns the original question unchanged
    without calling the model.
    """
    tool_type, guidance = _classify_tool(tool_name)

    if tool_type == "default":
        return question

    prompt = QUERY_ADAPTER_SYSTEM.format(tool_type=tool_type, guidance=guidance)
    response = await model.ainvoke([
        {"role": "system", "content": prompt},
        {"role": "user", "content": question},
    ])
    return response.content.strip()


async def query_adapter_node(state: ResearchState, *, model: Any) -> dict:
    """Reformulate each sub-question for its assigned tools."""
    plan = state.get("research_plan", [])
    for sq in plan:
        if sq.status == "answered":
            continue
        for tool_name in sq.assigned_tools:
            adapted = await _reformulate(model, sq.question, tool_name)
            sq.adapted_queries[tool_name] = adapted
    return {"research_plan": plan}
