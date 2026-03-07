"""Planner node — breaks query into sub-questions and assigns tools."""

from __future__ import annotations

import json
import logging
import uuid
from typing import Any

from deep_research.models import SubQuestion
from deep_research.prompts import PLANNER_SYSTEM
from deep_research.state import ResearchState

logger = logging.getLogger(__name__)


async def planner_node(state: ResearchState, *, model: Any) -> dict:
    """Create research plan with sub-questions and tool assignments."""
    query = state.get("clarified_query") or state["user_query"]
    tools = state.get("selected_tools", [])
    iteration = state.get("iteration_count", 0)

    prompt = PLANNER_SYSTEM.format(tools=", ".join(tools))

    response = await model.ainvoke([
        {"role": "system", "content": prompt},
        {"role": "user", "content": query},
    ])

    try:
        result = json.loads(response.content)
    except (json.JSONDecodeError, AttributeError):
        # Fallback: one sub-question per tool
        sub_questions = [
            SubQuestion(
                subquestion_id=f"sq-{uuid.uuid4().hex[:8]}",
                question=query,
                assigned_tools=tools,
                iteration_created=iteration,
            )
        ]
        return {"research_plan": sub_questions}

    sub_questions = []
    for sq in result.get("sub_questions", []):
        sub_questions.append(
            SubQuestion(
                subquestion_id=f"sq-{uuid.uuid4().hex[:8]}",
                question=sq.get("question", query),
                assigned_tools=sq.get("assigned_tools", tools),
                iteration_created=iteration,
            )
        )

    return {"research_plan": sub_questions}
