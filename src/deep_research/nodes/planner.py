"""Planner node — breaks query into sub-questions and assigns tools.

On first invocation, creates a fresh plan. On replanning (iteration > 0),
merges new sub-questions with existing plan to avoid losing previous work.
"""

from __future__ import annotations

import json
import logging
import uuid
from typing import Any

from deep_research.models import SubQuestion
from deep_research.prompts import PLANNER_SYSTEM
from deep_research.state import ResearchState

logger = logging.getLogger(__name__)


def _normalize_question(q: str) -> str:
    """Normalize a question for dedup comparison."""
    return " ".join(q.lower().strip().split())


def _merge_plans(
    existing: list[SubQuestion],
    new_items: list[SubQuestion],
) -> list[SubQuestion]:
    """Merge new sub-questions into existing plan without duplicates."""
    existing_questions = {_normalize_question(sq.question) for sq in existing}
    merged = list(existing)

    for sq in new_items:
        normalized = _normalize_question(sq.question)
        if normalized not in existing_questions:
            merged.append(sq)
            existing_questions.add(normalized)
        else:
            logger.debug(f"Skipping duplicate sub-question: {sq.question[:60]}")

    return merged


async def planner_node(state: ResearchState, *, model: Any) -> dict:
    """Create or extend research plan with sub-questions and tool assignments."""
    query = state.get("clarified_query") or state["user_query"]
    tools = state.get("selected_tools", [])
    iteration = state.get("iteration_count", 0)
    existing_plan = state.get("research_plan", [])
    missing_facets = state.get("missing_facets", [])

    # Build context-aware prompt for replanning
    user_content = query
    if existing_plan and iteration > 0:
        answered = [sq for sq in existing_plan if sq.status == "answered"]
        unanswered = [sq for sq in existing_plan if sq.status != "answered"]
        context_parts = [f"Original query: {query}"]
        if answered:
            context_parts.append(
                f"Already answered: {', '.join(sq.question for sq in answered)}"
            )
        if unanswered:
            context_parts.append(
                f"Still unanswered: {', '.join(sq.question for sq in unanswered)}"
            )
        if missing_facets:
            context_parts.append(f"Gaps identified: {', '.join(missing_facets)}")
        context_parts.append("Generate ONLY new sub-questions to fill the gaps.")
        user_content = "\n".join(context_parts)

    prompt = PLANNER_SYSTEM.format(tools=", ".join(tools))

    response = await model.ainvoke([
        {"role": "system", "content": prompt},
        {"role": "user", "content": user_content},
    ])

    try:
        result = json.loads(response.content)
    except (json.JSONDecodeError, AttributeError):
        if not existing_plan:
            return {
                "research_plan": [
                    SubQuestion(
                        subquestion_id=f"sq-{uuid.uuid4().hex[:8]}",
                        question=query,
                        assigned_tools=tools,
                        iteration_created=iteration,
                    )
                ]
            }
        return {"research_plan": existing_plan}

    new_sub_questions = []
    for sq in result.get("sub_questions", []):
        new_sub_questions.append(
            SubQuestion(
                subquestion_id=f"sq-{uuid.uuid4().hex[:8]}",
                question=sq.get("question", query),
                assigned_tools=sq.get("assigned_tools", tools),
                iteration_created=iteration,
            )
        )

    merged = _merge_plans(existing_plan, new_sub_questions)
    return {"research_plan": merged}
