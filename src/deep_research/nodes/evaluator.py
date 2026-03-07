"""Evaluator node — assesses if collected evidence is sufficient."""

from __future__ import annotations

import json
import logging
from typing import Any

from deep_research.models import EvaluatorDecision
from deep_research.prompts import EVALUATOR_SYSTEM
from deep_research.state import ResearchState

logger = logging.getLogger(__name__)


async def evaluator_node(state: ResearchState, *, model: Any) -> dict:
    """Evaluate evidence sufficiency and decide to continue or stop."""
    plan = state.get("research_plan", [])
    evidence = state.get("evidence", [])
    iteration = state.get("iteration_count", 0)
    budget = state.get("budget")
    max_iter = budget.max_iterations if budget else 5

    sub_questions_str = ", ".join(sq.question for sq in plan)

    prompt = EVALUATOR_SYSTEM.format(
        sub_questions=sub_questions_str,
        evidence_count=len(evidence),
        iteration=iteration,
        max_iterations=max_iter,
    )

    response = await model.ainvoke([
        {"role": "system", "content": prompt},
        {"role": "user", "content": f"Evidence snippets:\n" + "\n".join(
            f"- [{e.source_id}] {e.snippet[:200]}" for e in evidence
        )},
    ])

    try:
        result = json.loads(response.content)
    except (json.JSONDecodeError, AttributeError):
        # Fallback: always stop on parse failure to avoid looping
        return {
            "evaluator_decision": EvaluatorDecision(
                sufficiency_score=0.5,
                missing_facets=[],
                recommended_actions=[],
                decision="stop",
                reason="Failed to parse evaluator response — stopping",
            ),
            "sufficiency_score": 0.5,
        }

    budget_exhausted = iteration >= max_iter
    decision = result.get("decision", "stop")
    if budget_exhausted and decision == "continue":
        decision = "stop"

    # Stop if no new evidence was gathered this iteration (no progress)
    if decision == "continue" and iteration > 0:
        new_evidence = [e for e in evidence if e.iteration == iteration]
        if not new_evidence:
            decision = "stop"
            logger.info("No new evidence gathered — stopping research loop")

    evaluator_decision = EvaluatorDecision(
        sufficiency_score=result.get("sufficiency_score", 0.5),
        missing_facets=result.get("missing_facets", []),
        recommended_actions=result.get("recommended_actions", []),
        decision=decision,
        reason=result.get("reason", ""),
        budget_exhausted=budget_exhausted,
    )

    # Mark sub-questions as answered based on evidence
    for sq in plan:
        if sq.evidence_ids and sq.status != "answered":
            sq.status = "answered"

    return {
        "evaluator_decision": evaluator_decision,
        "sufficiency_score": evaluator_decision.sufficiency_score,
        "missing_facets": evaluator_decision.missing_facets,
        "research_plan": plan,
    }
