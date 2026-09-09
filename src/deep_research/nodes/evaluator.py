"""Evaluator node — assesses if collected evidence is sufficient."""

from __future__ import annotations

import json
import logging
from typing import Any

from deep_research.models import EvaluatorDecision
from deep_research.prompts import EVALUATOR_SYSTEM_V2
from deep_research.state import ResearchState
from deep_research.token_usage import accumulate_usage

logger = logging.getLogger(__name__)


async def evaluator_node(state: ResearchState, *, model: Any) -> dict:
    """Evaluate evidence sufficiency with per-sub-question verdicts."""
    plan = state.get("research_plan", [])
    evidence = state.get("evidence", [])
    prior_evidence = state.get("prior_evidence", [])
    iteration = state.get("iteration_count", 0)
    budget = state.get("budget")
    max_iter = budget.max_iterations if budget else 5
    total_evidence_count = len(evidence) + len(prior_evidence)

    # Build per-sub-question evidence bundles with full text + relevance scores
    sq_details = []
    for sq in plan:
        sq_evidence = [e for e in evidence if e.evidence_id in sq.evidence_ids]
        evidence_text = "\n".join(
            f"  - [{e.source_id}] (relevance={e.confidence:.2f}) {e.snippet}"
            for e in sq_evidence
        )
        sq_details.append(
            f"Sub-question [{sq.subquestion_id}]: {sq.question}\n"
            f"  Status: {sq.status}\n"
            f"  Evidence ({len(sq_evidence)} items):\n{evidence_text or '  (none)'}"
        )

    prompt = EVALUATOR_SYSTEM_V2.format(
        sub_question_details="\n\n".join(sq_details),
        evidence_count=total_evidence_count,
        iteration=iteration,
        max_iterations=max_iter,
    )

    response = await model.ainvoke([
        {"role": "system", "content": prompt},
        {"role": "user", "content": "Evaluate the evidence and provide your assessment."},
    ])

    try:
        result = json.loads(response.content)
    except (json.JSONDecodeError, AttributeError):
        return {
            "evaluator_decision": EvaluatorDecision(
                sufficiency_score=0.5, missing_facets=[], recommended_actions=[],
                decision="stop", reason="Failed to parse evaluator response — stopping",
            ),
            "sufficiency_score": 0.5,
            "_token_usage": accumulate_usage(state, response),
        }

    budget_exhausted = iteration >= max_iter
    decision = result.get("decision", "stop")
    if budget_exhausted and decision == "continue":
        decision = "stop"

    # Stop if no new evidence was gathered this iteration
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

    # Compute source diversity (backup if LLM doesn't return it)
    single_source_sqs = []
    for sq in plan:
        sq_evidence = [e for e in evidence if e.evidence_id in sq.evidence_ids]
        unique_tools = set(e.tool_that_produced_it for e in sq_evidence)
        if len(sq_evidence) > 0 and len(unique_tools) <= 1:
            tool_name = next(iter(unique_tools)) if unique_tools else "unknown"
            single_source_sqs.append(f"{sq.subquestion_id} relies only on {tool_name}")
            logger.info(f"Single source: {sq.subquestion_id} uses only {unique_tools}")

    if single_source_sqs:
        logger.info(
            "Source diversity warning: %d sub-question(s) rely on a single tool",
            len(single_source_sqs),
        )

    # Log LLM-provided source diversity info if present
    llm_diversity_score = result.get("source_diversity_score")
    llm_single_source = result.get("single_source_questions", [])
    if llm_diversity_score is not None:
        logger.info("LLM source_diversity_score: %.2f", llm_diversity_score)
    if llm_single_source:
        logger.info("LLM single_source_questions: %s", llm_single_source)

    # Apply per-sub-question verdicts from the LLM
    verdicts = {v["id"]: v for v in result.get("sub_question_verdicts", [])}
    for sq in plan:
        verdict = verdicts.get(sq.subquestion_id, {})
        new_status = verdict.get("verdict")
        if new_status in ("answered", "partially_answered", "unanswered"):
            if new_status == "unanswered":
                sq.status = "pending"  # Map back to valid status
            else:
                sq.status = new_status

    return {
        "evaluator_decision": evaluator_decision,
        "sufficiency_score": evaluator_decision.sufficiency_score,
        "missing_facets": evaluator_decision.missing_facets,
        "research_plan": plan,
        "_token_usage": accumulate_usage(state, response),
    }
