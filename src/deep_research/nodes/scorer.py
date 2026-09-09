"""Scorer node — rates evidence relevance to sub-questions using the critic model."""

from __future__ import annotations

import json
import logging
from typing import Any

from deep_research.prompts import SCORER_SYSTEM
from deep_research.state import ResearchState
from deep_research.token_usage import add_usage, charge, empty_usage

logger = logging.getLogger(__name__)


async def _score_relevance(
    model: Any, question: str, snippet: str, usage: dict[str, int]
) -> float:
    """Score a single evidence item's relevance to a question."""
    prompt = SCORER_SYSTEM.format(question=question)
    response = await model.ainvoke([
        {"role": "system", "content": prompt},
        {"role": "user", "content": snippet},
    ])
    charge(usage, response)
    try:
        result = json.loads(response.content)
        return max(0.0, min(1.0, float(result.get("score", 0.5))))
    except (json.JSONDecodeError, ValueError, AttributeError):
        return 0.5  # Default to moderate relevance on parse failure


async def scorer_node(state: ResearchState, *, model: Any) -> dict:
    """Rate each evidence item's relevance to its sub-question using the critic model.

    Replaces hardcoded confidence=0.8 with LLM-assessed relevance scores.
    Filters out evidence below 0.3 relevance threshold.
    """
    plan = state.get("research_plan", [])
    evidence = list(state.get("evidence", []))
    usage = empty_usage()

    for sq in plan:
        sq_evidence = [e for e in evidence if e.evidence_id in sq.evidence_ids]
        for ev in sq_evidence:
            score = await _score_relevance(model, sq.question, ev.snippet, usage)
            ev.confidence = score

    # Filter low-relevance evidence
    filtered = [e for e in evidence if e.confidence >= 0.3]
    logger.info(f"Scorer: {len(evidence)} items → {len(filtered)} after filtering (threshold=0.3)")

    return {
        "evidence": filtered,
        "_token_usage": add_usage(state.get("_token_usage"), usage),
    }
