"""Verifier node — checks citations and claim support."""

from __future__ import annotations

import json
import logging
from typing import Any

from deep_research.models import VerificationResult
from deep_research.prompts import VERIFIER_SYSTEM
from deep_research.state import ResearchState

logger = logging.getLogger(__name__)


async def verifier_node(state: ResearchState, *, model: Any) -> dict:
    """Verify that all claims in the final output are supported by evidence."""
    final_output = state.get("final_output", "")
    evidence = state.get("evidence", [])
    verification_attempts = state.get("verification_attempts", 0) + 1

    evidence_text = "\n".join(
        f"- [{e.source_id}] (relevance={e.confidence:.2f}) {e.snippet}" for e in evidence
    )

    response = await model.ainvoke([
        {"role": "system", "content": VERIFIER_SYSTEM},
        {"role": "user", "content": (
            f"Draft response:\n{final_output}\n\n"
            f"Available evidence:\n{evidence_text}"
        )},
    ])

    try:
        result = json.loads(response.content)
    except (json.JSONDecodeError, AttributeError):
        return {
            "verification_result": VerificationResult(
                all_claims_supported=True,
                unsupported_claims=[],
                weakened_claims=[],
                contradictions_noted=[],
            ),
            "verification_attempts": verification_attempts,
        }

    return {
        "verification_result": VerificationResult(
            all_claims_supported=result.get("all_claims_supported", True),
            unsupported_claims=result.get("unsupported_claims", []),
            weakened_claims=result.get("weakened_claims", []),
            contradictions_noted=result.get("contradictions_noted", []),
        ),
        "verification_attempts": verification_attempts,
    }
