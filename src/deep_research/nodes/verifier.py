"""Verifier node — checks citations and claim support."""

from __future__ import annotations

import json
import logging
from typing import Any

from deep_research.citations import UNVERIFIED_LABEL, build_evidence_index
from deep_research.models import VerificationResult
from deep_research.prompts import VERIFIER_SYSTEM
from deep_research.state import ResearchState
from deep_research.token_usage import accumulate_usage

logger = logging.getLogger(__name__)


async def verifier_node(state: ResearchState, *, model: Any) -> dict:
    """Verify that all claims in the final output are supported by evidence."""
    final_output = state.get("final_output", "")
    evidence = state.get("evidence", [])
    verification_attempts = state.get("verification_attempts", 0) + 1

    # The draft cites evidence as [E1]..[En] and marks anything it invented as
    # [unverified]. The verifier has to be shown the same numbering, or it cannot
    # match a marker to an item and reports every cited claim as unsupported.
    evidence_index = build_evidence_index(evidence)
    evidence_text = "\n".join(
        f"- [{marker}] (relevance={e.confidence:.2f}) {e.snippet}"
        for marker, e in evidence_index.items()
    )

    response = await model.ainvoke([
        {"role": "system", "content": VERIFIER_SYSTEM},
        {"role": "user", "content": (
            f"Draft response:\n{final_output}\n\n"
            f"Available evidence:\n{evidence_text}\n\n"
            f"The draft cites evidence by marker, e.g. [E1]. A marker written as "
            f"{UNVERIFIED_LABEL} is one the drafting step already found to be "
            f"fabricated and stripped; treat the claim it sits on as unsupported."
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
            "_token_usage": accumulate_usage(state, response),
        }

    return {
        "verification_result": VerificationResult(
            all_claims_supported=result.get("all_claims_supported", True),
            unsupported_claims=result.get("unsupported_claims", []),
            weakened_claims=result.get("weakened_claims", []),
            contradictions_noted=result.get("contradictions_noted", []),
        ),
        "verification_attempts": verification_attempts,
        "_token_usage": accumulate_usage(state, response),
    }
