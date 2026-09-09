"""Compressor node — distills evidence into structured findings."""

from __future__ import annotations

import json
import logging
from typing import Any

from deep_research.models import CompressedFindings
from deep_research.prompts import COMPRESSOR_SYSTEM
from deep_research.state import ResearchState
from deep_research.token_usage import accumulate_usage

logger = logging.getLogger(__name__)


async def compressor_node(state: ResearchState, *, model: Any) -> dict:
    """Compress evidence into key findings, open questions, etc."""
    evidence = state.get("evidence", [])

    evidence_text = "\n".join(
        f"- [{e.source_id}] {e.snippet}" for e in evidence
    )

    response = await model.ainvoke([
        {"role": "system", "content": COMPRESSOR_SYSTEM},
        {"role": "user", "content": f"Evidence to compress:\n{evidence_text}"},
    ])

    try:
        result = json.loads(response.content)
    except (json.JSONDecodeError, AttributeError):
        return {
            "compressed_findings": CompressedFindings(
                key_findings=[e.snippet for e in evidence[:5]],
                open_questions=[],
                uncertainties=[],
                contradictions=[],
            ),
            "_token_usage": accumulate_usage(state, response),
        }

    return {
        "compressed_findings": CompressedFindings(
            key_findings=result.get("key_findings", []),
            open_questions=result.get("open_questions", []),
            uncertainties=result.get("uncertainties", []),
            contradictions=result.get("contradictions", []),
        ),
        "_token_usage": accumulate_usage(state, response),
    }
