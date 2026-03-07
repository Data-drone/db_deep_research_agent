"""Synthesizer node — produces final answer or report from findings."""

from __future__ import annotations

import logging
from typing import Any

from deep_research.prompts import SYNTHESIZER_CHAT_SYSTEM, SYNTHESIZER_REPORT_SYSTEM
from deep_research.state import ResearchState

logger = logging.getLogger(__name__)


async def synthesizer_node(state: ResearchState, *, model: Any) -> dict:
    """Generate final output from compressed findings."""
    findings = state.get("compressed_findings")
    evidence = state.get("evidence", [])
    output_mode = state.get("output_mode", "chat")
    query = state.get("clarified_query") or state["user_query"]

    system_prompt = (
        SYNTHESIZER_REPORT_SYSTEM if output_mode == "report"
        else SYNTHESIZER_CHAT_SYSTEM
    )

    # Build context for synthesis
    context_parts = [f"User query: {query}"]

    if findings:
        context_parts.append(f"Key findings: {', '.join(findings.key_findings)}")
        if findings.open_questions:
            context_parts.append(f"Open questions: {', '.join(findings.open_questions)}")
        if findings.uncertainties:
            context_parts.append(f"Uncertainties: {', '.join(findings.uncertainties)}")
        if findings.contradictions:
            context_parts.append(f"Contradictions: {', '.join(findings.contradictions)}")

    if evidence:
        context_parts.append("Evidence:")
        for e in evidence:
            context_parts.append(f"- [{e.source_id}] {e.snippet}")

    response = await model.ainvoke([
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": "\n".join(context_parts)},
    ])

    return {"final_output": response.content}
