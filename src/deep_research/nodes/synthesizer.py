"""Synthesizer node — produces final answer or report from findings."""

from __future__ import annotations

import logging
from typing import Any

from deep_research.prompts import SYNTHESIZER_CHAT_SYSTEM, SYNTHESIZER_REPORT_SYSTEM
from deep_research.state import ResearchState

logger = logging.getLogger(__name__)


async def synthesizer_node(state: ResearchState, *, model: Any) -> dict:
    """Generate final output from compressed findings, streaming tokens when possible."""
    findings = state.get("compressed_findings")
    evidence = state.get("evidence", [])
    output_mode = state.get("output_mode", "chat")
    query = state.get("clarified_query") or state["user_query"]

    # Job context for token streaming (injected by _run_graph)
    job_manager = state.get("_job_manager")
    job_id = state.get("_job_id")

    system_prompt = (
        SYNTHESIZER_REPORT_SYSTEM if output_mode == "report"
        else SYNTHESIZER_CHAT_SYSTEM
    )

    # Build context for synthesis
    context_parts = []

    # Include conversation context for follow-up queries
    conversation_history = state.get("conversation_history", [])
    if conversation_history:
        context_parts.append("Conversation context (prior turns):")
        for msg in conversation_history[-6:]:  # Last 3 exchanges max
            role = msg.get("role", "user")
            content = msg.get("content", "")
            # Truncate long messages to keep prompt manageable
            if len(content) > 500:
                content = content[:500] + "..."
            context_parts.append(f"  {role}: {content}")
        context_parts.append("(Use prior conversation as background context only; prioritize current query and evidence.)")
        context_parts.append("")

    context_parts.append(f"User query: {query}")

    if findings:
        context_parts.append(f"Key findings: {', '.join(findings.key_findings)}")
        if findings.open_questions:
            context_parts.append(f"Open questions: {', '.join(findings.open_questions)}")
        if findings.uncertainties:
            context_parts.append(f"Uncertainties: {', '.join(findings.uncertainties)}")
        if findings.contradictions:
            context_parts.append(f"Contradictions: {', '.join(findings.contradictions)}")

    if evidence:
        context_parts.append("Evidence (cite using [Source: Tool — description] format):")
        for e in evidence:
            tool_label = (e.tool_that_produced_it or "Unknown").replace("]", ")")
            title_label = (e.title or e.source_id).replace("]", ")")
            context_parts.append(f"- [Source: {tool_label} — {title_label}] {e.snippet}")

    messages = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": "\n".join(context_parts)},
    ]

    # Try streaming for token-level output
    if job_manager and job_id and hasattr(model, "astream"):
        chunks = []
        async for chunk in model.astream(messages):
            token = chunk.content if hasattr(chunk, "content") else str(chunk)
            if token:
                chunks.append(token)
                job_manager.push_event(job_id, {"type": "token", "content": token})
        final_output = "".join(chunks)
    else:
        response = await model.ainvoke(messages)
        final_output = response.content

    return {"final_output": final_output}
