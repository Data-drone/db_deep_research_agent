"""Synthesizer node — produces final answer or report from findings."""

from __future__ import annotations

import logging
from typing import Any

from deep_research.citations import (
    build_evidence_index,
    extract_citations,
    render_evidence_block,
)
from deep_research.prompts import SYNTHESIZER_CHAT_SYSTEM, SYNTHESIZER_REPORT_SYSTEM
from deep_research.state import ResearchState
from deep_research.token_usage import (
    accumulate_usage,
    add_usage,
    empty_usage,
    extract_stream_usage,
)

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

    # Enumerate evidence so every citation the model writes can be resolved back
    # to a real item. Free-text citations were unverifiable by construction.
    evidence_index = build_evidence_index(evidence)
    if evidence_index:
        context_parts.append("Evidence — cite by marker, e.g. [E1]:")
        context_parts.extend(render_evidence_block(evidence_index))

    messages = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": "\n".join(context_parts)},
    ]

    # Try streaming for token-level output
    if job_manager and job_id and hasattr(model, "astream"):
        chunks = []
        stream_usage = empty_usage()
        # Providers only attach usage to streamed chunks when asked. astream
        # takes **kwargs, so an unsupported flag would reach the provider rather
        # than raise — gate on the model actually declaring the field. Both
        # implementations we target take it as a named parameter of ``_astream``
        # and consume it there, so it never reaches the request body:
        # ``ChatDatabricks`` (which already defaults it True) and ``ChatOpenAI``
        # (which defaults it False, and is the reason to keep passing it).
        if hasattr(model, "stream_usage"):
            stream = model.astream(messages, stream_usage=True)
        else:
            stream = model.astream(messages)
        async for chunk in stream:
            token = chunk.content if hasattr(chunk, "content") else str(chunk)
            if token:
                chunks.append(token)
                job_manager.push_event(job_id, {"type": "token", "content": token})
            # Chunk usage is read via the additive ``usage_metadata`` field only —
            # see extract_stream_usage for why response_metadata is unsafe to sum.
            stream_usage = add_usage(stream_usage, extract_stream_usage(chunk))
        final_output = "".join(chunks)
        token_usage = add_usage(state.get("_token_usage"), stream_usage)
    else:
        response = await model.ainvoke(messages)
        final_output = response.content
        token_usage = accumulate_usage(state, response)

    final_output, citations, unverified = extract_citations(final_output, evidence_index)
    if unverified:
        logger.warning(
            "Dropped %d fabricated citation(s) from the synthesized output", len(unverified)
        )

    return {
        "final_output": final_output,
        "citations": citations,
        "unverified_citations": sorted(set(unverified)),
        "_token_usage": token_usage,
    }
