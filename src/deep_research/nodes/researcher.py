"""Researcher node — executes tool calls to gather evidence."""

from __future__ import annotations

import logging
import time
import uuid
from datetime import datetime, timezone
from typing import Any

from deep_research.models import Evidence, ToolCall
from deep_research.state import ResearchState

logger = logging.getLogger(__name__)


async def researcher_node(
    state: ResearchState, *, model: Any, mcp_manager: Any
) -> dict:
    """Execute tool calls from the research plan, collecting evidence."""
    plan = state.get("research_plan", [])
    iteration = state.get("iteration_count", 0) + 1
    existing_evidence = list(state.get("evidence", []))
    existing_tool_calls = list(state.get("tool_call_log", []))
    tool_calls_used = state.get("tool_calls_used", 0)
    budget = state.get("budget")

    new_evidence: list[Evidence] = []
    new_tool_calls: list[ToolCall] = []

    for sq in plan:
        if sq.status == "answered":
            continue

        for tool_name in sq.assigned_tools:
            # Budget check
            if budget and tool_calls_used >= budget.max_tool_calls:
                logger.warning("Tool call budget exhausted")
                break

            tc_id = f"tc-{uuid.uuid4().hex[:8]}"
            start_time = time.monotonic()
            started_at = datetime.now(timezone.utc)

            try:
                result = await mcp_manager.call_tool(
                    tool_name, "query", {"query": sq.question}
                )
                elapsed_ms = int((time.monotonic() - start_time) * 1000)

                tc = ToolCall(
                    tool_call_id=tc_id,
                    tool_name=tool_name,
                    server_name=tool_name,
                    input_data={"query": sq.question},
                    output_data=result,
                    latency_ms=elapsed_ms,
                    status="success",
                    iteration=iteration,
                    started_at=started_at,
                    completed_at=datetime.now(timezone.utc),
                )
                new_tool_calls.append(tc)
                tool_calls_used += 1

                # Create evidence from result
                ev = Evidence(
                    evidence_id=f"ev-{uuid.uuid4().hex[:8]}",
                    source_id=tool_name,
                    source_type="mcp_tool_call",
                    title=f"Result from {tool_name}",
                    uri=result.get("table") or result.get("document"),
                    snippet=result.get("result", str(result)),
                    confidence=0.8,
                    freshness=started_at.isoformat(),
                    tool_that_produced_it=tool_name,
                    tool_call_id=tc_id,
                    iteration=iteration,
                    retrieved_at=started_at,
                )
                new_evidence.append(ev)
                sq.evidence_ids.append(ev.evidence_id)
                sq.status = "in_progress"

            except Exception as e:
                elapsed_ms = int((time.monotonic() - start_time) * 1000)
                tc = ToolCall(
                    tool_call_id=tc_id,
                    tool_name=tool_name,
                    server_name=tool_name,
                    input_data={"query": sq.question},
                    output_data={},
                    latency_ms=elapsed_ms,
                    status="error",
                    iteration=iteration,
                    started_at=started_at,
                    completed_at=datetime.now(timezone.utc),
                    error_type=type(e).__name__,
                    error_message=str(e),
                )
                new_tool_calls.append(tc)
                tool_calls_used += 1
                logger.warning(f"Tool call failed: {tool_name} - {e}")

    return {
        "evidence": existing_evidence + new_evidence,
        "tool_call_log": existing_tool_calls + new_tool_calls,
        "iteration_count": iteration,
        "tool_calls_used": tool_calls_used,
        "research_plan": plan,
    }
