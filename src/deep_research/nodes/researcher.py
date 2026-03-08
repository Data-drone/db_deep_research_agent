"""Researcher node — executes tool calls to gather evidence."""

from __future__ import annotations

import asyncio
import logging
import time
from datetime import datetime, timezone
from typing import Any

from deep_research.evidence_factory import build_evidence_from_tool_result, build_tool_call_record
from deep_research.models import Evidence, ToolCall
from deep_research.state import ResearchState

logger = logging.getLogger(__name__)


async def researcher_node(
    state: ResearchState, *, model: Any, mcp_manager: Any
) -> dict:
    """Execute tool calls from the research plan, collecting evidence.

    Sub-questions are executed in parallel via asyncio.gather.
    Tools *within* a single sub-question still run sequentially
    (since they may depend on each other).
    """
    plan = state.get("research_plan", [])
    iteration = state.get("iteration_count", 0) + 1
    existing_evidence = list(state.get("evidence", []))
    existing_tool_calls = list(state.get("tool_call_log", []))
    tool_calls_used = state.get("tool_calls_used", 0)
    budget = state.get("budget")

    # Identify pending sub-questions
    pending = [sq for sq in plan if sq.status != "answered"]
    if not pending:
        return {
            "evidence": existing_evidence,
            "tool_call_log": existing_tool_calls,
            "iteration_count": iteration,
            "tool_calls_used": tool_calls_used,
            "research_plan": plan,
        }

    # Remaining budget (used for per-sub-question caps)
    remaining_budget = (budget.max_tool_calls - tool_calls_used) if budget else 999
    # Distribute budget slots across pending sub-questions
    per_sq_budget = max(remaining_budget // len(pending), 1) if pending else remaining_budget

    async def _research_one(sq):
        """Execute all tools for one sub-question sequentially."""
        new_evidence: list[Evidence] = []
        new_tool_calls: list[ToolCall] = []
        local_calls = 0

        for tool_name in sq.assigned_tools:
            # Per-sub-question budget cap
            if local_calls >= per_sq_budget:
                logger.warning("Per-sub-question budget cap reached")
                break

            # Use adapted query if available (Gap #2 integration)
            query = sq.adapted_queries.get(tool_name, sq.question)

            start_time = time.monotonic()
            started_at = datetime.now(timezone.utc)
            input_data = {"query": query}

            try:
                result = await mcp_manager.call_tool(
                    tool_name, "query", input_data
                )
                elapsed_ms = int((time.monotonic() - start_time) * 1000)

                tc = build_tool_call_record(
                    tool_name=tool_name,
                    server_name=tool_name,
                    input_data=input_data,
                    output_data=result,
                    latency_ms=elapsed_ms,
                    iteration=iteration,
                    started_at=started_at,
                )
                new_tool_calls.append(tc)
                local_calls += 1

                # Create evidence from result
                evidences = build_evidence_from_tool_result(
                    tool_name=tool_name,
                    tool_call_id=tc.tool_call_id,
                    result=result,
                    iteration=iteration,
                    started_at=started_at,
                )
                new_evidence.extend(evidences)
                for ev in evidences:
                    sq.evidence_ids.append(ev.evidence_id)
                sq.status = "in_progress"

            except Exception as e:
                elapsed_ms = int((time.monotonic() - start_time) * 1000)
                tc = build_tool_call_record(
                    tool_name=tool_name,
                    server_name=tool_name,
                    input_data=input_data,
                    output_data={},
                    latency_ms=elapsed_ms,
                    iteration=iteration,
                    started_at=started_at,
                    status="error",
                    error_type=type(e).__name__,
                    error_message=str(e),
                )
                new_tool_calls.append(tc)
                local_calls += 1
                logger.warning(f"Tool call failed: {tool_name} - {e}")

        return new_evidence, new_tool_calls, local_calls

    # Execute all pending sub-questions in parallel
    results = await asyncio.gather(
        *[_research_one(sq) for sq in pending],
        return_exceptions=True,
    )

    # Merge results
    all_evidence = list(existing_evidence)
    all_tool_calls = list(existing_tool_calls)
    total_calls = tool_calls_used

    for i, result in enumerate(results):
        if isinstance(result, Exception):
            logger.error(
                f"Sub-question research failed: {pending[i].question[:60]} - {result}"
            )
            continue
        evidence, tool_calls, calls = result
        all_evidence.extend(evidence)
        all_tool_calls.extend(tool_calls)
        total_calls += calls

    return {
        "evidence": all_evidence,
        "tool_call_log": all_tool_calls,
        "iteration_count": iteration,
        "tool_calls_used": total_calls,
        "research_plan": plan,
    }
