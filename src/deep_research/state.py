"""LangGraph state schema for the research agent."""

from __future__ import annotations

from datetime import datetime
from typing import Any, Literal, TypedDict

from deep_research.models import (
    Budget,
    Citation,
    CompressedFindings,
    Evidence,
    EvaluatorDecision,
    SubQuestion,
    ToolCall,
    VerificationResult,
)


class ResearchState(TypedDict, total=False):
    """Typed state for the research agent graph."""

    # Input
    user_query: str
    selected_tools: list[str]
    output_mode: Literal["chat", "report"]

    # Clarification
    clarified_query: str | None

    # Planning
    research_plan: list[SubQuestion]
    tool_assignments: dict[str, list[str]]
    budget: Budget
    perspectives: list[str]

    # Research
    evidence: list[Evidence]
    tool_call_log: list[ToolCall]
    iteration_count: int
    tool_calls_used: int
    started_at: datetime | None

    # Evaluation
    sufficiency_score: float
    missing_facets: list[str]
    stop_reason: str | None
    evaluator_decision: EvaluatorDecision | None

    # Output
    compressed_findings: CompressedFindings | None
    final_output: str
    citations: list[Citation]
    verification_result: VerificationResult | None
    verification_attempts: int

    # Tracing
    trace_id: str
    job_id: str

    # Job control
    cancelled: bool

    # Runtime-only (injected by _run_graph for token streaming, not persisted)
    _job_manager: Any
    _job_id: str


def create_initial_state(
    user_query: str,
    selected_tools: list[str],
    output_mode: Literal["chat", "report"] = "chat",
    job_id: str = "",
    trace_id: str = "",
    budget: Budget | None = None,
) -> ResearchState:
    """Create a fresh state for a new research query."""
    return ResearchState(
        user_query=user_query,
        selected_tools=selected_tools,
        output_mode=output_mode,
        clarified_query=None,
        research_plan=[],
        tool_assignments={},
        budget=budget or Budget(),
        perspectives=[],
        evidence=[],
        tool_call_log=[],
        iteration_count=0,
        tool_calls_used=0,
        started_at=None,
        sufficiency_score=0.0,
        missing_facets=[],
        stop_reason=None,
        evaluator_decision=None,
        compressed_findings=None,
        final_output="",
        citations=[],
        verification_result=None,
        verification_attempts=0,
        trace_id=trace_id,
        job_id=job_id,
        cancelled=False,
    )
