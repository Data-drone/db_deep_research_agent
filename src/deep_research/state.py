"""LangGraph state schema for the research agent."""

from __future__ import annotations

from typing import Literal, TypedDict

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
    clarification_needed: bool

    # Planning
    research_plan: list[SubQuestion]
    tool_assignments: dict[str, list[str]]
    budget: Budget

    # Research
    evidence: list[Evidence]
    tool_call_log: list[ToolCall]
    iteration_count: int

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

    # Tracing
    trace_id: str
    job_id: str

    # Job control
    cancelled: bool


def create_initial_state(
    user_query: str,
    selected_tools: list[str],
    output_mode: Literal["chat", "report"] = "chat",
    job_id: str = "",
    trace_id: str = "",
) -> ResearchState:
    """Create a fresh state for a new research query."""
    return ResearchState(
        user_query=user_query,
        selected_tools=selected_tools,
        output_mode=output_mode,
        clarified_query=None,
        clarification_needed=False,
        research_plan=[],
        tool_assignments={},
        budget=Budget(),
        evidence=[],
        tool_call_log=[],
        iteration_count=0,
        sufficiency_score=0.0,
        missing_facets=[],
        stop_reason=None,
        evaluator_decision=None,
        compressed_findings=None,
        final_output="",
        citations=[],
        verification_result=None,
        trace_id=trace_id,
        job_id=job_id,
        cancelled=False,
    )
