"""Core data models for the deep research agent."""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime
from typing import Any, Literal


@dataclass
class Evidence:
    evidence_id: str
    source_id: str
    source_type: str
    title: str
    uri: str | None
    snippet: str
    confidence: float  # 0.0 to 1.0
    freshness: str
    tool_that_produced_it: str
    tool_call_id: str
    iteration: int
    retrieved_at: datetime | None = None
    content_hash: str | None = None
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass
class SubQuestion:
    subquestion_id: str
    question: str
    assigned_tools: list[str]
    status: Literal["pending", "in_progress", "answered", "partially_answered", "abandoned"] = "pending"
    answer_summary: str | None = None
    evidence_ids: list[str] = field(default_factory=list)
    iteration_created: int = 0
    adapted_queries: dict[str, str] = field(default_factory=dict)


@dataclass
class Budget:
    max_iterations: int = 5
    max_tool_calls: int = 20
    time_cap_seconds: int = 120


@dataclass
class ToolCall:
    tool_call_id: str
    tool_name: str
    server_name: str
    input_data: dict[str, Any]
    output_data: dict[str, Any]
    latency_ms: int
    status: Literal["pending", "success", "error", "timeout", "cancelled"] = "success"
    iteration: int = 0
    started_at: datetime | None = None
    completed_at: datetime | None = None
    error_type: str | None = None
    error_message: str | None = None


@dataclass
class EvaluatorDecision:
    sufficiency_score: float  # 0.0 to 1.0
    missing_facets: list[str]
    recommended_actions: list[str]
    decision: Literal["continue", "stop"]
    reason: str
    budget_exhausted: bool = False


@dataclass
class Citation:
    claim_id: str
    claim: str
    evidence_ids: list[str]
    confidence: float  # 0.0 to 1.0


@dataclass
class CompressedFindings:
    key_findings: list[str]
    open_questions: list[str]
    uncertainties: list[str]
    contradictions: list[str]


@dataclass
class VerificationResult:
    all_claims_supported: bool
    unsupported_claims: list[str]
    weakened_claims: list[str]
    contradictions_noted: list[str]
