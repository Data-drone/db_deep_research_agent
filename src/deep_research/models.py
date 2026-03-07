"""Core data models for the deep research agent."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Literal


@dataclass
class Evidence:
    evidence_id: str
    source_id: str
    source_type: str
    title: str
    uri: str | None
    snippet: str
    confidence: float
    freshness: str
    tool_that_produced_it: str
    iteration: int
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass
class SubQuestion:
    question: str
    assigned_tools: list[str]
    answered: bool = False


@dataclass
class Budget:
    max_iterations: int = 5
    max_tool_calls: int = 20
    time_cap_seconds: int = 120


@dataclass
class ToolCall:
    tool_name: str
    server_name: str
    input_data: dict[str, Any]
    output_data: dict[str, Any]
    latency_ms: int
    success: bool
    error: str | None = None


@dataclass
class EvaluatorDecision:
    sufficiency_score: float
    missing_facets: list[str]
    recommended_actions: list[str]
    decision: Literal["continue", "stop"]
    reason: str


@dataclass
class Citation:
    claim: str
    evidence_ids: list[str]
    confidence: float


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
