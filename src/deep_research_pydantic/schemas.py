"""Structured Pydantic output schemas for research-agent LLM calls."""

from __future__ import annotations

from typing import Any, Literal

from pydantic import BaseModel, Field, field_validator


class ClarifierOutput(BaseModel):
    """Output from the query clarifier."""

    clarified_query: str | None = None
    needs_clarification: bool = False
    question: str | None = None
    options: list[str] = Field(default_factory=list)
    best_guess: str | None = None

    @field_validator(
        "clarified_query",
        "question",
        "best_guess",
        mode="before",
    )
    @classmethod
    def coerce_optional_text(cls, value: Any) -> str | None:
        if value is None or isinstance(value, str):
            return value
        return str(value)

    @field_validator("options", mode="before")
    @classmethod
    def normalize_and_cap_options(cls, value: Any) -> list[str]:
        if not isinstance(value, list):
            return []
        return [str(option) for option in value[:3]]


class PerspectiveOutput(BaseModel):
    """Output from the STORM-style perspective generator."""

    perspectives: list[str] = Field(default_factory=list)

    @field_validator("perspectives", mode="before")
    @classmethod
    def cap_perspectives(cls, value: Any) -> list[Any]:
        if not isinstance(value, list):
            return []
        return value[:3]


class PlannedSubQuestion(BaseModel):
    """One planned research sub-question."""

    question: str | None = None
    assigned_tools: list[str] = Field(default_factory=list)


class PlannerOutput(BaseModel):
    """Output from the research planner."""

    sub_questions: list[PlannedSubQuestion] = Field(default_factory=list)
    estimated_iterations: int | None = None


class ScorerOutput(BaseModel):
    """Output from the evidence relevance scorer."""

    score: float = 0.5

    @field_validator("score", mode="after")
    @classmethod
    def clamp_score(cls, value: float) -> float:
        return max(0.0, min(1.0, float(value)))


class SubQuestionVerdict(BaseModel):
    """Evaluator verdict for one sub-question."""

    id: str
    verdict: Literal[
        "answered",
        "partially_answered",
        "unanswered",
    ] | None = None
    reason: str | None = None


class EvaluatorOutput(BaseModel):
    """Output matching EVALUATOR_SYSTEM_V2."""

    sufficiency_score: float = 0.5
    sub_question_verdicts: list[SubQuestionVerdict] = Field(
        default_factory=list
    )
    missing_facets: list[str] = Field(default_factory=list)
    recommended_actions: list[str] = Field(default_factory=list)
    decision: Literal["continue", "stop"] = "stop"
    reason: str = ""
    source_diversity_score: float | None = None
    single_source_questions: list[str] = Field(default_factory=list)


class CompressorOutput(BaseModel):
    """Output from the evidence compressor."""

    key_findings: list[str] = Field(default_factory=list)
    open_questions: list[str] = Field(default_factory=list)
    uncertainties: list[str] = Field(default_factory=list)
    contradictions: list[str] = Field(default_factory=list)


class VerifierOutput(BaseModel):
    """Output from the citation and claim verifier."""

    all_claims_supported: bool = True
    unsupported_claims: list[str] = Field(default_factory=list)
    weakened_claims: list[str] = Field(default_factory=list)
    contradictions_noted: list[str] = Field(default_factory=list)
