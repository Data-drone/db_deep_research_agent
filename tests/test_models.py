"""Tests for core data models."""

import json
from dataclasses import asdict
from datetime import datetime, timezone

from deep_research.models import (
    Budget,
    Citation,
    CompressedFindings,
    Evidence,
    EvaluatorDecision,
    SubQuestion,
    ToolCall,
)


def test_evidence_creation():
    e = Evidence(
        evidence_id="ev-001",
        source_id="genie_sales",
        source_type="genie_query",
        title="Q3 Revenue",
        uri="catalog.schema.sales_table",
        snippet="Q3 revenue was $12M, up 15% YoY",
        confidence=0.85,
        freshness="2026-03-01T00:00:00Z",
        tool_that_produced_it="genie_sales",
        tool_call_id="tc-001",
        iteration=1,
        metadata={"query": "Q3 revenue"},
    )
    assert e.evidence_id == "ev-001"
    assert e.confidence == 0.85
    assert e.tool_call_id == "tc-001"


def test_evidence_optional_fields():
    e = Evidence(
        evidence_id="ev-002",
        source_id="vs",
        source_type="vector_search",
        title="Doc",
        uri=None,
        snippet="text",
        confidence=0.5,
        freshness="2026-01-01",
        tool_that_produced_it="vs",
        tool_call_id="tc-002",
        iteration=0,
    )
    assert e.retrieved_at is None
    assert e.content_hash is None
    assert e.metadata == {}


def test_budget_defaults():
    b = Budget()
    assert b.max_iterations == 5
    assert b.max_tool_calls == 20
    assert b.time_cap_seconds == 120


def test_budget_custom():
    b = Budget(max_iterations=10, max_tool_calls=50, time_cap_seconds=300)
    assert b.max_iterations == 10


def test_evaluator_decision():
    d = EvaluatorDecision(
        sufficiency_score=0.7,
        missing_facets=["competitor analysis"],
        recommended_actions=["query vector_search for competitors"],
        decision="continue",
        reason="Missing competitor data",
    )
    assert d.decision == "continue"
    assert len(d.missing_facets) == 1
    assert d.budget_exhausted is False


def test_evaluator_budget_exhausted():
    d = EvaluatorDecision(
        sufficiency_score=0.4,
        missing_facets=["more data"],
        recommended_actions=[],
        decision="stop",
        reason="Budget exhausted",
        budget_exhausted=True,
    )
    assert d.budget_exhausted is True


def test_sub_question():
    sq = SubQuestion(
        subquestion_id="sq-001",
        question="What was Q3 revenue?",
        assigned_tools=["genie_sales"],
    )
    assert sq.status == "pending"
    assert sq.evidence_ids == []
    assert sq.iteration_created == 0


def test_sub_question_status_transition():
    sq = SubQuestion(
        subquestion_id="sq-001",
        question="Q3 revenue?",
        assigned_tools=["genie_sales"],
        status="answered",
        answer_summary="$12M",
        evidence_ids=["ev-001"],
    )
    assert sq.status == "answered"
    assert sq.answer_summary == "$12M"


def test_citation():
    c = Citation(
        claim_id="cl-001",
        claim="Revenue grew 15% YoY",
        evidence_ids=["ev-001"],
        confidence=0.85,
    )
    assert c.claim_id == "cl-001"
    assert len(c.evidence_ids) == 1


def test_tool_call():
    tc = ToolCall(
        tool_call_id="tc-001",
        tool_name="genie_sales",
        server_name="genie_sales",
        input_data={"query": "Q3 revenue"},
        output_data={"result": "$12M"},
        latency_ms=1500,
    )
    assert tc.status == "success"
    assert tc.latency_ms == 1500
    assert tc.iteration == 0


def test_tool_call_error():
    tc = ToolCall(
        tool_call_id="tc-002",
        tool_name="genie_sales",
        server_name="genie_sales",
        input_data={"query": "bad"},
        output_data={},
        latency_ms=500,
        status="error",
        error_type="auth_failure",
        error_message="Service principal expired",
    )
    assert tc.status == "error"
    assert tc.error_type == "auth_failure"


def test_compressed_findings():
    cf = CompressedFindings(
        key_findings=["Revenue grew 15%"],
        open_questions=["What about Q4?"],
        uncertainties=["Competitor data may be stale"],
        contradictions=[],
    )
    assert len(cf.key_findings) == 1
    assert len(cf.contradictions) == 0


def test_models_json_serializable():
    """All models should be serializable to JSON-compatible dicts."""
    e = Evidence(
        evidence_id="ev-001",
        source_id="genie",
        source_type="genie_query",
        title="Test",
        uri=None,
        snippet="data",
        confidence=0.9,
        freshness="2026-01-01",
        tool_that_produced_it="genie",
        tool_call_id="tc-001",
        iteration=1,
        retrieved_at=datetime(2026, 3, 7, tzinfo=timezone.utc),
    )
    d = asdict(e)
    # datetime needs to be converted for JSON
    d["retrieved_at"] = d["retrieved_at"].isoformat() if d["retrieved_at"] else None
    json_str = json.dumps(d)
    assert "ev-001" in json_str

    tc = ToolCall(
        tool_call_id="tc-001",
        tool_name="genie",
        server_name="genie",
        input_data={"q": "test"},
        output_data={"r": "data"},
        latency_ms=100,
    )
    d2 = asdict(tc)
    json_str2 = json.dumps(d2, default=str)
    assert "tc-001" in json_str2
