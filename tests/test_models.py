"""Tests for core data models."""

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
        iteration=1,
        metadata={"query": "Q3 revenue"},
    )
    assert e.evidence_id == "ev-001"
    assert e.confidence == 0.85


def test_budget_defaults():
    b = Budget()
    assert b.max_iterations == 5
    assert b.max_tool_calls == 20
    assert b.time_cap_seconds == 120


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


def test_sub_question():
    sq = SubQuestion(
        question="What was Q3 revenue?",
        assigned_tools=["genie_sales"],
        answered=False,
    )
    assert sq.answered is False


def test_citation():
    c = Citation(
        claim="Revenue grew 15% YoY",
        evidence_ids=["ev-001"],
        confidence=0.85,
    )
    assert len(c.evidence_ids) == 1


def test_tool_call():
    tc = ToolCall(
        tool_name="genie_sales",
        server_name="genie_sales",
        input_data={"query": "Q3 revenue"},
        output_data={"result": "$12M"},
        latency_ms=1500,
        success=True,
    )
    assert tc.success is True
    assert tc.latency_ms == 1500


def test_compressed_findings():
    cf = CompressedFindings(
        key_findings=["Revenue grew 15%"],
        open_questions=["What about Q4?"],
        uncertainties=["Competitor data may be stale"],
        contradictions=[],
    )
    assert len(cf.key_findings) == 1
    assert len(cf.contradictions) == 0
