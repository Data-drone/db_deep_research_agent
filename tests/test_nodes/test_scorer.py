"""Tests for the scorer node."""

from unittest.mock import AsyncMock, MagicMock

import pytest

from deep_research.models import Evidence, SubQuestion
from deep_research.nodes.scorer import scorer_node
from deep_research.state import create_initial_state


def _make_evidence(evidence_id: str, snippet: str, confidence: float = 0.8) -> Evidence:
    """Create an Evidence instance with sensible defaults."""
    return Evidence(
        evidence_id=evidence_id,
        source_id="src-1",
        source_type="vector_search",
        title="Test Evidence",
        uri=None,
        snippet=snippet,
        confidence=confidence,
        freshness="2026-01-01",
        tool_that_produced_it="vs_index",
        tool_call_id="tc-001",
        iteration=1,
    )


def _make_mock_model(score: float, reason: str = "test reason"):
    """Create a mock model that returns a JSON score response."""
    import json

    model = AsyncMock()
    response = MagicMock()
    response.content = json.dumps({"score": score, "reason": reason})
    model.ainvoke.return_value = response
    return model


def _make_mock_model_bad_json():
    """Create a mock model that returns unparseable content."""
    model = AsyncMock()
    response = MagicMock()
    response.content = "this is not valid json"
    model.ainvoke.return_value = response
    return model


# --- Test: evidence confidence is updated from model's score ---


@pytest.mark.asyncio
async def test_scorer_updates_confidence():
    """Evidence confidence should be updated to the model's assessed score."""
    model = _make_mock_model(0.75)
    ev = _make_evidence("ev-1", "Q3 revenue was $12M", confidence=0.8)
    sq = SubQuestion(
        subquestion_id="sq-1",
        question="What was Q3 revenue?",
        assigned_tools=["vs_index"],
        evidence_ids=["ev-1"],
    )
    state = create_initial_state(user_query="test", selected_tools=["vs_index"])
    state["research_plan"] = [sq]
    state["evidence"] = [ev]

    result = await scorer_node(state, model=model)

    assert len(result["evidence"]) == 1
    assert result["evidence"][0].confidence == 0.75


# --- Test: evidence below 0.3 is removed ---


@pytest.mark.asyncio
async def test_scorer_filters_low_relevance():
    """Evidence scored below 0.3 should be filtered out."""
    model = _make_mock_model(0.1)
    ev = _make_evidence("ev-1", "Irrelevant content", confidence=0.8)
    sq = SubQuestion(
        subquestion_id="sq-1",
        question="What was Q3 revenue?",
        assigned_tools=["vs_index"],
        evidence_ids=["ev-1"],
    )
    state = create_initial_state(user_query="test", selected_tools=["vs_index"])
    state["research_plan"] = [sq]
    state["evidence"] = [ev]

    result = await scorer_node(state, model=model)

    assert len(result["evidence"]) == 0


# --- Test: evidence above 0.3 is kept ---


@pytest.mark.asyncio
async def test_scorer_keeps_high_relevance():
    """Evidence scored at or above 0.3 should be kept."""
    model = _make_mock_model(0.85)
    ev = _make_evidence("ev-1", "Q3 revenue was $12M, up 15% YoY", confidence=0.8)
    sq = SubQuestion(
        subquestion_id="sq-1",
        question="What was Q3 revenue?",
        assigned_tools=["vs_index"],
        evidence_ids=["ev-1"],
    )
    state = create_initial_state(user_query="test", selected_tools=["vs_index"])
    state["research_plan"] = [sq]
    state["evidence"] = [ev]

    result = await scorer_node(state, model=model)

    assert len(result["evidence"]) == 1
    assert result["evidence"][0].confidence == 0.85


# --- Test: returns 0.5 on parse failure (not filtered) ---


@pytest.mark.asyncio
async def test_scorer_handles_parse_error():
    """On JSON parse failure, confidence defaults to 0.5 (above threshold, kept)."""
    model = _make_mock_model_bad_json()
    ev = _make_evidence("ev-1", "Some evidence text", confidence=0.8)
    sq = SubQuestion(
        subquestion_id="sq-1",
        question="What was Q3 revenue?",
        assigned_tools=["vs_index"],
        evidence_ids=["ev-1"],
    )
    state = create_initial_state(user_query="test", selected_tools=["vs_index"])
    state["research_plan"] = [sq]
    state["evidence"] = [ev]

    result = await scorer_node(state, model=model)

    # Default 0.5 is above threshold 0.3, so evidence should be kept
    assert len(result["evidence"]) == 1
    assert result["evidence"][0].confidence == 0.5


# --- Test: only scores evidence linked to each sub-question ---


@pytest.mark.asyncio
async def test_scorer_matches_evidence_to_sub_questions():
    """Only evidence whose evidence_id is in a sub-question's evidence_ids gets scored."""
    import json

    # Model returns different scores per call
    call_count = 0

    async def mock_ainvoke(messages):
        nonlocal call_count
        call_count += 1
        response = MagicMock()
        response.content = json.dumps({"score": 0.9, "reason": "relevant"})
        return response

    model = AsyncMock()
    model.ainvoke = mock_ainvoke

    ev1 = _make_evidence("ev-1", "Evidence for SQ1", confidence=0.8)
    ev2 = _make_evidence("ev-2", "Evidence for SQ2", confidence=0.8)
    ev3 = _make_evidence("ev-3", "Unlinked evidence", confidence=0.8)

    sq1 = SubQuestion(
        subquestion_id="sq-1",
        question="Question 1?",
        assigned_tools=["vs_index"],
        evidence_ids=["ev-1"],
    )
    sq2 = SubQuestion(
        subquestion_id="sq-2",
        question="Question 2?",
        assigned_tools=["vs_index"],
        evidence_ids=["ev-2"],
    )

    state = create_initial_state(user_query="test", selected_tools=["vs_index"])
    state["research_plan"] = [sq1, sq2]
    state["evidence"] = [ev1, ev2, ev3]

    result = await scorer_node(state, model=model)

    # Model should be called twice (ev-1 for sq-1, ev-2 for sq-2) — ev-3 is unlinked
    assert call_count == 2
    # ev-1 and ev-2 should have updated confidence (0.9)
    scored = {e.evidence_id: e.confidence for e in result["evidence"]}
    assert scored["ev-1"] == 0.9
    assert scored["ev-2"] == 0.9
    # ev-3 retains original confidence (0.8) — not scored but still above threshold
    assert scored["ev-3"] == 0.8


# --- Test: handles empty evidence list ---


@pytest.mark.asyncio
async def test_scorer_empty_evidence():
    """Empty evidence list should return empty list without errors."""
    model = _make_mock_model(0.9)
    state = create_initial_state(user_query="test", selected_tools=["vs_index"])
    state["research_plan"] = []
    state["evidence"] = []

    result = await scorer_node(state, model=model)

    assert result["evidence"] == []
    model.ainvoke.assert_not_called()
