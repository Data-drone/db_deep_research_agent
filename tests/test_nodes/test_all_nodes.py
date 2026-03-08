"""Tests for all agent graph nodes using mock LLM."""

import json
from unittest.mock import AsyncMock, MagicMock

import pytest

from deep_research.models import Budget, CompressedFindings, Evidence, SubQuestion
from deep_research.nodes.clarifier import clarifier_node
from deep_research.nodes.planner import planner_node
from deep_research.nodes.authorizer import authorizer_node
from deep_research.nodes.researcher import researcher_node
from deep_research.nodes.normalizer import normalizer_node
from deep_research.nodes.evaluator import evaluator_node
from deep_research.nodes.compressor import compressor_node
from deep_research.nodes.synthesizer import synthesizer_node
from deep_research.nodes.verifier import verifier_node
from deep_research.state import create_initial_state


def make_mock_model(content: str):
    """Create a mock model that returns a fixed response."""
    model = AsyncMock()
    response = MagicMock()
    response.content = content
    model.ainvoke.return_value = response
    return model


def make_mock_mcp():
    """Create a mock MCP manager."""
    mcp = AsyncMock()
    mcp.call_tool.return_value = {
        "result": "Q3 revenue was $12.4M, up 15% YoY",
        "source": "genie_sales",
        "table": "catalog.sales.quarterly_revenue",
    }
    return mcp


# --- Clarifier ---

@pytest.mark.asyncio
async def test_clarifier_clear_query():
    model = make_mock_model('{"clarification_needed": false, "clarified_query": "What was Q3 revenue?"}')
    state = create_initial_state(user_query="What was Q3 revenue?", selected_tools=["genie"])
    result = await clarifier_node(state, model=model)
    assert result["clarification_needed"] is False
    assert result["clarified_query"] == "What was Q3 revenue?"


@pytest.mark.asyncio
async def test_clarifier_ambiguous_query():
    model = make_mock_model('{"clarification_needed": true, "question": "Which quarter do you mean?"}')
    state = create_initial_state(user_query="What was revenue?", selected_tools=["genie"])
    result = await clarifier_node(state, model=model)
    assert result["clarification_needed"] is True


@pytest.mark.asyncio
async def test_clarifier_parse_error_fallback():
    model = make_mock_model("this is not json")
    state = create_initial_state(user_query="test query", selected_tools=[])
    result = await clarifier_node(state, model=model)
    assert result["clarification_needed"] is False
    assert result["clarified_query"] == "test query"


# --- Planner ---

@pytest.mark.asyncio
async def test_planner_creates_sub_questions():
    model = make_mock_model(json.dumps({
        "sub_questions": [
            {"question": "What was Q3 revenue?", "assigned_tools": ["genie_sales"]},
            {"question": "How does it compare to Q2?", "assigned_tools": ["genie_sales"]},
        ],
        "estimated_iterations": 1,
    }))
    state = create_initial_state(user_query="Q3 revenue analysis", selected_tools=["genie_sales"])
    result = await planner_node(state, model=model)
    assert len(result["research_plan"]) == 2
    assert all(isinstance(sq, SubQuestion) for sq in result["research_plan"])
    assert result["research_plan"][0].subquestion_id.startswith("sq-")


@pytest.mark.asyncio
async def test_planner_fallback_on_parse_error():
    model = make_mock_model("broken json")
    state = create_initial_state(user_query="test", selected_tools=["tool_a"])
    result = await planner_node(state, model=model)
    assert len(result["research_plan"]) == 1


# --- Authorizer ---

@pytest.mark.asyncio
async def test_authorizer_maps_tools():
    model = make_mock_model("")
    sq = SubQuestion(subquestion_id="sq-1", question="Q?", assigned_tools=["genie", "vs"])
    state = create_initial_state(user_query="test", selected_tools=[])
    state["research_plan"] = [sq]
    result = await authorizer_node(state, model=model)
    assert "genie" in result["tool_assignments"]
    assert "vs" in result["tool_assignments"]


# --- Researcher ---

@pytest.mark.asyncio
async def test_researcher_gathers_evidence():
    model = make_mock_model("")
    mcp = make_mock_mcp()
    sq = SubQuestion(subquestion_id="sq-1", question="Q3 revenue?", assigned_tools=["genie_sales"])
    state = create_initial_state(user_query="test", selected_tools=["genie_sales"])
    state["research_plan"] = [sq]
    state["budget"] = Budget(max_tool_calls=10)

    result = await researcher_node(state, model=model, mcp_manager=mcp)
    assert len(result["evidence"]) == 1
    assert result["evidence"][0].tool_call_id.startswith("tc-")
    assert len(result["tool_call_log"]) == 1
    assert result["iteration_count"] == 1
    assert result["tool_calls_used"] == 1


@pytest.mark.asyncio
async def test_researcher_handles_tool_error():
    model = make_mock_model("")
    mcp = AsyncMock()
    mcp.call_tool.side_effect = RuntimeError("Connection failed")
    sq = SubQuestion(subquestion_id="sq-1", question="test", assigned_tools=["broken_tool"])
    state = create_initial_state(user_query="test", selected_tools=["broken_tool"])
    state["research_plan"] = [sq]
    state["budget"] = Budget(max_tool_calls=10)

    result = await researcher_node(state, model=model, mcp_manager=mcp)
    assert len(result["evidence"]) == 0
    assert len(result["tool_call_log"]) == 1
    assert result["tool_call_log"][0].status == "error"
    assert result["tool_call_log"][0].error_type == "RuntimeError"


@pytest.mark.asyncio
async def test_researcher_respects_budget():
    model = make_mock_model("")
    mcp = make_mock_mcp()
    sq = SubQuestion(subquestion_id="sq-1", question="test", assigned_tools=["a", "b", "c"])
    state = create_initial_state(user_query="test", selected_tools=["a", "b", "c"])
    state["research_plan"] = [sq]
    state["budget"] = Budget(max_tool_calls=2)

    result = await researcher_node(state, model=model, mcp_manager=mcp)
    assert result["tool_calls_used"] == 2  # stopped at budget


@pytest.mark.asyncio
async def test_researcher_uses_adapted_queries():
    """When adapted_queries is populated, the researcher should use the
    adapted query string instead of sq.question when calling the tool."""
    model = make_mock_model("")
    mcp = make_mock_mcp()
    sq = SubQuestion(
        subquestion_id="sq-1",
        question="What was Q3 revenue?",
        assigned_tools=["genie_sales"],
        adapted_queries={"genie_sales": "SELECT revenue FROM quarterly WHERE quarter='Q3'"},
    )
    state = create_initial_state(user_query="test", selected_tools=["genie_sales"])
    state["research_plan"] = [sq]
    state["budget"] = Budget(max_tool_calls=10)

    result = await researcher_node(state, model=model, mcp_manager=mcp)

    # Verify the adapted query was sent, not the original question
    mcp.call_tool.assert_called_once()
    call_args = mcp.call_tool.call_args
    assert call_args[0][2] == {"query": "SELECT revenue FROM quarterly WHERE quarter='Q3'"}
    # Evidence should still be collected
    assert len(result["evidence"]) == 1
    assert result["tool_calls_used"] == 1


@pytest.mark.asyncio
async def test_researcher_parallel_multiple_sub_questions():
    """Two sub-questions should both produce evidence when run in parallel."""
    model = make_mock_model("")
    mcp = AsyncMock()
    mcp.call_tool.return_value = {
        "result": "some data",
        "source": "test_source",
    }
    sq1 = SubQuestion(subquestion_id="sq-1", question="Question A?", assigned_tools=["tool_a"])
    sq2 = SubQuestion(subquestion_id="sq-2", question="Question B?", assigned_tools=["tool_b"])
    state = create_initial_state(user_query="test", selected_tools=["tool_a", "tool_b"])
    state["research_plan"] = [sq1, sq2]
    state["budget"] = Budget(max_tool_calls=10)

    result = await researcher_node(state, model=model, mcp_manager=mcp)

    # Both sub-questions should produce evidence
    assert len(result["evidence"]) == 2
    assert len(result["tool_call_log"]) == 2
    assert result["tool_calls_used"] == 2
    # Both sub-questions should be in_progress
    assert sq1.status == "in_progress"
    assert sq2.status == "in_progress"
    # Each sub-question should have its own evidence
    assert len(sq1.evidence_ids) == 1
    assert len(sq2.evidence_ids) == 1
    # call_tool should have been called twice (once per sub-question)
    assert mcp.call_tool.call_count == 2


@pytest.mark.asyncio
async def test_researcher_parallel_one_fails():
    """If one sub-question's tool fails, the other should still succeed."""
    model = make_mock_model("")
    mcp = AsyncMock()

    async def _selective_call(tool_name, method, input_data):
        if tool_name == "broken_tool":
            raise RuntimeError("Connection failed")
        return {"result": "good data", "source": "ok_source"}

    mcp.call_tool.side_effect = _selective_call

    sq_ok = SubQuestion(subquestion_id="sq-1", question="Good question?", assigned_tools=["ok_tool"])
    sq_bad = SubQuestion(
        subquestion_id="sq-2", question="Bad question?", assigned_tools=["broken_tool"]
    )
    state = create_initial_state(
        user_query="test", selected_tools=["ok_tool", "broken_tool"]
    )
    state["research_plan"] = [sq_ok, sq_bad]
    state["budget"] = Budget(max_tool_calls=10)

    result = await researcher_node(state, model=model, mcp_manager=mcp)

    # The good sub-question should have evidence
    assert len(result["evidence"]) == 1
    assert result["evidence"][0].snippet == "good data"
    # Both sub-questions produced tool call logs (success + error)
    assert len(result["tool_call_log"]) == 2
    assert result["tool_calls_used"] == 2
    # Find the error tool call
    error_tc = [tc for tc in result["tool_call_log"] if tc.status == "error"]
    assert len(error_tc) == 1
    assert error_tc[0].tool_name == "broken_tool"
    assert error_tc[0].error_type == "RuntimeError"
    # Find the success tool call
    ok_tc = [tc for tc in result["tool_call_log"] if tc.status == "success"]
    assert len(ok_tc) == 1
    assert ok_tc[0].tool_name == "ok_tool"


# --- Normalizer ---

@pytest.mark.asyncio
async def test_normalizer_deduplicates():
    model = make_mock_model("")
    ev1 = Evidence(
        evidence_id="ev-1", source_id="genie", source_type="query",
        title="A", uri=None, snippet="same content", confidence=0.8,
        freshness="2026-01-01", tool_that_produced_it="genie",
        tool_call_id="tc-1", iteration=1,
    )
    ev2 = Evidence(
        evidence_id="ev-2", source_id="genie", source_type="query",
        title="B", uri=None, snippet="same content", confidence=0.8,
        freshness="2026-01-01", tool_that_produced_it="genie",
        tool_call_id="tc-2", iteration=2,
    )
    state = create_initial_state(user_query="test", selected_tools=[])
    state["evidence"] = [ev1, ev2]

    result = await normalizer_node(state, model=model)
    assert len(result["evidence"]) == 1  # deduplicated


@pytest.mark.asyncio
async def test_normalizer_keeps_different():
    model = make_mock_model("")
    ev1 = Evidence(
        evidence_id="ev-1", source_id="genie", source_type="query",
        title="A", uri=None, snippet="content A", confidence=0.8,
        freshness="2026-01-01", tool_that_produced_it="genie",
        tool_call_id="tc-1", iteration=1,
    )
    ev2 = Evidence(
        evidence_id="ev-2", source_id="vs", source_type="search",
        title="B", uri=None, snippet="different content", confidence=0.7,
        freshness="2026-01-01", tool_that_produced_it="vs",
        tool_call_id="tc-2", iteration=1,
    )
    state = create_initial_state(user_query="test", selected_tools=[])
    state["evidence"] = [ev1, ev2]

    result = await normalizer_node(state, model=model)
    assert len(result["evidence"]) == 2


# --- Evaluator ---

@pytest.mark.asyncio
async def test_evaluator_stop():
    model = make_mock_model(json.dumps({
        "sufficiency_score": 0.9,
        "missing_facets": [],
        "recommended_actions": [],
        "decision": "stop",
        "reason": "All answered",
    }))
    state = create_initial_state(user_query="test", selected_tools=[])
    state["research_plan"] = [
        SubQuestion(subquestion_id="sq-1", question="Q?", assigned_tools=["g"], evidence_ids=["ev-1"])
    ]
    state["evidence"] = [MagicMock(source_id="g", snippet="data")]

    result = await evaluator_node(state, model=model)
    assert result["evaluator_decision"].decision == "stop"
    assert result["sufficiency_score"] == 0.9


@pytest.mark.asyncio
async def test_evaluator_continue():
    model = make_mock_model(json.dumps({
        "sufficiency_score": 0.3,
        "missing_facets": ["competitor data"],
        "recommended_actions": ["search vs"],
        "decision": "continue",
        "reason": "Incomplete",
    }))
    state = create_initial_state(user_query="test", selected_tools=[])
    state["research_plan"] = [
        SubQuestion(subquestion_id="sq-1", question="Q?", assigned_tools=["g"])
    ]
    state["evidence"] = []

    result = await evaluator_node(state, model=model)
    assert result["evaluator_decision"].decision == "continue"


@pytest.mark.asyncio
async def test_evaluator_budget_forces_stop():
    model = make_mock_model(json.dumps({
        "sufficiency_score": 0.3,
        "missing_facets": ["more data"],
        "recommended_actions": [],
        "decision": "continue",
        "reason": "Need more",
    }))
    state = create_initial_state(user_query="test", selected_tools=[])
    state["research_plan"] = []
    state["evidence"] = []
    state["iteration_count"] = 5  # at max
    state["budget"] = Budget(max_iterations=5)

    result = await evaluator_node(state, model=model)
    assert result["evaluator_decision"].decision == "stop"
    assert result["evaluator_decision"].budget_exhausted is True


# --- Compressor ---

@pytest.mark.asyncio
async def test_compressor_produces_findings():
    model = make_mock_model(json.dumps({
        "key_findings": ["Revenue $12.4M"],
        "open_questions": [],
        "uncertainties": ["Estimate only"],
        "contradictions": [],
    }))
    ev = Evidence(
        evidence_id="ev-1", source_id="g", source_type="q", title="T",
        uri=None, snippet="Revenue $12.4M", confidence=0.8,
        freshness="2026-01-01", tool_that_produced_it="g",
        tool_call_id="tc-1", iteration=1,
    )
    state = create_initial_state(user_query="test", selected_tools=[])
    state["evidence"] = [ev]

    result = await compressor_node(state, model=model)
    assert isinstance(result["compressed_findings"], CompressedFindings)
    assert "Revenue $12.4M" in result["compressed_findings"].key_findings


# --- Synthesizer ---

@pytest.mark.asyncio
async def test_synthesizer_chat_mode():
    model = make_mock_model("Q3 revenue was $12.4M, up 15% YoY. [Source: Sales Data]")
    state = create_initial_state(user_query="Q3 revenue?", selected_tools=[])
    state["output_mode"] = "chat"
    state["compressed_findings"] = CompressedFindings(
        key_findings=["Revenue $12.4M"], open_questions=[], uncertainties=[], contradictions=[]
    )

    result = await synthesizer_node(state, model=model)
    assert "12.4M" in result["final_output"]


@pytest.mark.asyncio
async def test_synthesizer_report_mode():
    model = make_mock_model("## Executive Summary\nRevenue was $12.4M.\n## Sources\n- Sales Data")
    state = create_initial_state(user_query="Revenue analysis", selected_tools=[])
    state["output_mode"] = "report"
    state["compressed_findings"] = CompressedFindings(
        key_findings=["Revenue $12.4M"], open_questions=[], uncertainties=[], contradictions=[]
    )

    result = await synthesizer_node(state, model=model)
    assert "Executive Summary" in result["final_output"]


@pytest.mark.asyncio
async def test_synthesizer_streams_tokens():
    """Synthesizer pushes token events to job_manager when available."""
    from deep_research.api.jobs import JobManager

    jm = JobManager()
    job_id = jm.create_job(query="test", tools=[])

    # Mock model with astream that yields chunks
    class _StreamingModel:
        async def astream(self, messages):
            for word in ["Hello", " world", "!"]:
                chunk = MagicMock()
                chunk.content = word
                yield chunk

        async def ainvoke(self, messages):
            resp = MagicMock()
            resp.content = "Hello world!"
            return resp

    model = _StreamingModel()
    state = create_initial_state(user_query="test", selected_tools=[])
    state["_job_manager"] = jm
    state["_job_id"] = job_id

    result = await synthesizer_node(state, model=model)
    assert result["final_output"] == "Hello world!"

    # Check token events were pushed
    queue = jm.get_event_queue(job_id)
    events = []
    while not queue.empty():
        events.append(queue.get_nowait())
    token_events = [e for e in events if e["type"] == "token"]
    assert len(token_events) == 3
    assert token_events[0]["content"] == "Hello"
    assert token_events[1]["content"] == " world"
    assert token_events[2]["content"] == "!"


# --- Verifier ---

@pytest.mark.asyncio
async def test_verifier_all_supported():
    model = make_mock_model(json.dumps({
        "all_claims_supported": True,
        "unsupported_claims": [],
        "weakened_claims": [],
        "contradictions_noted": [],
    }))
    state = create_initial_state(user_query="test", selected_tools=[])
    state["final_output"] = "Revenue was $12.4M"
    state["evidence"] = [MagicMock(source_id="g", snippet="Revenue $12.4M")]

    result = await verifier_node(state, model=model)
    assert result["verification_result"].all_claims_supported is True


@pytest.mark.asyncio
async def test_verifier_unsupported_claims():
    model = make_mock_model(json.dumps({
        "all_claims_supported": False,
        "unsupported_claims": ["Market share grew 20%"],
        "weakened_claims": [],
        "contradictions_noted": [],
    }))
    state = create_initial_state(user_query="test", selected_tools=[])
    state["final_output"] = "Revenue grew. Market share grew 20%."
    state["evidence"] = [MagicMock(source_id="g", snippet="Revenue grew")]

    result = await verifier_node(state, model=model)
    assert result["verification_result"].all_claims_supported is False
    assert "Market share grew 20%" in result["verification_result"].unsupported_claims
