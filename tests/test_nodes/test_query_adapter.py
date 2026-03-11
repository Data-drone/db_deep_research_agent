"""Tests for the query_adapter node."""

from unittest.mock import AsyncMock, MagicMock

import pytest

from deep_research.models import SubQuestion
from deep_research.nodes.query_adapter import query_adapter_node
from deep_research.state import create_initial_state


def make_mock_model(content: str):
    """Create a mock model that returns a fixed response."""
    model = AsyncMock()
    response = MagicMock()
    response.content = content
    model.ainvoke.return_value = response
    return model


# --- Test: adapted_queries populated for each tool ---


@pytest.mark.asyncio
async def test_query_adapter_populates_adapted_queries():
    """Each tool in assigned_tools gets an entry in adapted_queries."""
    model = make_mock_model("What was the quarterly revenue breakdown?")
    sq = SubQuestion(
        subquestion_id="sq-1",
        question="What was Q3 revenue?",
        assigned_tools=["genie_sales", "vector_search_docs"],
    )
    state = create_initial_state(user_query="test", selected_tools=["genie_sales", "vector_search_docs"])
    state["research_plan"] = [sq]

    result = await query_adapter_node(state, model=model)

    plan = result["research_plan"]
    assert len(plan) == 1
    assert "genie_sales" in plan[0].adapted_queries
    assert "vector_search_docs" in plan[0].adapted_queries
    # Both tools are known types, so the model should have been called twice
    assert model.ainvoke.call_count == 2


@pytest.mark.asyncio
async def test_query_adapter_genie_tool_calls_model():
    """A genie tool should invoke the model with data_query tool type."""
    model = make_mock_model("Show me the Q3 revenue figures")
    sq = SubQuestion(
        subquestion_id="sq-1",
        question="What was Q3 revenue?",
        assigned_tools=["genie_sales"],
    )
    state = create_initial_state(user_query="test", selected_tools=["genie_sales"])
    state["research_plan"] = [sq]

    result = await query_adapter_node(state, model=model)

    plan = result["research_plan"]
    assert plan[0].adapted_queries["genie_sales"] == "Show me the Q3 revenue figures"
    # Verify model was called with correct system prompt containing data_query
    call_args = model.ainvoke.call_args[0][0]
    assert "data_query" in call_args[0]["content"]


@pytest.mark.asyncio
async def test_query_adapter_vector_search_tool():
    """A vector_search tool should invoke the model with semantic_search type."""
    model = make_mock_model("quarterly revenue analysis trends")
    sq = SubQuestion(
        subquestion_id="sq-1",
        question="What was Q3 revenue?",
        assigned_tools=["vector_search_reports"],
    )
    state = create_initial_state(user_query="test", selected_tools=["vector_search_reports"])
    state["research_plan"] = [sq]

    result = await query_adapter_node(state, model=model)

    plan = result["research_plan"]
    assert plan[0].adapted_queries["vector_search_reports"] == "quarterly revenue analysis trends"
    call_args = model.ainvoke.call_args[0][0]
    assert "semantic_search" in call_args[0]["content"]


@pytest.mark.asyncio
async def test_query_adapter_knowledge_tool():
    """A knowledge tool should invoke the model with knowledge_retrieval type."""
    model = make_mock_model("What is Q3 revenue for 2025?")
    sq = SubQuestion(
        subquestion_id="sq-1",
        question="What was Q3 revenue?",
        assigned_tools=["knowledge_base"],
    )
    state = create_initial_state(user_query="test", selected_tools=["knowledge_base"])
    state["research_plan"] = [sq]

    result = await query_adapter_node(state, model=model)

    plan = result["research_plan"]
    assert plan[0].adapted_queries["knowledge_base"] == "What is Q3 revenue for 2025?"
    call_args = model.ainvoke.call_args[0][0]
    assert "knowledge_retrieval" in call_args[0]["content"]


# --- Test: answered sub-questions are skipped ---


@pytest.mark.asyncio
async def test_query_adapter_skips_answered():
    """Sub-questions with status 'answered' should not be processed."""
    model = make_mock_model("should not be called")
    sq_answered = SubQuestion(
        subquestion_id="sq-1",
        question="Already answered",
        assigned_tools=["genie_sales"],
        status="answered",
    )
    sq_pending = SubQuestion(
        subquestion_id="sq-2",
        question="Still pending",
        assigned_tools=["genie_sales"],
    )
    state = create_initial_state(user_query="test", selected_tools=["genie_sales"])
    state["research_plan"] = [sq_answered, sq_pending]

    result = await query_adapter_node(state, model=model)

    plan = result["research_plan"]
    # Answered sub-question should have no adapted_queries
    assert plan[0].adapted_queries == {}
    # Pending sub-question should have adapted_queries populated
    assert "genie_sales" in plan[1].adapted_queries
    # Model should be called only once (for the pending sub-question)
    assert model.ainvoke.call_count == 1


# --- Test: unknown tool falls back to original question ---


@pytest.mark.asyncio
async def test_query_adapter_unknown_tool_uses_original():
    """Unknown tool types should use the original question without calling model."""
    model = make_mock_model("should not be called")
    sq = SubQuestion(
        subquestion_id="sq-1",
        question="What was Q3 revenue?",
        assigned_tools=["custom_unknown_tool"],
    )
    state = create_initial_state(user_query="test", selected_tools=["custom_unknown_tool"])
    state["research_plan"] = [sq]

    result = await query_adapter_node(state, model=model)

    plan = result["research_plan"]
    assert plan[0].adapted_queries["custom_unknown_tool"] == "What was Q3 revenue?"
    # Model should NOT be called for unknown tools
    model.ainvoke.assert_not_called()


@pytest.mark.asyncio
async def test_query_adapter_mixed_known_and_unknown_tools():
    """Mixed tool types: known tools call model, unknown tools use original."""
    model = make_mock_model("Reformulated query for genie")
    sq = SubQuestion(
        subquestion_id="sq-1",
        question="What was Q3 revenue?",
        assigned_tools=["genie_sales", "some_random_tool"],
    )
    state = create_initial_state(user_query="test", selected_tools=["genie_sales", "some_random_tool"])
    state["research_plan"] = [sq]

    result = await query_adapter_node(state, model=model)

    plan = result["research_plan"]
    # genie tool should use model's response
    assert plan[0].adapted_queries["genie_sales"] == "Reformulated query for genie"
    # unknown tool should use original question
    assert plan[0].adapted_queries["some_random_tool"] == "What was Q3 revenue?"
    # Model called only once (for the genie tool)
    assert model.ainvoke.call_count == 1


@pytest.mark.asyncio
async def test_query_adapter_empty_plan():
    """Empty research plan should return empty plan without errors."""
    model = make_mock_model("should not be called")
    state = create_initial_state(user_query="test", selected_tools=[])
    state["research_plan"] = []

    result = await query_adapter_node(state, model=model)

    assert result["research_plan"] == []
    model.ainvoke.assert_not_called()


@pytest.mark.asyncio
async def test_query_adapter_vs_shorthand():
    """Tool names containing 'vs' should be classified as semantic_search."""
    model = make_mock_model("semantic search terms")
    sq = SubQuestion(
        subquestion_id="sq-1",
        question="Find relevant docs",
        assigned_tools=["vs_index"],
    )
    state = create_initial_state(user_query="test", selected_tools=["vs_index"])
    state["research_plan"] = [sq]

    result = await query_adapter_node(state, model=model)

    plan = result["research_plan"]
    assert plan[0].adapted_queries["vs_index"] == "semantic search terms"
    call_args = model.ainvoke.call_args[0][0]
    assert "semantic_search" in call_args[0]["content"]


@pytest.mark.asyncio
async def test_query_adapter_ka_shorthand():
    """Tool names containing 'ka' should be classified as knowledge_retrieval."""
    model = make_mock_model("knowledge query")
    sq = SubQuestion(
        subquestion_id="sq-1",
        question="Find relevant info",
        assigned_tools=["ka_internal"],
    )
    state = create_initial_state(user_query="test", selected_tools=["ka_internal"])
    state["research_plan"] = [sq]

    result = await query_adapter_node(state, model=model)

    plan = result["research_plan"]
    assert plan[0].adapted_queries["ka_internal"] == "knowledge query"
    call_args = model.ainvoke.call_args[0][0]
    assert "knowledge_retrieval" in call_args[0]["content"]
