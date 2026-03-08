"""Tests for LangGraph state schema."""

from deep_research.models import Budget
from deep_research.state import ResearchState, create_initial_state


def test_create_initial_state():
    state = create_initial_state(
        user_query="What was Q3 revenue?",
        selected_tools=["genie_sales"],
        output_mode="chat",
    )
    assert state["user_query"] == "What was Q3 revenue?"
    assert state["selected_tools"] == ["genie_sales"]
    assert state["output_mode"] == "chat"
    assert state["iteration_count"] == 0
    assert state["tool_calls_used"] == 0
    assert state["evidence"] == []
    assert state["clarification_needed"] is False
    assert state["verification_attempts"] == 0
    assert state["perspectives"] == []


def test_initial_state_report_mode():
    state = create_initial_state(
        user_query="Deep analysis of churn",
        selected_tools=["genie_sales", "vector_search_kb"],
        output_mode="report",
    )
    assert state["output_mode"] == "report"
    assert len(state["selected_tools"]) == 2


def test_initial_state_custom_budget():
    budget = Budget(max_iterations=10, max_tool_calls=50, time_cap_seconds=300)
    state = create_initial_state(
        user_query="test",
        selected_tools=[],
        budget=budget,
    )
    assert state["budget"].max_iterations == 10
    assert state["budget"].max_tool_calls == 50


def test_initial_state_default_budget():
    state = create_initial_state(user_query="test", selected_tools=[])
    assert state["budget"].max_iterations == 5
    assert state["budget"].max_tool_calls == 20
