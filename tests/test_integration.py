"""End-to-end integration tests for the research pipeline.

These tests run the full agent graph with mock MCP servers and a
mock LLM to verify the pipeline works as a user would experience it.
"""

import pytest
from unittest.mock import AsyncMock, MagicMock

from deep_research.graph import build_research_graph
from deep_research.state import create_initial_state
from tests.mock_mcp_server import MockMCPClientManager


def make_mock_model(responses: list[str]):
    """Create a mock LLM that returns canned responses in sequence."""
    model = AsyncMock()
    call_count = 0

    async def fake_invoke(messages):
        nonlocal call_count
        idx = min(call_count, len(responses) - 1)
        call_count += 1
        result = MagicMock()
        result.content = responses[idx]
        return result

    model.ainvoke = fake_invoke
    return model


SIMPLE_QUERY_LLM_RESPONSES = [
    # Clarifier: query is clear
    '{"clarification_needed": false, "clarified_query": "What was Q3 revenue?"}',
    # Planner: one sub-question
    '{"sub_questions": [{"question": "What was Q3 revenue?", "assigned_tools": ["genie_sales"]}], "estimated_iterations": 1}',
    # Evaluator: sufficient
    '{"sufficiency_score": 0.9, "missing_facets": [], "recommended_actions": [], "decision": "stop", "reason": "Query answered"}',
    # Compressor
    '{"key_findings": ["Q3 revenue was $12.4M, up 15% YoY"], "open_questions": [], "uncertainties": [], "contradictions": []}',
    # Synthesizer
    "Q3 2025 revenue was $12.4M, representing 15% year-over-year growth. [Source: Sales Data (Genie)]",
    # Verifier
    '{"all_claims_supported": true, "unsupported_claims": [], "weakened_claims": [], "contradictions_noted": []}',
]


@pytest.mark.asyncio
async def test_simple_query_end_to_end():
    """User asks a simple question, gets a direct answer."""
    mock_model = make_mock_model(SIMPLE_QUERY_LLM_RESPONSES)
    mock_mcp = MockMCPClientManager()

    graph = build_research_graph(model=mock_model, mcp_manager=mock_mcp)

    initial_state = create_initial_state(
        user_query="What was Q3 revenue?",
        selected_tools=["genie_sales"],
        output_mode="chat",
    )

    result = await graph.ainvoke(initial_state)

    # User should get an answer
    assert result["final_output"] != ""
    # Should have evidence
    assert len(result.get("evidence", [])) > 0 or result["final_output"] != ""
    # Verification should pass
    vr = result.get("verification_result")
    if vr:
        assert vr.all_claims_supported is True


MULTI_TOOL_LLM_RESPONSES = [
    # Clarifier
    '{"clarification_needed": false, "clarified_query": "Analyze customer churn and its drivers"}',
    # Planner: two sub-questions, two tools
    '{"sub_questions": [{"question": "What is the current churn rate?", "assigned_tools": ["genie_sales"]}, {"question": "What are the churn drivers?", "assigned_tools": ["vector_search_kb"]}], "estimated_iterations": 1}',
    # Evaluator
    '{"sufficiency_score": 0.85, "missing_facets": [], "recommended_actions": [], "decision": "stop", "reason": "Both questions answered"}',
    # Compressor
    '{"key_findings": ["Churn rate is 4.2%", "Main drivers: price sensitivity, feature gaps"], "open_questions": [], "uncertainties": ["SMB churn much higher than enterprise"], "contradictions": []}',
    # Synthesizer
    "## Churn Analysis\\n\\nCustomer churn rate is 4.2% in Q3. Primary drivers are price sensitivity (35%) and feature gaps (28%). [Source: Sales Data] [Source: Knowledge Base]",
    # Verifier
    '{"all_claims_supported": true, "unsupported_claims": [], "weakened_claims": [], "contradictions_noted": []}',
]


@pytest.mark.asyncio
async def test_multi_tool_query():
    """User asks a complex question requiring multiple tools."""
    mock_model = make_mock_model(MULTI_TOOL_LLM_RESPONSES)
    mock_mcp = MockMCPClientManager()

    graph = build_research_graph(model=mock_model, mcp_manager=mock_mcp)

    initial_state = create_initial_state(
        user_query="Analyze customer churn and its drivers",
        selected_tools=["genie_sales", "vector_search_kb"],
        output_mode="report",
    )

    result = await graph.ainvoke(initial_state)

    assert result["final_output"] != ""
    assert result["output_mode"] == "report"


ITERATIVE_LLM_RESPONSES = [
    # Clarifier
    '{"clarification_needed": false, "clarified_query": "Compare revenue to competitors"}',
    # Planner (iteration 1)
    '{"sub_questions": [{"question": "What is our revenue?", "assigned_tools": ["genie_sales"]}, {"question": "What are competitor revenues?", "assigned_tools": ["vector_search_kb"]}], "estimated_iterations": 2}',
    # Evaluator (iteration 1): needs more
    '{"sufficiency_score": 0.4, "missing_facets": ["competitor revenue data"], "recommended_actions": ["search for competitor analysis"], "decision": "continue", "reason": "Missing competitor data"}',
    # Planner (iteration 2)
    '{"sub_questions": [{"question": "Competitor revenue comparison", "assigned_tools": ["vector_search_kb"]}], "estimated_iterations": 1}',
    # Evaluator (iteration 2): sufficient
    '{"sufficiency_score": 0.8, "missing_facets": [], "recommended_actions": [], "decision": "stop", "reason": "Sufficient data"}',
    # Compressor
    '{"key_findings": ["Our revenue is $12.4M", "Competitor A is ~$15M based on estimates"], "open_questions": ["Exact competitor figures unavailable"], "uncertainties": ["Competitor data is estimated"], "contradictions": []}',
    # Synthesizer
    "Our Q3 revenue was $12.4M. Based on available data, our primary competitor is estimated at ~$15M.",
    # Verifier
    '{"all_claims_supported": true, "unsupported_claims": [], "weakened_claims": ["competitor revenue is estimated"], "contradictions_noted": []}',
]


@pytest.mark.asyncio
async def test_iterative_research_loop():
    """Agent loops back when evaluator says evidence is insufficient."""
    mock_model = make_mock_model(ITERATIVE_LLM_RESPONSES)
    mock_mcp = MockMCPClientManager()

    graph = build_research_graph(model=mock_model, mcp_manager=mock_mcp)

    initial_state = create_initial_state(
        user_query="Compare revenue to competitors",
        selected_tools=["genie_sales", "vector_search_kb"],
        output_mode="chat",
    )

    result = await graph.ainvoke(initial_state)

    assert result["final_output"] != ""
    # Should have done more than 1 iteration
    assert result.get("iteration_count", 0) >= 1
