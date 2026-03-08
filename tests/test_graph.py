"""Tests for the LangGraph research graph."""

from deep_research.graph import build_research_graph


def test_graph_builds_without_error():
    graph = build_research_graph(model=None, mcp_manager=None)
    assert graph is not None


def test_graph_has_expected_nodes():
    graph = build_research_graph(model=None, mcp_manager=None)
    node_names = set(graph.get_graph().nodes.keys())
    expected = {
        "clarifier", "planner", "authorizer", "researcher",
        "normalizer", "evaluator", "compressor", "synthesizer", "verifier",
    }
    assert expected.issubset(node_names)


def test_graph_builds_with_critic_model():
    graph = build_research_graph(model=None, mcp_manager=None, critic_model=None)
    assert graph is not None


def test_graph_has_expected_nodes_with_critic():
    graph = build_research_graph(model=None, mcp_manager=None, critic_model=None)
    node_names = set(graph.get_graph().nodes.keys())
    expected = {
        "clarifier", "planner", "authorizer", "researcher",
        "normalizer", "evaluator", "compressor", "synthesizer", "verifier",
    }
    assert expected.issubset(node_names)
