"""Tests for the LangGraph research graph."""

from deep_research.graph import build_research_graph


def test_graph_builds_without_error():
    graph = build_research_graph(model=None, mcp_manager=None)
    assert graph is not None


def test_graph_has_expected_nodes():
    graph = build_research_graph(model=None, mcp_manager=None)
    node_names = set(graph.get_graph().nodes.keys())
    expected = {
        "clarifier", "planner", "authorizer", "query_adapter", "researcher",
        "normalizer", "scorer", "evaluator", "compressor", "synthesizer", "verifier",
    }
    assert expected.issubset(node_names)


def test_graph_builds_with_critic_model():
    graph = build_research_graph(model=None, mcp_manager=None, critic_model=None)
    assert graph is not None


def test_graph_has_expected_nodes_with_critic():
    graph = build_research_graph(model=None, mcp_manager=None, critic_model=None)
    node_names = set(graph.get_graph().nodes.keys())
    expected = {
        "clarifier", "planner", "authorizer", "query_adapter", "researcher",
        "normalizer", "scorer", "evaluator", "compressor", "synthesizer", "verifier",
    }
    assert expected.issubset(node_names)


def test_graph_query_adapter_between_authorizer_and_researcher():
    """Verify query_adapter sits between authorizer and researcher in the graph."""
    graph = build_research_graph(model=None, mcp_manager=None)
    g = graph.get_graph()
    # Check edges: authorizer -> query_adapter -> researcher
    authorizer_edges = [e.target for e in g.edges if e.source == "authorizer"]
    query_adapter_edges = [e.target for e in g.edges if e.source == "query_adapter"]
    assert "query_adapter" in authorizer_edges
    assert "researcher" in query_adapter_edges


def test_graph_scorer_between_normalizer_and_evaluator():
    """Verify scorer sits between normalizer and evaluator in the graph."""
    graph = build_research_graph(model=None, mcp_manager=None)
    g = graph.get_graph()
    normalizer_edges = [e.target for e in g.edges if e.source == "normalizer"]
    scorer_edges = [e.target for e in g.edges if e.source == "scorer"]
    assert "scorer" in normalizer_edges
    assert "evaluator" in scorer_edges


def test_graph_verifier_feedback_loop():
    """Verify verifier has conditional edges back to planner or to END."""
    graph = build_research_graph(model=None, mcp_manager=None)
    g = graph.get_graph()
    verifier_edges = [e.target for e in g.edges if e.source == "verifier"]
    assert "planner" in verifier_edges, "verifier should have edge back to planner"
    assert "__end__" in verifier_edges, "verifier should have edge to END"
