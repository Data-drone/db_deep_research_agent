"""Tests for the LangGraph research graph."""

from langgraph.graph import END

from deep_research.graph import _should_revise, build_research_graph
from deep_research.models import Budget, VerificationResult


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


# ── Bounded verifier revision loop ───────────────────────────────────────────

def _verification(supported: bool, claims=None):
    return VerificationResult(
        all_claims_supported=supported,
        unsupported_claims=claims or [],
        weakened_claims=[],
        contradictions_noted=[],
    )


def test_should_revise_ends_when_all_claims_are_supported():
    state = {"verification_result": _verification(True)}
    assert _should_revise(state) == END


def test_should_revise_ends_when_the_verifier_did_not_run():
    assert _should_revise({}) == END


def test_should_revise_returns_to_planner_on_unsupported_claims():
    state = {
        "verification_result": _verification(False, ["revenue tripled"]),
        "budget": Budget(),
        "verification_attempts": 1,
    }
    assert _should_revise(state) == "planner"


def test_default_budget_allows_exactly_one_revision():
    """The regression that made the loop dead: the first verifier pass is the
    check, not a revision, so comparing the raw pass count against a cap of 1
    ended the run before the planner was ever re-entered."""
    state = {
        "verification_result": _verification(False, ["revenue tripled"]),
        "budget": Budget(),  # max_verification_attempts defaults to 1
        "verification_attempts": 1,  # verifier has run once
    }
    assert _should_revise(state) == "planner"
    state["verification_attempts"] = 2  # verifier has run again after the revision
    assert _should_revise(state) == END


def test_should_revise_honours_a_raised_attempt_budget():
    state = {
        "verification_result": _verification(False, ["claim"]),
        "budget": Budget(max_verification_attempts=3),
        "verification_attempts": 3,
    }
    assert _should_revise(state) == "planner"
    state["verification_attempts"] = 4
    assert _should_revise(state) == END


def test_should_revise_never_loops_when_revisions_are_disabled():
    state = {
        "verification_result": _verification(False, ["claim"]),
        "budget": Budget(max_verification_attempts=0),
        "verification_attempts": 1,
    }
    assert _should_revise(state) == END


def test_should_revise_defaults_to_one_revision_without_a_budget():
    state = {"verification_result": _verification(False, ["claim"]), "verification_attempts": 1}
    assert _should_revise(state) == "planner"
    state["verification_attempts"] = 2
    assert _should_revise(state) == END


def test_verifier_pass_count_and_revision_gate_agree():
    """Drive the counter the way ``verifier_node`` does rather than by hand, so
    the gate and the increment cannot drift apart again."""
    budget = Budget(max_verification_attempts=1)
    state = {"verification_result": _verification(False, ["claim"]), "budget": budget}
    passes = 0
    routes = []
    for _ in range(5):
        passes = state.get("verification_attempts", 0) + 1  # verifier_node's line
        state["verification_attempts"] = passes
        route = _should_revise(state)
        routes.append(route)
        if route == END:
            break
    assert routes == ["planner", END], routes
