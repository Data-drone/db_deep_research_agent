"""LangGraph research agent graph builder."""

from __future__ import annotations

from functools import partial
from typing import Any

from langgraph.graph import END, StateGraph

from deep_research.mcp_client import MCPClientManager
from deep_research.nodes import (
    authorizer_node,
    clarifier_node,
    compressor_node,
    evaluator_node,
    normalizer_node,
    planner_node,
    query_adapter_node,
    researcher_node,
    scorer_node,
    synthesizer_node,
    verifier_node,
)
from deep_research.state import ResearchState


def _should_continue(state: ResearchState) -> str:
    """Routing function after evaluator: continue research or compress."""
    decision = state.get("evaluator_decision")
    if decision and decision.decision == "continue":
        budget = state.get("budget")
        iteration = state.get("iteration_count", 0)
        if budget and iteration >= budget.max_iterations:
            return "compressor"
        return "planner"
    return "compressor"


def build_research_graph(
    model: Any,
    mcp_manager: MCPClientManager | None,
    critic_model: Any | None = None,
) -> Any:
    """Build and compile the full research agent graph."""
    _critic = critic_model or model

    graph = StateGraph(ResearchState)

    # Worker nodes
    graph.add_node("clarifier", partial(clarifier_node, model=model))
    graph.add_node("planner", partial(planner_node, model=model, critic_model=_critic))
    graph.add_node("authorizer", partial(authorizer_node, model=model))
    graph.add_node("query_adapter", partial(query_adapter_node, model=model))
    graph.add_node("researcher", partial(researcher_node, model=model, mcp_manager=mcp_manager))
    graph.add_node("normalizer", partial(normalizer_node, model=model))
    graph.add_node("compressor", partial(compressor_node, model=model))
    graph.add_node("synthesizer", partial(synthesizer_node, model=model))

    # Critic nodes (GPT-5.4 when available, falls back to worker)
    graph.add_node("scorer", partial(scorer_node, model=_critic))
    graph.add_node("evaluator", partial(evaluator_node, model=_critic))
    graph.add_node("verifier", partial(verifier_node, model=_critic))

    # Define edges
    graph.set_entry_point("clarifier")
    graph.add_edge("clarifier", "planner")
    graph.add_edge("planner", "authorizer")
    graph.add_edge("authorizer", "query_adapter")
    graph.add_edge("query_adapter", "researcher")
    graph.add_edge("researcher", "normalizer")
    graph.add_edge("normalizer", "scorer")
    graph.add_edge("scorer", "evaluator")
    graph.add_conditional_edges("evaluator", _should_continue, {
        "planner": "planner",
        "compressor": "compressor",
    })
    graph.add_edge("compressor", "synthesizer")
    graph.add_edge("synthesizer", "verifier")
    graph.add_edge("verifier", END)

    return graph.compile()
