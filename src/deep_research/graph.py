"""LangGraph research agent graph builder."""

from __future__ import annotations

import logging
from functools import partial
from typing import Any

logger = logging.getLogger(__name__)

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


def _should_revise(state: ResearchState) -> str:
    """Routing function after verifier: re-research unsupported claims, or finish.

    A revision re-runs the whole research pipeline (including Genie polling),
    which costs 60-90s, so the number of revisions is capped by
    ``Budget.max_verification_attempts`` (default 1). Beyond the cap, unsupported
    claims are logged and the draft ships as-is rather than looping indefinitely.

    ``verification_attempts`` counts verifier *passes*, and the first pass is not
    a revision - it is the initial check that decides whether one is needed. So
    the number of revisions already sent back to the planner is one less than the
    pass count, and comparing the raw pass count against the cap would end the
    run before any revision happened.
    """
    vr = state.get("verification_result")
    if not vr or vr.all_claims_supported:
        return END

    budget = state.get("budget")
    max_revisions = getattr(budget, "max_verification_attempts", 1) if budget else 1
    passes = state.get("verification_attempts", 0)
    revisions_done = max(passes - 1, 0)

    if revisions_done >= max_revisions:
        logger.info(
            "Verifier found unsupported claims after %d revision(s): %s - shipping as-is",
            revisions_done,
            vr.unsupported_claims[:3],
        )
        return END

    logger.info(
        "Verifier found unsupported claims (revision %d/%d): %s - re-researching",
        revisions_done + 1,
        max_revisions,
        vr.unsupported_claims[:3],
    )
    return "planner"


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
    graph.add_node("planner", partial(planner_node, model=model, critic_model=_critic, mcp_manager=mcp_manager))
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
    graph.add_conditional_edges("verifier", _should_revise, {
        "planner": "planner",
        END: END,
    })

    return graph.compile()
