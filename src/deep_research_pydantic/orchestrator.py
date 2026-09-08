"""Explicit asynchronous orchestration for the Pydantic AI pipeline."""

from __future__ import annotations

import inspect
import logging
from collections.abc import Awaitable, Callable
from typing import Any

from deep_research_pydantic.nodes import (
    NodeDeps,
    authorizer,
    clarifier,
    compressor,
    evaluator,
    normalizer,
    planner,
    query_adapter,
    researcher,
    scorer,
    synthesizer,
    verifier,
)
from deep_research_pydantic.state import ResearchState

logger = logging.getLogger(__name__)

NodeUpdate = dict[str, Any]
ProgressCallback = Callable[
    [str, NodeUpdate],
    Awaitable[None] | None,
]

_NODE_FUNCTIONS = {
    "clarifier": clarifier,
    "planner": planner,
    "authorizer": authorizer,
    "query_adapter": query_adapter,
    "researcher": researcher,
    "normalizer": normalizer,
    "scorer": scorer,
    "evaluator": evaluator,
    "compressor": compressor,
    "synthesizer": synthesizer,
    "verifier": verifier,
}


def _should_continue(state: ResearchState) -> str:
    decision = state.get("evaluator_decision")
    if decision and decision.decision == "continue":
        budget = state.get("budget")
        iteration = state.get("iteration_count", 0)
        if budget and iteration >= budget.max_iterations:
            return "compressor"
        return "planner"
    return "compressor"


def _should_revise(
    state: ResearchState,
    max_verification_attempts: int,
) -> str | None:
    verification_result = state.get("verification_result")

    if (
        verification_result
        and not verification_result.all_claims_supported
    ):
        if (
            max_verification_attempts > 0
            and state.get("verification_attempts", 0)
            < max_verification_attempts
        ):
            logger.info(
                "Verifier found unsupported claims; returning to planner: %s",
                verification_result.unsupported_claims[:3],
            )
            return "planner"

        logger.info(
            "Verifier found unsupported claims: %s — skipping re-research",
            verification_result.unsupported_claims[:3],
        )

    return None


def _next_node(
    current_node: str,
    state: ResearchState,
    max_verification_attempts: int,
) -> str | None:
    if current_node == "clarifier":
        return "planner"
    if current_node == "planner":
        return "authorizer"
    if current_node == "authorizer":
        return "query_adapter"
    if current_node == "query_adapter":
        return "researcher"
    if current_node == "researcher":
        return "normalizer"
    if current_node == "normalizer":
        return "scorer"
    if current_node == "scorer":
        return "evaluator"
    if current_node == "evaluator":
        return _should_continue(state)
    if current_node == "compressor":
        return "synthesizer"
    if current_node == "synthesizer":
        return "verifier"
    if current_node == "verifier":
        return _should_revise(
            state,
            max_verification_attempts,
        )
    raise ValueError(f"Unknown node: {current_node}")


async def _notify_progress(
    callback: ProgressCallback | None,
    node_name: str,
    update: NodeUpdate,
) -> None:
    if callback is None:
        return

    result = callback(node_name, update)
    if inspect.isawaitable(result):
        await result


async def orchestrate_research(
    initial_state: ResearchState,
    deps: NodeDeps,
    *,
    progress_callback: ProgressCallback | None = None,
    max_verification_attempts: int = 0,
    max_node_executions: int = 50,
) -> ResearchState:
    """Run the research topology until completion or cancellation.

    ``max_verification_attempts=0`` preserves the existing behavior in which
    verifier feedback is logged but never routed back to the planner.
    """

    if max_node_executions <= 0:
        raise ValueError("max_node_executions must be greater than zero")
    if max_verification_attempts < 0:
        raise ValueError(
            "max_verification_attempts cannot be negative"
        )

    state = ResearchState(**dict(initial_state))
    current_node: str | None = "clarifier"
    execution_count = 0

    while current_node is not None:
        if state.get("cancelled", False):
            logger.info(
                "Research pipeline cancelled before node %s",
                current_node,
            )
            break

        if execution_count >= max_node_executions:
            raise RuntimeError(
                "Research pipeline exceeded the node execution limit "
                f"of {max_node_executions}"
            )

        node_function = _NODE_FUNCTIONS[current_node]
        update = await node_function(state, deps)
        execution_count += 1

        state.update(update)
        await _notify_progress(
            progress_callback,
            current_node,
            update,
        )

        if state.get("cancelled", False):
            logger.info(
                "Research pipeline cancelled after node %s",
                current_node,
            )
            break

        current_node = _next_node(
            current_node,
            state,
            max_verification_attempts,
        )

    return state
