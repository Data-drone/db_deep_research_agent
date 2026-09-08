"""LangGraph-like compatibility runner for the Pydantic AI pipeline."""

from __future__ import annotations

import asyncio
from collections.abc import AsyncIterator
from copy import deepcopy
from typing import Any

from deep_research.mcp_client import MCPClientManager
from deep_research_pydantic.agents import build_agents
from deep_research_pydantic.nodes import NodeDeps
from deep_research_pydantic.orchestrator import orchestrate_research
from deep_research_pydantic.state import ResearchState


class PydanticResearchPipeline:
    """Compatibility surface exposing ``astream`` and ``ainvoke``."""

    def __init__(
        self,
        deps: NodeDeps,
        *,
        max_verification_attempts: int = 0,
        max_node_executions: int = 50,
    ) -> None:
        self._deps = deps
        self._max_verification_attempts = max_verification_attempts
        self._max_node_executions = max_node_executions

    async def astream(
        self,
        initial_state: ResearchState,
        config: dict[str, Any] | None = None,
        *,
        stream_mode: str = "updates",
    ) -> AsyncIterator[dict[str, dict[str, Any]]]:
        """Yield one ``{node_name: update}`` event after each node."""

        if stream_mode != "updates":
            raise ValueError(
                "PydanticResearchPipeline supports only stream_mode='updates'"
            )

        recursion_limit = self._max_node_executions
        if config and "recursion_limit" in config:
            recursion_limit = int(config["recursion_limit"])

        queue: asyncio.Queue[
            tuple[str, dict[str, Any]] | object
        ] = asyncio.Queue()
        sentinel = object()

        async def on_progress(
            node_name: str,
            update: dict[str, Any],
        ) -> None:
            await queue.put((node_name, deepcopy(update)))

        async def run_pipeline() -> ResearchState:
            try:
                return await orchestrate_research(
                    initial_state,
                    self._deps,
                    progress_callback=on_progress,
                    max_verification_attempts=(
                        self._max_verification_attempts
                    ),
                    max_node_executions=recursion_limit,
                )
            finally:
                await queue.put(sentinel)

        task = asyncio.create_task(run_pipeline())

        try:
            while True:
                item = await queue.get()
                if item is sentinel:
                    break

                node_name, update = item
                yield {node_name: update}

            await task
        finally:
            if not task.done():
                task.cancel()
            await asyncio.gather(task, return_exceptions=True)

    async def ainvoke(
        self,
        initial_state: ResearchState,
        config: dict[str, Any] | None = None,
    ) -> ResearchState:
        """Run the pipeline and return the final accumulated state."""

        recursion_limit = self._max_node_executions
        if config and "recursion_limit" in config:
            recursion_limit = int(config["recursion_limit"])

        return await orchestrate_research(
            initial_state,
            self._deps,
            max_verification_attempts=self._max_verification_attempts,
            max_node_executions=recursion_limit,
        )


def build_research_pipeline(
    model: Any,
    mcp_manager: MCPClientManager | None,
    critic_model: Any | None = None,
) -> PydanticResearchPipeline:
    """Build a Pydantic AI pipeline using the LangGraph factory signature."""

    agents = build_agents(
        worker_model=model,
        critic_model=critic_model,
    )
    deps = NodeDeps(
        agents=agents,
        mcp_manager=mcp_manager,
    )
    return PydanticResearchPipeline(deps)
