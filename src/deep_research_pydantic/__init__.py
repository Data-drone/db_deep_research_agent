"""Pydantic AI implementation of the deep research pipeline."""

from __future__ import annotations

from deep_research_pydantic.agents import AgentBundle, build_agents, build_model
from deep_research_pydantic.nodes import NodeDeps
from deep_research_pydantic.orchestrator import orchestrate_research
from deep_research_pydantic.runner import (
    PydanticResearchPipeline,
    build_research_pipeline,
)
from deep_research_pydantic.state import ResearchState, create_initial_state

__all__ = [
    "AgentBundle",
    "NodeDeps",
    "PydanticResearchPipeline",
    "ResearchState",
    "build_agents",
    "build_model",
    "build_research_pipeline",
    "create_initial_state",
    "orchestrate_research",
]
