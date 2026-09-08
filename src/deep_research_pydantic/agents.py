"""Pydantic AI agent and Databricks model factories."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import httpx
from pydantic_ai import Agent
from pydantic_ai.models.openai import OpenAIChatModel
from pydantic_ai.providers.openai import OpenAIProvider

from deep_research_pydantic.schemas import (
    ClarifierOutput,
    CompressorOutput,
    EvaluatorOutput,
    PerspectiveOutput,
    PlannerOutput,
    ScorerOutput,
    VerifierOutput,
)


@dataclass(frozen=True)
class AgentBundle:
    """All per-node Pydantic AI agents."""

    clarifier: Any
    perspective: Any
    planner: Any
    query_adapter: Any
    scorer: Any
    evaluator: Any
    compressor: Any
    synthesizer: Any
    verifier: Any


def build_model(
    host: str,
    endpoint: str,
    token: str | None = None,
    http_client: httpx.AsyncClient | None = None,
) -> OpenAIChatModel:
    """Build an OpenAI-compatible Databricks serving-endpoint model."""

    if token is None and http_client is None:
        raise ValueError("Either token or http_client must be provided")

    provider = OpenAIProvider(
        base_url=f"{host.rstrip('/')}/serving-endpoints",
        api_key=token or "sdk-auth-managed",
        http_client=http_client,
    )
    return OpenAIChatModel(endpoint, provider=provider)


def build_agents(
    worker_model: Any,
    critic_model: Any | None = None,
) -> AgentBundle:
    """Build node agents, falling back to the worker for critic tasks.

    Node instructions are supplied per invocation. Constructor instructions are
    intentionally omitted because Pydantic AI concatenates constructor and
    per-run instructions.
    """

    critic = critic_model or worker_model

    return AgentBundle(
        clarifier=Agent(
            worker_model,
            output_type=ClarifierOutput,
            name="research_clarifier",
        ),
        perspective=Agent(
            critic,
            output_type=PerspectiveOutput,
            name="research_perspective_generator",
        ),
        planner=Agent(
            worker_model,
            output_type=PlannerOutput,
            name="research_planner",
        ),
        query_adapter=Agent(
            worker_model,
            output_type=str,
            name="research_query_adapter",
        ),
        scorer=Agent(
            critic,
            output_type=ScorerOutput,
            name="research_evidence_scorer",
        ),
        evaluator=Agent(
            critic,
            output_type=EvaluatorOutput,
            name="research_evaluator",
        ),
        compressor=Agent(
            worker_model,
            output_type=CompressorOutput,
            name="research_compressor",
        ),
        synthesizer=Agent(
            worker_model,
            output_type=str,
            name="research_synthesizer",
        ),
        verifier=Agent(
            critic,
            output_type=VerifierOutput,
            name="research_verifier",
        ),
    )
