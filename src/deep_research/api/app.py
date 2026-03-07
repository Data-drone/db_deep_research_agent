"""FastAPI application factory."""

from __future__ import annotations

import asyncio
import logging
import uuid
from typing import Any, Literal

from fastapi import FastAPI, HTTPException, Request
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field

from deep_research.api.jobs import JobManager, JobStatus
from deep_research.state import create_initial_state

logger = logging.getLogger(__name__)


class ResearchRequest(BaseModel):
    query: str = Field(..., min_length=1)
    tools: list[str] = Field(default_factory=list)
    output_mode: Literal["chat", "report"] = "chat"


class FeedbackRequest(BaseModel):
    query_id: str
    rating: Literal["thumbs_up", "thumbs_down"]
    comment: str = ""


async def _run_graph(
    graph: Any,
    job_manager: JobManager,
    job_id: str,
    initial_state: dict,
) -> None:
    """Execute the research graph in the background, updating job state."""
    try:
        job_manager.update_state(job_id, "running")

        # Try streaming for node-level progress tracking
        if hasattr(graph, "astream"):
            final_state = None
            async for event in graph.astream(initial_state, stream_mode="updates"):
                for node_name in event:
                    logger.info(f"Job {job_id}: completed node '{node_name}'")
                    job_manager.update_state(
                        job_id, "running", current_node=node_name
                    )
                    final_state = event[node_name]

            final_output = ""
            if final_state and isinstance(final_state, dict):
                final_output = final_state.get("final_output", "")
            if not final_output:
                final_output = "Research completed but produced no output."
        else:
            result = await graph.ainvoke(initial_state)
            final_output = result.get("final_output", "")
            if not final_output:
                final_output = "Research completed but produced no output."

        job_manager.update_state(job_id, "completed", result=final_output)
    except Exception as exc:
        logger.exception(f"Graph execution failed for job {job_id}")
        job_manager.update_state(
            job_id, "failed", error=f"Research execution failed: {exc}"
        )


def create_app(use_mocks: bool = False) -> FastAPI:
    app = FastAPI(title="Deep Research Agent", version="0.1.0")

    app.add_middleware(
        CORSMiddleware,
        allow_origins=["*"],
        allow_methods=["*"],
        allow_headers=["*"],
    )

    job_manager = JobManager()

    @app.get("/health")
    async def health():
        return {"status": "ok"}

    @app.get("/api/tools")
    async def list_tools(request: Request):
        if use_mocks:
            return [
                {"name": "genie_sales", "display_name": "Sales Data (Genie)", "risk_tier": "safe"},
                {"name": "vector_search_kb", "display_name": "Knowledge Base", "risk_tier": "safe"},
            ]
        mcp_manager = getattr(request.app.state, "mcp_manager", None)
        if mcp_manager is None:
            return []
        return [
            {
                "name": cfg.name,
                "display_name": cfg.display_name,
                "risk_tier": cfg.risk_tier,
            }
            for cfg in mcp_manager.get_available_servers().values()
        ]

    @app.post("/api/research")
    async def submit_research(req: ResearchRequest, request: Request):
        job_id = job_manager.create_job(
            query=req.query, tools=req.tools, output_mode=req.output_mode
        )

        graph = getattr(request.app.state, "graph", None)
        if graph is None:
            job_manager.update_state(
                job_id, "failed", error="Research graph not initialized"
            )
            return {"job_id": job_id, "status": "failed"}

        config = getattr(request.app.state, "config", None)
        budget = None
        if config:
            from deep_research.models import Budget
            budget = Budget(
                max_iterations=config.max_iterations,
                max_tool_calls=config.max_tool_calls,
                time_cap_seconds=config.time_cap_seconds,
            )

        initial_state = create_initial_state(
            user_query=req.query,
            selected_tools=req.tools,
            output_mode=req.output_mode,
            job_id=job_id,
            trace_id=f"trace-{uuid.uuid4().hex[:12]}",
            budget=budget,
        )

        asyncio.create_task(_run_graph(graph, job_manager, job_id, initial_state))

        return {"job_id": job_id, "status": "pending"}

    @app.get("/api/research/{job_id}")
    async def get_research_status(job_id: str):
        try:
            status = job_manager.get_status(job_id)
        except KeyError:
            raise HTTPException(status_code=404, detail="Job not found")
        return {
            "job_id": job_id,
            "status": status.state,
            "result": status.result,
            "current_node": status.current_node,
            "error": status.error,
        }

    @app.delete("/api/research/{job_id}")
    async def cancel_research(job_id: str):
        try:
            job_manager.cancel_job(job_id)
        except KeyError:
            raise HTTPException(status_code=404, detail="Job not found")
        return {"job_id": job_id, "status": "cancelled"}

    @app.post("/api/feedback")
    async def submit_feedback(req: FeedbackRequest):
        # TODO: Log to MLflow via feedback module
        return {"status": "ok", "query_id": req.query_id, "rating": req.rating}

    return app
