"""FastAPI application factory."""

from __future__ import annotations

from typing import Literal

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field

from deep_research.api.jobs import JobManager, JobStatus


class ResearchRequest(BaseModel):
    query: str = Field(..., min_length=1)
    tools: list[str] = Field(default_factory=list)
    output_mode: Literal["chat", "report"] = "chat"


class FeedbackRequest(BaseModel):
    query_id: str
    rating: Literal["thumbs_up", "thumbs_down"]
    comment: str = ""


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
    async def list_tools():
        # TODO: Return actual MCP server tool list from manager
        if use_mocks:
            return [
                {"name": "genie_sales", "display_name": "Sales Data (Genie)", "risk_tier": "safe"},
                {"name": "vector_search_kb", "display_name": "Knowledge Base", "risk_tier": "safe"},
            ]
        return []

    @app.post("/api/research")
    async def submit_research(req: ResearchRequest):
        job_id = job_manager.create_job(
            query=req.query, tools=req.tools, output_mode=req.output_mode
        )
        status = job_manager.get_status(job_id)
        return {"job_id": job_id, "status": status.state}

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
