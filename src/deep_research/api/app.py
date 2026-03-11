"""FastAPI application factory."""

from __future__ import annotations

import asyncio
import json
import logging
import uuid
from typing import Any, Literal

from fastapi import FastAPI, HTTPException, Request
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field
from starlette.responses import StreamingResponse

from deep_research.api.jobs import JobManager, JobStatus
from deep_research.session import SessionManager
from deep_research.state import create_initial_state

logger = logging.getLogger(__name__)


class ResearchRequest(BaseModel):
    query: str = Field(..., min_length=1)
    tools: list[str] = Field(default_factory=list)
    output_mode: Literal["chat", "report"] = "chat"
    response_mode: Literal["quick", "research"] = "quick"
    session_id: str | None = None


class FeedbackRequest(BaseModel):
    query_id: str
    rating: Literal["thumbs_up", "thumbs_down"]
    comment: str = ""


async def _run_graph(
    graph: Any,
    job_manager: JobManager,
    job_id: str,
    initial_state: dict,
    session_manager: SessionManager | None = None,
    session_id: str | None = None,
) -> None:
    """Execute the research graph in the background, updating job state."""
    try:
        job_manager.update_state(job_id, "running")

        # Inject job context so nodes (e.g. synthesizer) can push token events
        initial_state["_job_manager"] = job_manager
        initial_state["_job_id"] = job_id

        # Try streaming for node-level progress tracking
        if hasattr(graph, "astream"):
            accumulated_state: dict = {}
            async for event in graph.astream(
                initial_state,
                stream_mode="updates",
                config={"recursion_limit": 50},
            ):
                for node_name in event:
                    logger.info(f"Job {job_id}: completed node '{node_name}'")
                    job_manager.update_state(
                        job_id, "running", current_node=node_name
                    )
                    job_manager.push_event(job_id, {"type": "node_started", "node": node_name})
                    # Merge each node's partial update into accumulated state
                    node_output = event[node_name]
                    if isinstance(node_output, dict):
                        accumulated_state.update(node_output)

            final_output = accumulated_state.get("final_output", "")
            if not final_output:
                final_output = "Research completed but produced no output."
            final_state = accumulated_state
        else:
            result = await graph.ainvoke(initial_state)
            final_output = result.get("final_output", "")
            if not final_output:
                final_output = "Research completed but produced no output."
            final_state = result

        job_manager.update_state(job_id, "completed", result=final_output)
        job_manager.push_event(job_id, {"type": "completed", "result": final_output})

        # Accumulate results into session
        if session_manager and session_id:
            session_manager.add_turn(session_id, "assistant", final_output)
            evidence = final_state.get("evidence", [])
            if evidence:
                session_manager.add_evidence(session_id, evidence)
            tool_calls = final_state.get("tool_call_log", [])
            if tool_calls:
                session_manager.add_tool_calls(session_id, tool_calls)

    except Exception as exc:
        logger.exception(f"Graph execution failed for job {job_id}")
        job_manager.update_state(
            job_id, "failed", error=f"Research execution failed: {exc}"
        )
        job_manager.push_event(job_id, {"type": "failed", "error": f"Research execution failed: {exc}"})


async def _run_quick_reply(
    model: Any,
    mcp_manager: Any,
    job_manager: JobManager,
    job_id: str,
    query: str,
    tools: list[str],
    session_manager: SessionManager,
    session_id: str,
    conversation_history: list[dict[str, str]],
) -> None:
    """Execute a quick reply: optional tool calls + single LLM response."""
    from deep_research.prompts import QUICK_REPLY_SYSTEM

    try:
        job_manager.update_state(job_id, "running", current_node="quick_reply")
        job_manager.push_event(job_id, {"type": "node_started", "node": "quick_reply"})

        # 1. Call each selected tool in parallel (one query per tool)
        tool_results: list[dict[str, str]] = []
        if tools and mcp_manager:
            async def _call_one(tool_name: str) -> dict[str, str]:
                try:
                    result = await mcp_manager.call_tool(
                        tool_name, "query", {"query": query}
                    )
                    return {"tool": tool_name, "result": result.get("result", "")}
                except Exception as e:
                    logger.warning(f"Quick reply tool call failed: {tool_name} - {e}")
                    return {"tool": tool_name, "result": f"(tool call failed: {e})"}

            results = await asyncio.gather(
                *[_call_one(t) for t in tools],
                return_exceptions=True,
            )
            for r in results:
                if isinstance(r, dict):
                    tool_results.append(r)

        # 2. Build tool context
        tool_context_parts = []
        if tool_results:
            tool_context_parts.append("Tool results:")
            for tr in tool_results:
                tool_context_parts.append(f"\n[{tr['tool']}]:\n{tr['result']}")
        tool_context = "\n".join(tool_context_parts) if tool_context_parts else "No tools were queried."

        # 3. Build conversation context
        conv_parts = []
        if conversation_history:
            conv_parts.append("Previous conversation:")
            for turn in conversation_history[-6:]:  # Last 3 exchanges
                role = turn.get("role", "user")
                content = turn.get("content", "")
                conv_parts.append(f"{role}: {content[:500]}")
        conversation_context = "\n".join(conv_parts) if conv_parts else ""

        # 4. Single LLM call
        system_prompt = QUICK_REPLY_SYSTEM.format(
            tool_context=tool_context,
            conversation_context=conversation_context,
        )
        response = await model.ainvoke([
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": query},
        ])
        reply = response.content if hasattr(response, "content") else str(response)

        # 5. Complete job
        job_manager.update_state(job_id, "completed", result=reply)
        job_manager.push_event(job_id, {"type": "completed", "result": reply})

        # 6. Update session
        session_manager.add_turn(session_id, "assistant", reply)

    except Exception as exc:
        logger.exception(f"Quick reply failed for job {job_id}")
        job_manager.update_state(
            job_id, "failed", error=f"Quick reply failed: {exc}"
        )
        job_manager.push_event(job_id, {"type": "failed", "error": f"Quick reply failed: {exc}"})


def create_app(use_mocks: bool = False) -> FastAPI:
    app = FastAPI(title="Deep Research Agent", version="0.1.0")

    app.add_middleware(
        CORSMiddleware,
        allow_origins=["*"],
        allow_methods=["*"],
        allow_headers=["*"],
    )

    job_manager = JobManager()
    session_manager = SessionManager()

    @app.get("/health")
    async def health():
        return {"status": "ok"}

    @app.get("/api/tools")
    async def list_tools(request: Request):
        if use_mocks:
            return [
                {"name": "genie_aus_market", "display_name": "Australian Economic & Market Data (Genie)", "risk_tier": "safe"},
                {"name": "knowledge_assistant", "display_name": "ASX FY2024 Annual Reports (Knowledge Assistant)", "risk_tier": "safe"},
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

        # Get or create session — capture prior history BEFORE adding current turn
        session = session_manager.get_or_create(req.session_id)
        session_created = session.session_id != req.session_id
        prior_history = list(session.conversation_history)
        prior_evidence = list(session.accumulated_evidence)

        # Record user turn
        session_manager.add_turn(session.session_id, "user", req.query)

        if req.response_mode == "quick":
            # Quick reply path: direct LLM call with optional tool calls
            model = getattr(request.app.state, "model", None)
            if model is None:
                job_manager.update_state(
                    job_id, "failed", error="LLM model not initialized"
                )
                return {"job_id": job_id, "status": "failed", "session_id": session.session_id}

            mcp_manager = getattr(request.app.state, "mcp_manager", None)
            asyncio.create_task(
                _run_quick_reply(
                    model=model,
                    mcp_manager=mcp_manager,
                    job_manager=job_manager,
                    job_id=job_id,
                    query=req.query,
                    tools=req.tools,
                    session_manager=session_manager,
                    session_id=session.session_id,
                    conversation_history=prior_history,
                )
            )
        else:
            # Deep research path: full graph pipeline
            graph = getattr(request.app.state, "graph", None)
            if graph is None:
                job_manager.update_state(
                    job_id, "failed", error="Research graph not initialized"
                )
                return {"job_id": job_id, "status": "failed", "session_id": session.session_id}

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
                conversation_history=prior_history,
                prior_evidence=prior_evidence,
            )

            asyncio.create_task(
                _run_graph(
                    graph, job_manager, job_id, initial_state,
                    session_manager=session_manager,
                    session_id=session.session_id,
                )
            )

        return {
            "job_id": job_id,
            "status": "pending",
            "session_id": session.session_id,
            "session_created": session_created,
        }

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

    @app.get("/api/research/{job_id}/stream")
    async def stream_research(job_id: str):
        try:
            job_manager.get_status(job_id)
        except KeyError:
            raise HTTPException(status_code=404, detail="Job not found")

        async def event_generator():
            # Check terminal state FIRST (handles late-connect case)
            status = job_manager.get_status(job_id)
            if status.state in ("completed", "failed", "cancelled"):
                yield f"data: {json.dumps({'type': status.state, 'result': status.result, 'error': status.error})}\n\n"
                return

            queue = job_manager.get_event_queue(job_id)
            while True:
                try:
                    event = await asyncio.wait_for(queue.get(), timeout=30.0)
                    yield f"data: {json.dumps(event)}\n\n"
                    if event.get("type") in ("completed", "failed", "cancelled"):
                        break
                except asyncio.TimeoutError:
                    # Send keepalive
                    yield ": keepalive\n\n"
                    # Check if job terminated externally
                    status = job_manager.get_status(job_id)
                    if status.state in ("completed", "failed", "cancelled"):
                        yield f"data: {json.dumps({'type': status.state, 'result': status.result, 'error': status.error})}\n\n"
                        break

        return StreamingResponse(
            event_generator(),
            media_type="text/event-stream",
            headers={
                "X-Accel-Buffering": "no",
                "Cache-Control": "no-cache",
            },
        )

    @app.delete("/api/research/{job_id}")
    async def cancel_research(job_id: str):
        try:
            job_manager.cancel_job(job_id)
        except KeyError:
            raise HTTPException(status_code=404, detail="Job not found")
        return {"job_id": job_id, "status": "cancelled"}

    @app.get("/api/sessions/{session_id}")
    async def get_session(session_id: str):
        session = session_manager.get_session(session_id)
        if session is None:
            raise HTTPException(status_code=404, detail="Session not found or expired")
        return session.to_dict()

    @app.delete("/api/sessions/{session_id}")
    async def delete_session(session_id: str):
        deleted = session_manager.delete_session(session_id)
        if not deleted:
            raise HTTPException(status_code=404, detail="Session not found")
        return {"status": "deleted", "session_id": session_id}

    @app.post("/api/feedback")
    async def submit_feedback(req: FeedbackRequest):
        # TODO: Log to MLflow via feedback module
        return {"status": "ok", "query_id": req.query_id, "rating": req.rating}

    return app
