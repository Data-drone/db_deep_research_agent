"""FastAPI application factory for the Pydantic AI pipeline."""

from __future__ import annotations

import asyncio
import json
import logging
import uuid
from typing import Any, Literal

from fastapi import FastAPI, HTTPException, Request
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field, ValidationError, field_validator
from pydantic_ai import Agent
from pydantic_ai.exceptions import ModelRetry, UnexpectedModelBehavior
from starlette.responses import StreamingResponse

from deep_research.api.clarification import InMemoryClarificationStore
from deep_research.api.jobs import JobManager, JobStatus
from deep_research.session import SessionManager
from deep_research_pydantic.schemas import ClarifierOutput
from deep_research_pydantic.state import create_initial_state

logger = logging.getLogger(__name__)

_STRUCTURED_OUTPUT_ERRORS = (
    UnexpectedModelBehavior,
    ModelRetry,
    ValidationError,
)


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


class ClarifyRequest(BaseModel):
    clarification_id: str = Field(..., min_length=1)
    answer: str = Field(..., min_length=1, max_length=500)

    @field_validator("answer")
    @classmethod
    def strip_answer(cls, value: str) -> str:
        value = value.strip()
        if not value:
            raise ValueError("Answer must not be blank")
        return value


async def _run_graph(
    graph: Any,
    job_manager: JobManager,
    job_id: str,
    initial_state: dict[str, Any],
    session_manager: SessionManager | None = None,
    session_id: str | None = None,
) -> None:
    """Execute the research pipeline in the background."""

    from deep_research.tracing import trace_research

    query = initial_state.get("user_query", "")
    tools = initial_state.get("selected_tools", [])
    output_mode = initial_state.get("output_mode", "chat")

    async with trace_research(
        job_id,
        query,
        tools,
        output_mode,
        mode="research",
    ) as trace_ctx:
        try:
            job_manager.update_state(job_id, "running")
            initial_state["_job_manager"] = job_manager
            initial_state["_job_id"] = job_id

            if hasattr(graph, "astream"):
                accumulated_state: dict[str, Any] = {}
                async for event in graph.astream(
                    initial_state,
                    {"recursion_limit": 50},
                    stream_mode="updates",
                ):
                    for node_name, node_output in event.items():
                        logger.info(
                            "Job %s: completed node '%s'",
                            job_id,
                            node_name,
                        )
                        job_manager.update_state(
                            job_id,
                            "running",
                            current_node=node_name,
                        )
                        job_manager.push_event(
                            job_id,
                            {"type": "node_started", "node": node_name},
                        )
                        if isinstance(node_output, dict):
                            accumulated_state.update(node_output)
                final_state = accumulated_state
            else:
                final_state = await graph.ainvoke(
                    initial_state,
                    config={"recursion_limit": 50},
                )

            final_output = final_state.get("final_output", "")
            if not final_output:
                final_output = "Research completed but produced no output."

            token_usage = final_state.get("_token_usage")
            job_manager.update_state(
                job_id,
                "completed",
                result=final_output,
            )
            completed_event: dict[str, Any] = {
                "type": "completed",
                "result": final_output,
            }
            if token_usage and (
                token_usage.get("input", 0) > 0
                or token_usage.get("output", 0) > 0
            ):
                completed_event["token_usage"] = {
                    **token_usage,
                    "scope": "answer",
                }
                status: JobStatus = job_manager.get_status(job_id)
                status.token_usage = token_usage

            job_manager.push_event(job_id, completed_event)
            trace_ctx["output"] = final_output
            trace_ctx["status"] = "completed"

            if session_manager and session_id:
                session_manager.add_turn(
                    session_id,
                    "assistant",
                    final_output,
                )
                evidence = final_state.get("evidence", [])
                if evidence:
                    session_manager.add_evidence(session_id, evidence)
                tool_calls = final_state.get("tool_call_log", [])
                if tool_calls:
                    session_manager.add_tool_calls(session_id, tool_calls)
        except Exception as exc:
            logger.exception("Graph execution failed for job %s", job_id)
            error = f"Research execution failed: {exc}"
            job_manager.update_state(job_id, "failed", error=error)
            job_manager.push_event(
                job_id,
                {"type": "failed", "error": error},
            )
            trace_ctx["status"] = "failed"
            trace_ctx["output"] = str(exc)


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
    """Execute a quick reply with optional parallel tool calls."""

    from deep_research.prompts import QUICK_REPLY_SYSTEM
    from deep_research.tracing import trace_research

    async with trace_research(
        job_id,
        query,
        tools,
        "chat",
        mode="quick",
    ) as trace_ctx:
        try:
            job_manager.update_state(
                job_id,
                "running",
                current_node="quick_reply",
            )
            job_manager.push_event(
                job_id,
                {"type": "node_started", "node": "quick_reply"},
            )

            async def call_one(tool_name: str) -> dict[str, str]:
                try:
                    result = await mcp_manager.call_tool(
                        tool_name,
                        "query",
                        {"query": query},
                    )
                    return {
                        "tool": tool_name,
                        "result": result.get("result", ""),
                    }
                except Exception as exc:
                    logger.warning(
                        "Quick reply tool call failed: %s - %s",
                        tool_name,
                        exc,
                    )
                    return {
                        "tool": tool_name,
                        "result": f"(tool call failed: {exc})",
                    }

            tool_results: list[dict[str, str]] = []
            if tools and mcp_manager:
                results = await asyncio.gather(
                    *[call_one(tool_name) for tool_name in tools],
                    return_exceptions=True,
                )
                tool_results = [
                    result
                    for result in results
                    if isinstance(result, dict)
                ]

            if tool_results:
                tool_context_parts = ["Tool results:"]
                for tool_result in tool_results:
                    tool_context_parts.append(
                        f"\n[{tool_result['tool']}]:\n"
                        f"{tool_result['result']}"
                    )
                tool_context = "\n".join(tool_context_parts)
            else:
                tool_context = "No tools were queried."

            conversation_parts: list[str] = []
            if conversation_history:
                conversation_parts.append("Previous conversation:")
                for turn in conversation_history[-6:]:
                    role = turn.get("role", "user")
                    content = turn.get("content", "")
                    conversation_parts.append(
                        f"{role}: {content[:500]}"
                    )

            instructions = QUICK_REPLY_SYSTEM.format(
                tool_context=tool_context,
                conversation_context="\n".join(conversation_parts),
            )
            response = await model.run(
                query,
                instructions=instructions,
            )
            reply = response.output
            token_usage = _extract_token_usage(response)

            job_manager.update_state(
                job_id,
                "completed",
                result=reply,
            )
            completed_event: dict[str, Any] = {
                "type": "completed",
                "result": reply,
            }
            if token_usage and (
                token_usage["input"] > 0
                or token_usage["output"] > 0
            ):
                completed_event["token_usage"] = {
                    **token_usage,
                    "scope": "answer",
                }
            job_manager.push_event(job_id, completed_event)

            trace_ctx["output"] = reply
            trace_ctx["status"] = "completed"
            session_manager.add_turn(session_id, "assistant", reply)
        except Exception as exc:
            logger.exception("Quick reply failed for job %s", job_id)
            error = f"Quick reply failed: {exc}"
            job_manager.update_state(job_id, "failed", error=error)
            job_manager.push_event(
                job_id,
                {"type": "failed", "error": error},
            )
            trace_ctx["status"] = "failed"
            trace_ctx["output"] = str(exc)


def _extract_token_usage(response: Any) -> dict[str, int]:
    """Extract token usage from a Pydantic AI or legacy response."""

    run_usage = getattr(response, "usage", None)
    if run_usage is not None and not callable(run_usage):
        input_tokens = getattr(run_usage, "input_tokens", None)
        output_tokens = getattr(run_usage, "output_tokens", None)
        if input_tokens is not None or output_tokens is not None:
            return {
                "input": int(input_tokens or 0),
                "output": int(output_tokens or 0),
            }

    meta = getattr(response, "response_metadata", {}) or {}
    usage = (
        meta.get("usage")
        or meta.get("token_usage")
        or getattr(response, "usage_metadata", None)
        or {}
    )
    return {
        "input": int(
            usage.get(
                "input_tokens",
                usage.get("prompt_tokens", 0),
            )
            or 0
        ),
        "output": int(
            usage.get(
                "output_tokens",
                usage.get("completion_tokens", 0),
            )
            or 0
        ),
    }


async def _run_research_with_clarification(
    model: Any,
    graph: Any,
    clarification_store: InMemoryClarificationStore,
    job_manager: JobManager,
    job_id: str,
    initial_state: dict[str, Any],
    session_manager: SessionManager,
    session_id: str,
) -> None:
    """Run the pre-pipeline clarifier and then the research pipeline."""

    try:
        job_manager.update_state(
            job_id,
            "running",
            current_node="clarifier",
        )
        job_manager.push_event(
            job_id,
            {"type": "node_started", "node": "clarifier"},
        )

        try:
            if model is None:
                raise RuntimeError("Clarifier model not initialized")
            clarifier_response = await model.run(
                initial_state["user_query"],
            )
            clarifier_output: ClarifierOutput = clarifier_response.output
        except _STRUCTURED_OUTPUT_ERRORS:
            logger.warning(
                "Unusable clarifier output for job %s; using original query",
                job_id,
                exc_info=True,
            )
            clarifier_output = ClarifierOutput(
                needs_clarification=False,
                clarified_query=initial_state["user_query"],
            )

        if clarifier_output.needs_clarification:
            question = (
                clarifier_output.question
                or "Could you clarify your question?"
            )
            options = clarifier_output.options
            best_guess = (
                clarifier_output.best_guess
                or initial_state["user_query"]
            )

            clarification_state = clarification_store.create(
                job_id,
                question,
                options,
                best_guess,
            )
            job_manager.push_event(
                job_id,
                {
                    "type": "clarification_needed",
                    "clarification_id": (
                        clarification_state.clarification_id
                    ),
                    "question": question,
                    "options": options,
                },
            )

            result_state = await clarification_store.wait_for_resolution(
                job_id,
                timeout=30.0,
            )

            if (
                result_state
                and result_state.status == "answered"
                and result_state.answer
            ):
                initial_state["clarified_query"] = result_state.answer
                job_manager.push_event(
                    job_id,
                    {
                        "type": "clarification_resolved",
                        "answer": result_state.answer,
                    },
                )
            elif (
                result_state
                and result_state.status == "cancelled"
            ):
                clarification_store.cleanup(job_id)
                return
            elif (
                result_state
                and result_state.status == "timed_out"
            ):
                initial_state["clarified_query"] = best_guess
                job_manager.push_event(
                    job_id,
                    {
                        "type": "clarification_timeout",
                        "best_guess": best_guess,
                    },
                )
            else:
                initial_state["clarified_query"] = best_guess
        else:
            initial_state["clarified_query"] = (
                clarifier_output.clarified_query
                or initial_state["user_query"]
            )

        initial_state["needs_clarification"] = False
        await _run_graph(
            graph,
            job_manager,
            job_id,
            initial_state,
            session_manager=session_manager,
            session_id=session_id,
        )
        clarification_store.cleanup(job_id)
    except Exception as exc:
        logger.exception(
            "Research with clarification failed for job %s",
            job_id,
        )
        error = f"Research execution failed: {exc}"
        job_manager.update_state(job_id, "failed", error=error)
        job_manager.push_event(
            job_id,
            {"type": "failed", "error": error},
        )
        clarification_store.cleanup(job_id)


def _make_quick_reply_agent(model: Any) -> Agent:
    return Agent(
        model,
        output_type=str,
        name="quick_reply",
    )


def _make_clarifier_agent(model: Any) -> Agent:
    from deep_research.prompts import CLARIFIER_SYSTEM

    return Agent(
        model,
        output_type=ClarifierOutput,
        instructions=CLARIFIER_SYSTEM,
        name="pre_graph_clarifier",
    )


def create_app(use_mocks: bool = False) -> FastAPI:
    app = FastAPI(
        title="Deep Research Agent",
        version="0.1.0",
    )
    app.add_middleware(
        CORSMiddleware,
        allow_origins=["*"],
        allow_methods=["*"],
        allow_headers=["*"],
    )

    job_manager = JobManager()
    session_manager = SessionManager()
    clarification_store = InMemoryClarificationStore()

    @app.get("/health")
    async def health():
        return {"status": "ok"}

    @app.get("/api/tools")
    async def list_tools(request: Request):
        if use_mocks:
            return [
                {
                    "name": "genie_aus_market",
                    "display_name": (
                        "Australian Economic & Market Data (Genie)"
                    ),
                    "risk_tier": "safe",
                },
                {
                    "name": "knowledge_assistant",
                    "display_name": (
                        "ASX FY2024 Annual Reports "
                        "(Knowledge Assistant)"
                    ),
                    "risk_tier": "safe",
                },
            ]

        mcp_manager = getattr(request.app.state, "mcp_manager", None)
        if mcp_manager is None:
            return []
        return [
            {
                "name": config.name,
                "display_name": config.display_name,
                "risk_tier": config.risk_tier,
            }
            for config in mcp_manager.get_available_servers().values()
        ]

    @app.post("/api/research")
    async def submit_research(
        req: ResearchRequest,
        request: Request,
    ):
        job_id = job_manager.create_job(
            query=req.query,
            tools=req.tools,
            output_mode=req.output_mode,
        )

        session = session_manager.get_or_create(req.session_id)
        session_created = session.session_id != req.session_id
        prior_history = list(session.conversation_history)
        prior_evidence = list(session.accumulated_evidence)
        session_manager.add_turn(
            session.session_id,
            "user",
            req.query,
        )

        if req.response_mode == "quick":
            quick_reply_agent = getattr(
                request.app.state,
                "quick_reply_agent",
                None,
            )
            if quick_reply_agent is None:
                raw_model = getattr(request.app.state, "model", None)
                if raw_model is not None:
                    quick_reply_agent = _make_quick_reply_agent(raw_model)

            if quick_reply_agent is None:
                job_manager.update_state(
                    job_id,
                    "failed",
                    error="LLM model not initialized",
                )
                return {
                    "job_id": job_id,
                    "status": "failed",
                    "session_id": session.session_id,
                }

            asyncio.create_task(
                _run_quick_reply(
                    model=quick_reply_agent,
                    mcp_manager=getattr(
                        request.app.state,
                        "mcp_manager",
                        None,
                    ),
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
            graph = getattr(request.app.state, "graph", None)
            if graph is None:
                job_manager.update_state(
                    job_id,
                    "failed",
                    error="Research graph not initialized",
                )
                return {
                    "job_id": job_id,
                    "status": "failed",
                    "session_id": session.session_id,
                }

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

            clarifier_agent = getattr(
                request.app.state,
                "clarifier_agent",
                None,
            )
            if clarifier_agent is None:
                raw_model = getattr(request.app.state, "model", None)
                if raw_model is not None:
                    clarifier_agent = _make_clarifier_agent(raw_model)

            asyncio.create_task(
                _run_research_with_clarification(
                    model=clarifier_agent,
                    graph=graph,
                    clarification_store=clarification_store,
                    job_manager=job_manager,
                    job_id=job_id,
                    initial_state=initial_state,
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

        response = {
            "job_id": job_id,
            "status": status.state,
            "result": status.result,
            "current_node": status.current_node,
            "error": status.error,
        }
        clarification_state = clarification_store.get(job_id)
        if (
            clarification_state
            and clarification_state.status == "pending"
        ):
            response["pending_clarification"] = {
                "clarification_id": (
                    clarification_state.clarification_id
                ),
                "question": clarification_state.question,
                "options": clarification_state.options,
            }
        return response

    @app.get("/api/research/{job_id}/stream")
    async def stream_research(job_id: str):
        try:
            job_manager.get_status(job_id)
        except KeyError:
            raise HTTPException(status_code=404, detail="Job not found")

        async def event_generator():
            status = job_manager.get_status(job_id)
            if status.state in ("completed", "failed", "cancelled"):
                event = {
                    "type": status.state,
                    "result": status.result,
                    "error": status.error,
                }
                yield f"data: {json.dumps(event)}\n\n"
                return

            queue = job_manager.get_event_queue(job_id)
            while True:
                try:
                    event = await asyncio.wait_for(
                        queue.get(),
                        timeout=30.0,
                    )
                    yield f"data: {json.dumps(event)}\n\n"
                    if event.get("type") in (
                        "completed",
                        "failed",
                        "cancelled",
                    ):
                        break
                except asyncio.TimeoutError:
                    yield ": keepalive\n\n"
                    status = job_manager.get_status(job_id)
                    if status.state in (
                        "completed",
                        "failed",
                        "cancelled",
                    ):
                        event = {
                            "type": status.state,
                            "result": status.result,
                            "error": status.error,
                        }
                        yield f"data: {json.dumps(event)}\n\n"
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
        clarification_store.mark_cancelled(job_id)
        return {"job_id": job_id, "status": "cancelled"}

    @app.post("/api/research/{job_id}/clarify")
    async def submit_clarification(
        job_id: str,
        req: ClarifyRequest,
    ):
        try:
            job_manager.get_status(job_id)
        except KeyError:
            raise HTTPException(status_code=404, detail="Job not found")

        clarification_state = clarification_store.get(job_id)
        if clarification_state is None:
            raise HTTPException(
                status_code=400,
                detail="No pending clarification for this job",
            )
        if (
            clarification_state.clarification_id
            != req.clarification_id
        ):
            raise HTTPException(
                status_code=409,
                detail="Clarification ID mismatch",
            )
        if clarification_state.status == "answered":
            return {
                "status": "already_answered",
                "used_answer": False,
            }
        if clarification_state.status == "timed_out":
            return {"status": "expired", "used_answer": False}
        if clarification_state.status == "cancelled":
            return {"status": "cancelled", "used_answer": False}

        result = clarification_store.submit_answer(
            job_id,
            req.clarification_id,
            req.answer,
        )
        if result is None:
            return {"status": "expired", "used_answer": False}
        return {"status": "accepted", "job_id": job_id}

    @app.get("/api/sessions/{session_id}")
    async def get_session(session_id: str):
        session = session_manager.get_session(session_id)
        if session is None:
            raise HTTPException(
                status_code=404,
                detail="Session not found or expired",
            )
        return session.to_dict()

    @app.delete("/api/sessions/{session_id}")
    async def delete_session(session_id: str):
        deleted = session_manager.delete_session(session_id)
        if not deleted:
            raise HTTPException(
                status_code=404,
                detail="Session not found",
            )
        return {"status": "deleted", "session_id": session_id}

    @app.post("/api/feedback")
    async def submit_feedback(req: FeedbackRequest):
        return {
            "status": "ok",
            "query_id": req.query_id,
            "rating": req.rating,
        }

    return app
