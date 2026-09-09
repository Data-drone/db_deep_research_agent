"""FastAPI application factory."""

from __future__ import annotations

import asyncio
from contextlib import asynccontextmanager, suppress
import json
import logging
import uuid
from typing import Any, Literal

from fastapi import FastAPI, HTTPException, Request
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field, field_validator
from starlette.responses import StreamingResponse

from deep_research.api.clarification import InMemoryClarificationStore
from deep_research.api.jobs import (
    STALL_GRACE_SECONDS,
    TERMINAL_STATES,
    JobManager,
    JobStatus,
)
from deep_research.session import SessionManager
from deep_research.state import create_initial_state
from deep_research.token_usage import add_usage, extract_usage

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


class ClarifyRequest(BaseModel):
    clarification_id: str = Field(..., min_length=1)
    answer: str = Field(..., min_length=1, max_length=500)

    @field_validator("answer")
    @classmethod
    def strip_answer(cls, v: str) -> str:
        v = v.strip()
        if not v:
            raise ValueError("Answer must not be blank")
        return v


async def _run_graph(
    graph: Any,
    job_manager: JobManager,
    job_id: str,
    initial_state: dict,
    session_manager: SessionManager | None = None,
    session_id: str | None = None,
) -> None:
    """Execute the research graph in the background, updating job state."""
    from deep_research.tracing import trace_research

    query = initial_state.get("user_query", "")
    tools = initial_state.get("selected_tools", [])
    output_mode = initial_state.get("output_mode", "chat")

    async with trace_research(job_id, query, tools, output_mode, mode="research") as trace_ctx:
        try:
            if job_manager.is_cancelled(job_id):
                trace_ctx["status"] = "cancelled"
                return
            job_manager.update_state(job_id, "running")

            # Inject job context so nodes (e.g. synthesizer) can push token events
            initial_state["_job_manager"] = job_manager
            initial_state["_job_id"] = job_id

            # LangGraph counts supersteps, and a research iteration is several
            # nodes; a verifier revision replays the whole loop. A fixed 50 was
            # close enough to the worst case at the default budget that raising
            # max_iterations would blow up with GraphRecursionError minutes into
            # a run, so derive it instead.
            recursion_limit = _recursion_limit(initial_state.get("budget"))

            # Try streaming for node-level progress tracking
            if hasattr(graph, "astream"):
                accumulated_state: dict = {}
                async for event in graph.astream(
                    initial_state,
                    stream_mode="updates",
                    config={"recursion_limit": recursion_limit},
                ):
                    if job_manager.is_cancelled(job_id):
                        logger.info(f"Job {job_id}: cancelled mid-graph, stopping")
                        trace_ctx["status"] = "cancelled"
                        return
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

            # Extract token usage if accumulated through graph
            token_usage = final_state.get("_token_usage")

            job_manager.update_state(job_id, "completed", result=final_output)
            completed_event: dict[str, Any] = {"type": "completed", "result": final_output}
            # Surface fabricated citations rather than only logging them — the
            # reader should know the model cited evidence that does not exist.
            unverified = final_state.get("unverified_citations") or []
            if unverified:
                completed_event["unverified_citations"] = unverified
                job_manager.get_status(job_id).unverified_citations = list(unverified)
            # Every LLM-calling node charges into _token_usage, so this covers
            # the whole run. Say "unavailable" rather than omitting the field:
            # a provider that reports no usage is not the same as zero cost.
            if token_usage and (token_usage.get("input", 0) > 0 or token_usage.get("output", 0) > 0):
                completed_event["token_usage"] = {**token_usage, "scope": "job"}
                status = job_manager.get_status(job_id)
                status.token_usage = completed_event["token_usage"]
            else:
                completed_event["token_usage"] = {"input": 0, "output": 0, "scope": "unavailable"}
            job_manager.push_event(job_id, completed_event)
            trace_ctx["output"] = final_output
            trace_ctx["status"] = "completed"

            # Accumulate results into session
            if session_manager and session_id:
                session_manager.add_turn(session_id, "assistant", final_output)
                evidence = final_state.get("evidence", [])
                if evidence:
                    session_manager.add_evidence(session_id, evidence)
                tool_calls = final_state.get("tool_call_log", [])
                if tool_calls:
                    session_manager.add_tool_calls(session_id, tool_calls)

        except asyncio.CancelledError:
            logger.info(f"Job {job_id}: graph execution cancelled")
            job_manager.update_state(job_id, "cancelled")
            job_manager.push_event(job_id, {"type": "cancelled"})
            trace_ctx["status"] = "cancelled"
            raise
        except Exception as exc:
            logger.exception(f"Graph execution failed for job {job_id}")
            job_manager.update_state(
                job_id, "failed", error=f"Research execution failed: {exc}"
            )
            job_manager.push_event(job_id, {"type": "failed", "error": f"Research execution failed: {exc}"})
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
    """Execute a quick reply: optional tool calls + single LLM response."""
    from deep_research.prompts import QUICK_REPLY_SYSTEM
    from deep_research.tracing import trace_research

    async with trace_research(job_id, query, tools, "chat", mode="quick") as trace_ctx:
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
            token_usage = _extract_token_usage(response)

            # 5. Complete job
            job_manager.update_state(job_id, "completed", result=reply)
            completed_event: dict[str, Any] = {"type": "completed", "result": reply}
            if token_usage and (token_usage["input"] > 0 or token_usage["output"] > 0):
                completed_event["token_usage"] = {**token_usage, "scope": "answer"}
            else:
                completed_event["token_usage"] = {"input": 0, "output": 0, "scope": "unavailable"}
            job_manager.push_event(job_id, completed_event)
            trace_ctx["output"] = reply
            trace_ctx["status"] = "completed"

            # 6. Update session
            session_manager.add_turn(session_id, "assistant", reply)

        except asyncio.CancelledError:
            # CancelledError is a BaseException, so the handler below never saw
            # it and the trace was left recording a run that is still in flight.
            logger.info(f"Job {job_id}: quick reply cancelled")
            job_manager.update_state(job_id, "cancelled")
            job_manager.push_event(job_id, {"type": "cancelled"})
            trace_ctx["status"] = "cancelled"
            raise
        except Exception as exc:
            logger.exception(f"Quick reply failed for job {job_id}")
            job_manager.update_state(
                job_id, "failed", error=f"Quick reply failed: {exc}"
            )
            job_manager.push_event(job_id, {"type": "failed", "error": f"Quick reply failed: {exc}"})
            trace_ctx["status"] = "failed"
            trace_ctx["output"] = str(exc)


def _extract_token_usage(response: Any) -> dict[str, int]:
    """Extract token usage from an LLM response's metadata.

    Thin alias kept for the existing call sites and tests; the implementation
    is shared with the graph nodes so the two paths cannot drift apart.
    """
    return extract_usage(response)


def _terminal_event(status: Any) -> dict[str, Any]:
    """Build the SSE frame for a job that is already in a terminal state.

    Used by the late-connect and keepalive paths so a client that reconnects
    after the fact sees the same payload a live client saw, token usage included.
    """
    event: dict[str, Any] = {
        "type": status.state,
        "result": status.result,
        "error": status.error,
    }
    if status.token_usage:
        event["token_usage"] = status.token_usage
    if status.unverified_citations:
        event["unverified_citations"] = status.unverified_citations
    return event


def _recursion_limit(budget: Any) -> int:
    """A superstep budget that scales with the configured research budget.

    Each research iteration is roughly six nodes and a verifier revision replays
    the whole loop, so the cost is (iterations x revisions) rather than additive.
    Padded, and floored at the previous fixed value so this can only ever raise
    the ceiling.
    """
    iterations = int(getattr(budget, "max_iterations", 5) or 5)
    revisions = int(getattr(budget, "max_verification_attempts", 1) or 1)
    return max(50, (iterations * (revisions + 1) * 8) + 20)


#: How long to wait for a human to answer a clarifying question. The old value
#: was 30s, which is shorter than it takes to read three options and type.
DEFAULT_CLARIFICATION_TIMEOUT = 180.0


async def _run_research_with_clarification(
    model: Any,
    graph: Any,
    clarification_store: InMemoryClarificationStore,
    job_manager: JobManager,
    job_id: str,
    initial_state: dict,
    session_manager: SessionManager,
    session_id: str,
    clarification_timeout: float = DEFAULT_CLARIFICATION_TIMEOUT,
) -> None:
    """Run clarifier pre-graph, handle clarification if needed, then run full graph."""
    from deep_research.nodes.clarifier import clarifier_node

    try:
        if job_manager.is_cancelled(job_id):
            return
        job_manager.update_state(job_id, "running", current_node="clarifier")
        job_manager.push_event(job_id, {"type": "node_started", "node": "clarifier"})

        # Run clarifier standalone (pre-graph)
        clarifier_result = await clarifier_node(initial_state, model=model)

        # The clarifier runs outside the graph, so its usage does not reach
        # _token_usage the way every in-graph node's does. Fold it in here or the
        # run's total silently excludes an LLM call the user paid for.
        initial_state["_token_usage"] = add_usage(
            initial_state.get("_token_usage"), clarifier_result.get("_token_usage")
        )

        # The clarifier is a multi-second LLM call; a cancel arriving during it
        # must not go on to raise a clarification prompt for a dead job.
        if job_manager.is_cancelled(job_id):
            job_manager.push_event(job_id, {"type": "cancelled"})
            return

        if clarifier_result.get("needs_clarification"):
            question = clarifier_result.get("clarification_question", "")
            options = clarifier_result.get("clarification_options", [])
            best_guess = clarifier_result.get("clarified_query", initial_state["user_query"])

            # Create persisted clarification state BEFORE emitting SSE
            clr_state = clarification_store.create(job_id, question, options, best_guess)

            # Push event to UI with clarification_id
            job_manager.push_event(job_id, {
                "type": "clarification_needed",
                "clarification_id": clr_state.clarification_id,
                "question": question,
                "options": options,
                "timeout_seconds": clarification_timeout,
            })

            # Wait for user answer using the store's wait method
            result_state = await clarification_store.wait_for_resolution(
                job_id, timeout=clarification_timeout
            )

            if result_state and result_state.status == "answered" and result_state.answer:
                initial_state["clarified_query"] = result_state.answer
                job_manager.push_event(job_id, {
                    "type": "clarification_resolved",
                    "answer": result_state.answer,
                })
            elif result_state and result_state.status == "cancelled":
                # Job was cancelled while waiting for clarification
                clarification_store.cleanup(job_id)
                return  # cancel_research already updated job state
            elif result_state and result_state.status == "timed_out":
                initial_state["clarified_query"] = best_guess
                job_manager.push_event(job_id, {
                    "type": "clarification_timeout",
                    "best_guess": best_guess,
                })
            else:
                initial_state["clarified_query"] = best_guess
        else:
            initial_state["clarified_query"] = clarifier_result.get(
                "clarified_query", initial_state["user_query"]
            )

        # A cancel may have landed while we were waiting or answering
        if job_manager.is_cancelled(job_id):
            clarification_store.cleanup(job_id)
            job_manager.push_event(job_id, {"type": "cancelled"})
            return

        # Ensure graph's clarifier short-circuits
        initial_state["needs_clarification"] = False

        # Run the full graph
        await _run_graph(
            graph, job_manager, job_id, initial_state,
            session_manager=session_manager,
            session_id=session_id,
        )

        # Cleanup clarification state after job completion
        clarification_store.cleanup(job_id)

    except asyncio.CancelledError:
        logger.info(f"Job {job_id}: cancelled during clarification flow")
        job_manager.update_state(job_id, "cancelled")
        job_manager.push_event(job_id, {"type": "cancelled"})
        clarification_store.cleanup(job_id)
        raise
    except Exception as exc:
        logger.exception(f"Research with clarification failed for job {job_id}")
        job_manager.update_state(
            job_id, "failed", error=f"Research execution failed: {exc}"
        )
        job_manager.push_event(job_id, {"type": "failed", "error": f"Research execution failed: {exc}"})
        clarification_store.cleanup(job_id)


def create_app(use_mocks: bool = False) -> FastAPI:
    job_manager = JobManager()
    session_manager = SessionManager()

    @asynccontextmanager
    async def lifespan(app: FastAPI):
        """Run a background reaper so job and session state cannot grow forever."""
        cfg = getattr(app.state, "config", None)
        ttl = int(getattr(cfg, "job_ttl_seconds", 900) or 900) if cfg else 900
        time_cap = int(getattr(cfg, "time_cap_seconds", 600) or 600) if cfg else 600
        stall_after = time_cap + STALL_GRACE_SECONDS
        interval = max(30, min(ttl, 300))

        async def reap() -> None:
            while True:
                await asyncio.sleep(interval)
                try:
                    # Order matters: stall the wedged jobs first so they become
                    # terminal, then let the TTL pass collect whatever is due.
                    job_manager.sweep_stalled(stall_after)
                    job_manager.sweep_terminal(ttl)
                    session_manager.cleanup_expired()
                except Exception:
                    logger.exception("Reaper pass failed")

        reaper = asyncio.create_task(reap())
        app.state.reaper_task = reaper
        try:
            yield
        finally:
            reaper.cancel()
            with suppress(asyncio.CancelledError):
                await reaper

    app = FastAPI(title="Deep Research Agent", version="0.1.0", lifespan=lifespan)

    app.add_middleware(
        CORSMiddleware,
        allow_origins=["*"],
        allow_methods=["*"],
        allow_headers=["*"],
    )

    clarification_store = InMemoryClarificationStore()

    # Both stores are closures over this app instance anyway; publishing them on
    # ``app.state`` is what the rest of the codebase already does for the graph
    # and model, and it lets a test inspect the same objects the handlers use.
    app.state.job_manager = job_manager
    app.state.session_manager = session_manager
    app.state.clarification_store = clarification_store

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
            task = asyncio.create_task(
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
            job_manager.register_task(job_id, task)
        else:
            # Deep research path: full graph pipeline
            graph = getattr(request.app.state, "graph", None)
            if graph is None:
                job_manager.update_state(
                    job_id, "failed", error="Research graph not initialized"
                )
                return {"job_id": job_id, "status": "failed", "session_id": session.session_id}

            # Same fast-fail the quick path gets. Without it the clarifier is the
            # first thing to touch the model, and the user gets a job that fails
            # a few seconds later with 'NoneType has no attribute ainvoke'.
            model = getattr(request.app.state, "model", None)
            if model is None:
                job_manager.update_state(
                    job_id, "failed", error="LLM model not initialized"
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
                    max_verification_attempts=config.max_verification_attempts,
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

            task = asyncio.create_task(
                _run_research_with_clarification(
                    model=model,
                    graph=graph,
                    clarification_store=clarification_store,
                    job_manager=job_manager,
                    job_id=job_id,
                    initial_state=initial_state,
                    session_manager=session_manager,
                    session_id=session.session_id,
                    clarification_timeout=float(
                        getattr(config, "clarification_timeout_seconds", DEFAULT_CLARIFICATION_TIMEOUT)
                        if config else DEFAULT_CLARIFICATION_TIMEOUT
                    ),
                )
            )
            job_manager.register_task(job_id, task)

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
            "token_usage": status.token_usage,
            "unverified_citations": status.unverified_citations,
        }
        clr_state = clarification_store.get(job_id)
        if clr_state and clr_state.status == "pending":
            response["pending_clarification"] = {
                "clarification_id": clr_state.clarification_id,
                "question": clr_state.question,
                "options": clr_state.options,
            }
        return response

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
                yield f"data: {json.dumps(_terminal_event(status))}\n\n"
                return

            # A reconnecting client attaches to a queue whose events were
            # already consumed by the connection that died, so a clarification
            # raised before the drop would never be seen and the run would sit
            # parked for the full timeout. Replay it from the store.
            clr_state = clarification_store.get(job_id)
            if clr_state and clr_state.status == "pending":
                yield "data: " + json.dumps({
                    "type": "clarification_needed",
                    "clarification_id": clr_state.clarification_id,
                    "question": clr_state.question,
                    "options": clr_state.options,
                }) + "\n\n"

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
                        yield f"data: {json.dumps(_terminal_event(status))}\n\n"
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
            was_terminal = job_manager.get_status(job_id).state in TERMINAL_STATES
            job_manager.cancel_job(job_id)
        except KeyError:
            raise HTTPException(status_code=404, detail="Job not found")
        # Wake any waiting clarification, and tell the stream right away rather
        # than letting the UI find out on the next 30s keepalive. A job that had
        # already finished is left alone: cancel_job is a no-op on it, so
        # reporting "cancelled" and pushing the event would tell the client its
        # completed result had been thrown away.
        if not was_terminal:
            clarification_store.mark_cancelled(job_id)
            job_manager.push_event(job_id, {"type": "cancelled"})
        return {"job_id": job_id, "status": job_manager.get_status(job_id).state}

    @app.post("/api/research/{job_id}/clarify")
    async def submit_clarification(job_id: str, req: ClarifyRequest):
        try:
            job_manager.get_status(job_id)
        except KeyError:
            raise HTTPException(status_code=404, detail="Job not found")

        clr_state = clarification_store.get(job_id)
        if clr_state is None:
            raise HTTPException(status_code=400, detail="No pending clarification for this job")

        if clr_state.clarification_id != req.clarification_id:
            raise HTTPException(status_code=409, detail="Clarification ID mismatch")

        if clr_state.status == "answered":
            return {"status": "already_answered", "used_answer": False}
        if clr_state.status == "timed_out":
            return {"status": "expired", "used_answer": False}
        if clr_state.status == "cancelled":
            return {"status": "cancelled", "used_answer": False}

        result = clarification_store.submit_answer(job_id, req.clarification_id, req.answer)
        if result is None:
            return {"status": "expired", "used_answer": False}

        return {"status": "accepted", "job_id": job_id}

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
