"""MLflow tracing integration for the research agent graph."""

from __future__ import annotations

import logging
from contextlib import asynccontextmanager, contextmanager
from typing import Any, AsyncGenerator, Generator

logger = logging.getLogger(__name__)

# MLflow imports are optional — tracing degrades gracefully if unavailable
try:
    import mlflow
    from mlflow.entities import SpanType

    MLFLOW_AVAILABLE = True
except ImportError:
    MLFLOW_AVAILABLE = False
    mlflow = None  # type: ignore[assignment]
    SpanType = None  # type: ignore[assignment,misc]


def configure_tracing(experiment_name: str = "deep-research-agent") -> None:
    """Enable MLflow tracing for the research agent.

    Sets the tracking URI to 'databricks' so that traces are persisted to
    the Databricks workspace MLflow server (not a local file store).

    Uses mlflow.langchain.autolog() for automatic LangChain/LangGraph tracing.
    Also provides explicit trace_research / trace_quick_reply wrappers for
    reliable tracing from background async tasks.

    Handles both import absence and runtime failures gracefully.
    """
    if not MLFLOW_AVAILABLE:
        logger.warning("mlflow not available — tracing disabled")
        return

    try:
        mlflow.set_tracking_uri("databricks")
        mlflow.set_experiment(experiment_name)
        mlflow.langchain.autolog()
        logger.info(
            f"MLflow tracing enabled (experiment: {experiment_name}, "
            f"tracking_uri: {mlflow.get_tracking_uri()})"
        )
    except Exception:
        logger.warning("MLflow tracing setup failed — continuing without tracing",
                       exc_info=True)


@asynccontextmanager
async def trace_research(
    job_id: str,
    query: str,
    tools: list[str],
    output_mode: str,
    mode: str = "research",
) -> AsyncGenerator[dict[str, Any], None]:
    """Async context manager that wraps a research execution in an MLflow trace.

    Creates a top-level trace so that graph/LLM calls inside are captured.
    Always yields — tracing failures never block execution.

    Usage:
        async with trace_research(job_id, query, tools, output_mode) as ctx:
            # run graph or quick reply
            ctx["output"] = result
    """
    ctx: dict[str, Any] = {}

    if not MLFLOW_AVAILABLE:
        yield ctx
        return

    trace = None
    try:
        trace = mlflow.start_trace(
            name=f"{mode}_research",
            attributes={
                "job_id": job_id,
                "query": query[:500],
                "tools": ",".join(tools),
                "output_mode": output_mode,
                "mode": mode,
            },
        )
        ctx["trace"] = trace
        logger.info(f"MLflow trace started for job {job_id}")
    except Exception:
        logger.warning(f"Failed to start MLflow trace for job {job_id}",
                       exc_info=True)

    try:
        yield ctx
    finally:
        if trace is not None:
            try:
                output_preview = str(ctx.get("output", ""))[:1000]
                status = ctx.get("status", "UNSET")
                mlflow.end_trace(
                    trace.request_id,
                    attributes={"output_preview": output_preview, "final_status": status},
                )
                logger.info(f"MLflow trace ended for job {job_id} (status={status})")
            except Exception:
                logger.warning(f"Failed to end MLflow trace for job {job_id}",
                               exc_info=True)


@contextmanager
def trace_node(
    node_name: str, trace_id: str = "", attributes: dict[str, Any] | None = None
) -> Generator[dict[str, Any], None, None]:
    """Context manager to create a span for a graph node execution.

    Guarantees the wrapped code always executes, even if tracing fails.
    Application exceptions are never swallowed by tracing errors.

    Usage:
        with trace_node("researcher", trace_id="abc") as span_data:
            # do work
            span_data["output"] = result
    """
    span_data: dict[str, Any] = {"node": node_name, "trace_id": trace_id}

    if not MLFLOW_AVAILABLE:
        yield span_data
        return

    # Try to create a span, but fall back to no-op if it fails
    span = None
    try:
        span_ctx = mlflow.start_span(
            name=node_name,
            span_type=SpanType.CHAIN,
            attributes={
                "node_name": node_name,
                "trace_id": trace_id,
                **(attributes or {}),
            },
        )
        span = span_ctx.__enter__()
        span_data["span"] = span
    except Exception:
        logger.warning(f"Failed to create tracing span for {node_name}",
                       exc_info=True)

    # Always yield — wrapped code must execute regardless of tracing
    try:
        yield span_data
    finally:
        # Clean up the span if we created one
        if span is not None:
            try:
                if "output" in span_data:
                    span.set_attribute(
                        "output_preview", str(span_data["output"])[:500]
                    )
                span_ctx.__exit__(None, None, None)
            except Exception:
                logger.warning(f"Failed to close tracing span for {node_name}",
                               exc_info=True)


def log_trace_metadata(
    trace_id: str,
    job_id: str,
    query: str,
    tools: list[str],
    output_mode: str,
) -> None:
    """Log top-level metadata for a research trace."""
    if not MLFLOW_AVAILABLE:
        return

    try:
        mlflow.log_params(
            {
                "trace_id": trace_id,
                "job_id": job_id,
                "query_preview": query[:200],
                "tools": ",".join(tools),
                "output_mode": output_mode,
            }
        )
    except Exception:
        logger.warning("Failed to log trace metadata", exc_info=True)
