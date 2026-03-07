"""MLflow tracing integration for the research agent graph."""

from __future__ import annotations

import logging
from contextlib import contextmanager
from typing import Any, Generator

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
    """Enable MLflow autologging for LangGraph if available.

    Handles both import absence and runtime failures gracefully.
    """
    if not MLFLOW_AVAILABLE:
        logger.warning("mlflow not available — tracing disabled")
        return

    try:
        mlflow.set_experiment(experiment_name)
        mlflow.langchain.autolog()
        logger.info(f"MLflow tracing enabled (experiment: {experiment_name})")
    except Exception:
        logger.warning("MLflow tracing setup failed — continuing without tracing",
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
