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
    """Enable MLflow autologging for LangGraph if available."""
    if not MLFLOW_AVAILABLE:
        logger.warning("mlflow not available — tracing disabled")
        return

    mlflow.set_experiment(experiment_name)
    mlflow.langchain.autolog()
    logger.info(f"MLflow tracing enabled (experiment: {experiment_name})")


@contextmanager
def trace_node(
    node_name: str, trace_id: str = "", attributes: dict[str, Any] | None = None
) -> Generator[dict[str, Any], None, None]:
    """Context manager to create a span for a graph node execution.

    Usage:
        with trace_node("researcher", trace_id="abc") as span_data:
            # do work
            span_data["output"] = result
    """
    span_data: dict[str, Any] = {"node": node_name, "trace_id": trace_id}

    if not MLFLOW_AVAILABLE:
        yield span_data
        return

    try:
        with mlflow.start_span(
            name=node_name,
            span_type=SpanType.CHAIN,
            attributes={
                "node_name": node_name,
                "trace_id": trace_id,
                **(attributes or {}),
            },
        ) as span:
            span_data["span"] = span
            yield span_data
            if "output" in span_data:
                span.set_attribute("output_preview", str(span_data["output"])[:500])
    except Exception:
        logger.exception(f"Tracing error for node {node_name}")
        yield span_data


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
        logger.exception("Failed to log trace metadata")
