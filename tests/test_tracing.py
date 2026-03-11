"""Tests for MLflow tracing integration."""

from deep_research.tracing import configure_tracing, trace_node, log_trace_metadata


def test_trace_node_yields_span_data():
    """trace_node should yield a dict even without mlflow."""
    with trace_node("researcher", trace_id="test-123") as span_data:
        span_data["output"] = "some result"

    assert span_data["node"] == "researcher"
    assert span_data["trace_id"] == "test-123"
    assert span_data["output"] == "some result"


def test_configure_tracing_no_mlflow():
    """configure_tracing should not raise when mlflow is unavailable."""
    # This test runs in an environment where mlflow may not be installed
    configure_tracing("test-experiment")


def test_log_trace_metadata_no_mlflow():
    """log_trace_metadata should not raise when mlflow is unavailable."""
    log_trace_metadata(
        trace_id="test-123",
        job_id="job-abc",
        query="What is revenue?",
        tools=["genie_sales"],
        output_mode="chat",
    )
