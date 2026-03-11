"""Factory functions for creating Evidence and ToolCall records from MCP results."""

from __future__ import annotations

import uuid
from datetime import datetime, timezone
from typing import Any

from deep_research.models import Evidence, ToolCall


def build_tool_call_record(
    *,
    tool_name: str,
    server_name: str,
    input_data: dict[str, Any],
    output_data: dict[str, Any],
    latency_ms: int,
    iteration: int,
    started_at: datetime,
    status: str = "success",
    error_type: str | None = None,
    error_message: str | None = None,
) -> ToolCall:
    """Create a ToolCall record with a unique ID."""
    return ToolCall(
        tool_call_id=f"tc-{uuid.uuid4().hex[:8]}",
        tool_name=tool_name,
        server_name=server_name,
        input_data=input_data,
        output_data=output_data,
        latency_ms=latency_ms,
        status=status,
        iteration=iteration,
        started_at=started_at,
        completed_at=datetime.now(timezone.utc),
        error_type=error_type,
        error_message=error_message,
    )


def build_evidence_from_tool_result(
    *,
    tool_name: str,
    tool_call_id: str,
    result: dict[str, Any],
    iteration: int,
    started_at: datetime,
) -> list[Evidence]:
    """Convert an MCP tool result into one or more Evidence records.

    Currently produces one Evidence per result. Future: may produce multiple
    if result contains multiple hits/documents.
    """
    snippet = result.get("result", str(result))
    uri = result.get("table") or result.get("document") or result.get("uri")
    source = result.get("source", tool_name)

    evidence = Evidence(
        evidence_id=f"ev-{uuid.uuid4().hex[:8]}",
        source_id=source,
        source_type="mcp_tool_call",
        title=f"Result from {tool_name}",
        uri=uri,
        snippet=snippet,
        confidence=0.8,
        freshness=started_at.isoformat(),
        tool_that_produced_it=tool_name,
        tool_call_id=tool_call_id,
        iteration=iteration,
        retrieved_at=started_at,
    )

    return [evidence]
