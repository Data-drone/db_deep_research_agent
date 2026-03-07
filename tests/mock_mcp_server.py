"""Mock MCP server for integration testing.

Returns canned responses for Genie and Vector Search queries.
Implements the same interface as MCPClientManager so the agent
graph can run end-to-end without Databricks access.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from deep_research.config import MCPServerConfig

FIXTURES = Path(__file__).parent / "fixtures"


class MockMCPServer:
    """Simulates an MCP server with pre-loaded responses."""

    def __init__(self, server_name: str, responses_file: str) -> None:
        self.server_name = server_name
        self._responses = self._load_responses(responses_file)

    def _load_responses(self, filename: str) -> dict[str, Any]:
        path = FIXTURES / filename
        if path.exists():
            with open(path) as f:
                return json.load(f)
        return {}

    async def call_tool(self, tool_name: str, arguments: dict) -> dict[str, Any]:
        """Return canned response matching the query, or a default."""
        query = arguments.get("query", "")
        for pattern, response in self._responses.items():
            if pattern.lower() in query.lower():
                return response
        return {
            "result": f"No mock data for query: {query}",
            "source": self.server_name,
        }


class MockMCPClientManager:
    """Drop-in replacement for MCPClientManager in tests."""

    def __init__(self) -> None:
        self._servers = {
            "genie_sales": MockMCPServer("genie_sales", "mock_genie_responses.json"),
            "vector_search_kb": MockMCPServer(
                "vector_search_kb", "mock_vector_search_responses.json"
            ),
        }

    def get_available_servers(self) -> dict[str, MCPServerConfig]:
        return {
            name: MCPServerConfig(
                name=name,
                url=f"mock://{name}",
                display_name=name.replace("_", " ").title(),
                server_kind="managed",
                enabled=True,
                risk_tier="safe",
                capability="read",
            )
            for name in self._servers
        }

    async def call_tool(
        self, server_name: str, tool_name: str, arguments: dict
    ) -> dict[str, Any]:
        server = self._servers.get(server_name)
        if not server:
            raise RuntimeError(f"No mock server: {server_name}")
        return await server.call_tool(tool_name, arguments)
