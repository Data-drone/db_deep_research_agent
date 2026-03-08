"""MCP client manager — connects to Databricks MCP servers via databricks-mcp."""

from __future__ import annotations

import asyncio
import json
import logging
import re
from typing import Any

import httpx

from deep_research.config import MCPServerConfig

logger = logging.getLogger(__name__)

# Maximum number of poll attempts for Genie async responses
GENIE_POLL_MAX_ATTEMPTS = 30
GENIE_POLL_INTERVAL_SECONDS = 2.0


def _extract_text_from_call_result(result: Any) -> str:
    """Extract text content from a CallToolResult."""
    if hasattr(result, "content"):
        parts = []
        for item in result.content:
            if hasattr(item, "text"):
                parts.append(item.text)
        return "\n".join(parts)
    return str(result)


def _parse_genie_async_response(text: str) -> dict[str, str] | None:
    """Parse a Genie async polling response to extract conversation_id and message_id.

    Genie tool calls return text like:
        The query is being processed. Status: FILTERING_CONTEXT.
        ... conversation_id: abc123, message_id: def456 ...

    Returns dict with conversation_id and message_id, or None if not a polling response.
    """
    if not any(status in text for status in (
        "FILTERING_CONTEXT", "EXECUTING_QUERY", "is being processed",
        "still processing", "poll_response",
    )):
        return None

    # Try JSON parsing first (Genie returns structured JSON via MCP)
    try:
        data = json.loads(text)
        if isinstance(data, dict):
            # Handle both camelCase (from Genie) and snake_case
            conv_id = data.get("conversationId") or data.get("conversation_id", "")
            msg_id = data.get("messageId") or data.get("message_id", "")
            if conv_id and msg_id:
                return {"conversation_id": conv_id, "message_id": msg_id}
    except (json.JSONDecodeError, TypeError):
        pass

    # Regex fallback for both camelCase and snake_case in text
    conv_match = re.search(
        r"(?:conversation_id|conversationId)[\"']?\s*[:=]\s*[\"']?([a-zA-Z0-9_-]+)", text
    )
    msg_match = re.search(
        r"(?:message_id|messageId)[\"']?\s*[:=]\s*[\"']?([a-zA-Z0-9_-]+)", text
    )

    if conv_match and msg_match:
        return {
            "conversation_id": conv_match.group(1),
            "message_id": msg_match.group(1),
        }

    return None


class MCPClientManager:
    """Manages connections to Databricks MCP servers using databricks-mcp."""

    def __init__(
        self,
        server_configs: dict[str, MCPServerConfig],
        token: str = "",
    ) -> None:
        self._configs = server_configs
        self._token = token
        self._connections: dict[str, Any] = {}

    def get_available_servers(self) -> dict[str, MCPServerConfig]:
        """Return only enabled server configurations."""
        return {
            name: cfg
            for name, cfg in self._configs.items()
            if cfg.enabled
        }

    def get_server_info(self, server_name: str) -> MCPServerConfig:
        """Get configuration for a specific server."""
        if server_name not in self._configs:
            raise KeyError(f"Unknown MCP server: {server_name}")
        return self._configs[server_name]

    def get_servers_by_risk_tier(
        self, risk_tier: str
    ) -> dict[str, MCPServerConfig]:
        """Return enabled servers matching a risk tier."""
        return {
            name: cfg
            for name, cfg in self.get_available_servers().items()
            if cfg.risk_tier == risk_tier
        }

    def get_servers_by_capability(
        self, capability: str
    ) -> dict[str, MCPServerConfig]:
        """Return enabled servers matching a capability."""
        return {
            name: cfg
            for name, cfg in self.get_available_servers().items()
            if cfg.capability == capability
        }

    def _create_workspace_client(self) -> Any:
        """Create a WorkspaceClient for authentication.

        In Databricks Apps, WorkspaceClient() auto-detects OAuth credentials
        (DATABRICKS_CLIENT_ID + DATABRICKS_CLIENT_SECRET).
        Falls back to DATABRICKS_TOKEN if available.
        """
        try:
            from databricks.sdk import WorkspaceClient
            return WorkspaceClient()
        except Exception:
            logger.warning(
                "Could not create WorkspaceClient — MCP calls may fail",
                exc_info=True,
            )
            return None

    async def connect_all(self) -> None:
        """Discover tools from all enabled MCP servers."""
        ws = self._create_workspace_client()
        if ws is None:
            logger.warning("No WorkspaceClient available — skipping MCP connections")
            return

        for name, cfg in self.get_available_servers().items():
            try:
                await self._connect_server(name, cfg, ws)
                tools = self._connections[name].get("tools", [])
                tool_names = [t.name for t in tools]
                logger.info(
                    f"Connected to MCP server: {name} ({cfg.display_name}) "
                    f"— tools: {tool_names}"
                )
            except Exception:
                logger.exception(f"Failed to connect to MCP server: {name}")

    async def _connect_server(
        self, name: str, cfg: MCPServerConfig, ws: Any
    ) -> None:
        """Connect to a single MCP server and discover its tools."""
        if cfg.managed_type == "knowledge_assistant":
            await self._connect_knowledge_assistant(name, cfg, ws)
            return

        from databricks_mcp import DatabricksMCPClient

        client = DatabricksMCPClient(server_url=cfg.url, workspace_client=ws)
        tools = await client.alist_tools()

        # Find the poll tool for Genie servers (poll_response_*)
        poll_tool = None
        query_tool = None
        for tool in tools:
            if tool.name.startswith("poll_response"):
                poll_tool = tool.name
            else:
                query_tool = tool.name

        self._connections[name] = {
            "client": client,
            "tools": tools,
            "query_tool": query_tool,
            "poll_tool": poll_tool,
            "config": cfg,
        }

    def _get_fresh_auth_headers(self, ws: Any) -> dict[str, str]:
        """Get fresh authentication headers from a WorkspaceClient.

        The SDK's authenticate() handles token refresh automatically for
        OAuth M2M (client credentials) — each call returns a valid bearer
        token, requesting a new one from the IdP when the cached one expires.
        """
        try:
            headers = ws.config.authenticate()
            if not headers:
                logger.warning(
                    "WorkspaceClient.authenticate() returned empty headers. "
                    "Check DATABRICKS_CLIENT_ID/DATABRICKS_CLIENT_SECRET or DATABRICKS_TOKEN."
                )
                return {}
            return headers
        except Exception:
            logger.warning("Failed to get auth headers from WorkspaceClient", exc_info=True)
            return {}

    async def _connect_knowledge_assistant(
        self, name: str, cfg: MCPServerConfig, ws: Any
    ) -> None:
        """Set up connection to a Knowledge Assistant serving endpoint.

        KA endpoints are not MCP servers — they use the Responses API format
        at /serving-endpoints/{name}/invocations.

        We store the WorkspaceClient reference so that each call fetches
        fresh auth headers — OAuth M2M tokens expire and must be refreshed.
        """
        # Verify auth works *now* — fail fast if credentials are missing
        auth_headers = self._get_fresh_auth_headers(ws)

        # Verify endpoint is reachable with a lightweight call
        if auth_headers:
            async with httpx.AsyncClient(timeout=30.0) as client:
                resp = await client.post(
                    cfg.url,
                    headers={**auth_headers, "Content-Type": "application/json"},
                    json={"input": [{"role": "user", "content": "ping"}]},
                )
                if resp.status_code != 200:
                    raise RuntimeError(
                        f"KA endpoint returned {resp.status_code}: {resp.text[:200]}"
                    )
        else:
            logger.warning(
                f"Skipping KA endpoint health check for {name} — no auth headers. "
                "KA calls will attempt fresh auth at request time."
            )

        from types import SimpleNamespace
        self._connections[name] = {
            "client": None,  # No MCP client for KA
            "tools": [SimpleNamespace(name=f"knowledge_assistant_{name}")],
            "query_tool": f"knowledge_assistant_{name}",
            "poll_tool": None,
            "config": cfg,
            "workspace_client": ws,  # Stored for fresh auth on each call
        }

    async def call_tool(
        self, server_name: str, tool_name: str, arguments: dict[str, Any]
    ) -> dict[str, Any]:
        """Call a tool on a specific MCP server.

        Args:
            server_name: The config key of the server to call.
            tool_name: Generic action name (e.g., "query"). Mapped to the
                       actual MCP tool name discovered during connect.
            arguments: Arguments to pass (e.g., {"query": "..."}).

        Returns:
            Dict with "result" (text) and "source" (server name).
        """
        if server_name not in self._connections:
            raise RuntimeError(f"Not connected to server: {server_name}")

        conn = self._connections[server_name]
        cfg = conn["config"]

        # Knowledge Assistant uses serving endpoint, not MCP
        if cfg.managed_type == "knowledge_assistant":
            return await self._call_knowledge_assistant(server_name, conn, arguments)

        client = conn["client"]
        actual_tool = conn["query_tool"]

        if actual_tool is None:
            raise RuntimeError(f"No query tool discovered for server: {server_name}")

        logger.info(f"Calling MCP tool: {actual_tool} on {server_name}")

        result = await client.acall_tool(actual_tool, arguments)
        text = _extract_text_from_call_result(result)

        # Handle Genie async polling
        if cfg.managed_type == "genie" and conn.get("poll_tool"):
            polling_info = _parse_genie_async_response(text)
            if polling_info:
                text = await self._poll_genie_result(
                    client, conn["poll_tool"], polling_info
                )

        return {"result": text, "source": server_name}

    async def _call_knowledge_assistant(
        self,
        server_name: str,
        conn: dict[str, Any],
        arguments: dict[str, Any],
    ) -> dict[str, Any]:
        """Call a Knowledge Assistant serving endpoint.

        Fetches fresh auth headers on every call to handle OAuth token expiry.
        Sends the query via the Responses API format and extracts the text response.
        """
        cfg = conn["config"]
        ws = conn.get("workspace_client")
        if ws is None:
            raise RuntimeError(
                f"No WorkspaceClient available for KA server: {server_name}"
            )
        auth_headers = self._get_fresh_auth_headers(ws)
        if not auth_headers:
            raise RuntimeError(
                f"Could not obtain auth headers for KA: {server_name}. "
                "Ensure DATABRICKS_CLIENT_ID/DATABRICKS_CLIENT_SECRET or "
                "DATABRICKS_TOKEN is configured."
            )
        query = arguments.get("query", "")

        logger.info(f"Calling Knowledge Assistant: {server_name} with query: {query[:100]}")

        async with httpx.AsyncClient(timeout=120.0) as client:
            resp = await client.post(
                cfg.url,
                headers={**auth_headers, "Content-Type": "application/json"},
                json={"input": [{"role": "user", "content": query}]},
            )

        if resp.status_code != 200:
            raise RuntimeError(
                f"KA endpoint returned {resp.status_code}: {resp.text[:200]}"
            )

        data = resp.json()
        # Extract text from Responses API output format
        text = ""
        for output_item in data.get("output", []):
            if output_item.get("type") == "message":
                for content_item in output_item.get("content", []):
                    if content_item.get("type") == "output_text":
                        text += content_item.get("text", "")

        if not text:
            text = json.dumps(data.get("output", data))

        return {"result": text, "source": server_name}

    async def _poll_genie_result(
        self,
        client: Any,
        poll_tool_name: str,
        polling_info: dict[str, str],
    ) -> str:
        """Poll a Genie MCP server until the query completes."""
        for attempt in range(GENIE_POLL_MAX_ATTEMPTS):
            await asyncio.sleep(GENIE_POLL_INTERVAL_SECONDS)

            result = await client.acall_tool(poll_tool_name, polling_info)
            text = _extract_text_from_call_result(result)

            logger.info(
                f"Genie poll attempt {attempt + 1}/{GENIE_POLL_MAX_ATTEMPTS}: "
                f"{text[:100]}..."
            )

            # Check if still processing
            if _parse_genie_async_response(text) is not None:
                continue

            # Response is complete
            return text

        logger.warning("Genie poll timed out — returning last response")
        return text

    async def disconnect_all(self) -> None:
        """Clear all MCP server connections."""
        for name in list(self._connections.keys()):
            try:
                del self._connections[name]
                logger.info(f"Disconnected from MCP server: {name}")
            except Exception:
                logger.exception(f"Error disconnecting from: {name}")
