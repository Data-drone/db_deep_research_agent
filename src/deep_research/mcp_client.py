"""MCP client manager — connects to and manages MCP servers."""

from __future__ import annotations

import logging
from typing import Any

from deep_research.config import MCPServerConfig

logger = logging.getLogger(__name__)


class MCPClientManager:
    """Manages connections to MCP servers and tool discovery."""

    def __init__(
        self,
        server_configs: dict[str, MCPServerConfig],
        token: str,
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

    async def connect_all(self) -> None:
        """Establish connections to all enabled MCP servers."""
        for name, cfg in self.get_available_servers().items():
            try:
                await self._connect_server(name, cfg)
                logger.info(f"Connected to MCP server: {name} ({cfg.display_name})")
            except Exception:
                logger.exception(f"Failed to connect to MCP server: {name}")

    async def _connect_server(self, name: str, cfg: MCPServerConfig) -> None:
        """Connect to a single MCP server. Placeholder for real MCP SDK integration."""
        # TODO: Replace with actual MCP client SDK connection
        # from mcp import ClientSession
        # session = await ClientSession.connect(cfg.url, token=self._token)
        # self._connections[name] = session
        self._connections[name] = {"url": cfg.url, "status": "connected"}

    async def call_tool(
        self, server_name: str, tool_name: str, arguments: dict[str, Any]
    ) -> dict[str, Any]:
        """Call a tool on a specific MCP server."""
        if server_name not in self._connections:
            raise RuntimeError(f"Not connected to server: {server_name}")

        # TODO: Replace with actual MCP tool call
        # session = self._connections[server_name]
        # result = await session.call_tool(tool_name, arguments)
        # return result
        return {"status": "placeholder", "server": server_name, "tool": tool_name}

    async def disconnect_all(self) -> None:
        """Close all MCP server connections."""
        for name in list(self._connections.keys()):
            try:
                # TODO: session.close()
                del self._connections[name]
                logger.info(f"Disconnected from MCP server: {name}")
            except Exception:
                logger.exception(f"Error disconnecting from: {name}")
