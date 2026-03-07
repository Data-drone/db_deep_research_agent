"""Tests for MCP client manager."""

import pytest

from deep_research.config import MCPServerConfig
from deep_research.mcp_client import MCPClientManager


@pytest.fixture
def mock_server_configs():
    return {
        "genie_sales": MCPServerConfig(
            name="genie_sales",
            url="https://test.databricks.net/api/2.0/mcp/genie/space-123",
            display_name="Sales Data (Genie)",
            server_kind="managed",
            enabled=True,
            risk_tier="safe",
            capability="read",
            managed_type="genie",
            description="Sales queries",
        ),
        "disabled_server": MCPServerConfig(
            name="disabled_server",
            url="https://test.databricks.net/api/2.0/mcp/genie/space-999",
            display_name="Disabled",
            server_kind="managed",
            enabled=False,
            risk_tier="safe",
            capability="read",
            managed_type="genie",
        ),
        "vector_kb": MCPServerConfig(
            name="vector_kb",
            url="https://test.databricks.net/api/2.0/mcp/vector-search/idx",
            display_name="Knowledge Base",
            server_kind="managed",
            enabled=True,
            risk_tier="safe",
            capability="read",
            managed_type="vector_search",
        ),
    }


def test_manager_filters_disabled_servers(mock_server_configs):
    manager = MCPClientManager(mock_server_configs, token="test")
    available = manager.get_available_servers()
    assert "genie_sales" in available
    assert "disabled_server" not in available


def test_manager_get_server_info(mock_server_configs):
    manager = MCPClientManager(mock_server_configs, token="test")
    info = manager.get_server_info("genie_sales")
    assert info.display_name == "Sales Data (Genie)"
    assert info.risk_tier == "safe"


def test_manager_get_server_info_unknown(mock_server_configs):
    manager = MCPClientManager(mock_server_configs, token="test")
    with pytest.raises(KeyError, match="Unknown MCP server"):
        manager.get_server_info("nonexistent")


def test_manager_get_servers_by_risk_tier(mock_server_configs):
    manager = MCPClientManager(mock_server_configs, token="test")
    safe = manager.get_servers_by_risk_tier("safe")
    assert "genie_sales" in safe
    assert "vector_kb" in safe
    # disabled not included
    assert "disabled_server" not in safe


def test_manager_get_servers_by_capability(mock_server_configs):
    manager = MCPClientManager(mock_server_configs, token="test")
    read_only = manager.get_servers_by_capability("read")
    assert "genie_sales" in read_only
