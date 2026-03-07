"""Tests for configuration loading."""

from deep_research.config import MCPServerConfig, load_mcp_config


def test_load_mcp_config_from_yaml(fixtures_dir):
    servers = load_mcp_config(fixtures_dir / "test_mcp_config.yaml")
    assert "genie_sales" in servers
    assert "vector_search_kb" in servers


def test_mcp_server_config_fields(fixtures_dir):
    servers = load_mcp_config(fixtures_dir / "test_mcp_config.yaml")
    genie = servers["genie_sales"]
    assert genie.display_name == "Sales Data (Genie)"
    assert genie.risk_tier == "safe"
    assert genie.capability == "read"
    assert genie.enabled is True


def test_mcp_server_config_defaults():
    server = MCPServerConfig(url="https://example.com", display_name="Test")
    assert server.enabled is True
    assert server.risk_tier == "safe"
    assert server.capability == "read"
    assert server.description == ""
