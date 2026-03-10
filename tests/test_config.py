"""Tests for configuration loading."""

import os

import pytest

from deep_research.config import ConfigError, MCPServerConfig, load_app_config, load_mcp_config


def test_load_mcp_config_from_yaml(fixtures_dir):
    servers = load_mcp_config(fixtures_dir / "test_mcp_config.yaml")
    assert "genie_aus_market" in servers
    assert "knowledge_assistant" in servers


def test_mcp_server_config_fields(fixtures_dir):
    servers = load_mcp_config(fixtures_dir / "test_mcp_config.yaml")
    genie = servers["genie_aus_market"]
    assert genie.name == "genie_aus_market"
    assert genie.display_name == "Australian Economic & Market Data (Genie)"
    assert genie.risk_tier == "safe"
    assert genie.capability == "read"
    assert genie.enabled is True
    assert genie.server_kind == "managed"
    assert genie.managed_type == "genie"


def test_mcp_server_config_defaults():
    server = MCPServerConfig(
        name="test", url="https://example.com", display_name="Test", server_kind="custom"
    )
    assert server.enabled is True
    assert server.risk_tier == "safe"
    assert server.capability == "read"
    assert server.description == ""
    assert server.managed_type is None


def test_server_kind_set_from_yaml_section(fixtures_dir):
    servers = load_mcp_config(fixtures_dir / "test_mcp_config.yaml")
    assert servers["genie_aus_market"].server_kind == "managed"
    assert servers["knowledge_assistant"].server_kind == "managed"


def test_managed_type_inferred(fixtures_dir):
    servers = load_mcp_config(fixtures_dir / "test_mcp_config.yaml")
    assert servers["genie_aus_market"].managed_type == "genie"
    assert servers["knowledge_assistant"].managed_type == "knowledge_assistant"


def test_duplicate_server_name_raises(tmp_path):
    yaml_content = """
managed_servers:
  dupe_server:
    url: "https://a.com"
    display_name: "A"
custom_servers:
  dupe_server:
    url: "https://b.com"
    display_name: "B"
"""
    path = tmp_path / "bad.yaml"
    path.write_text(yaml_content)
    with pytest.raises(ConfigError, match="Duplicate server name"):
        load_mcp_config(path)


def test_load_app_config_missing_env_raises(tmp_path, monkeypatch):
    monkeypatch.delenv("DATABRICKS_HOST", raising=False)
    monkeypatch.delenv("DATABRICKS_TOKEN", raising=False)
    with pytest.raises(ConfigError, match="DATABRICKS_HOST"):
        load_app_config(mcp_config_path=tmp_path / "nonexistent.yaml")


def test_load_app_config_success(tmp_path, monkeypatch, fixtures_dir):
    monkeypatch.setenv("DATABRICKS_HOST", "https://test.databricks.net")
    monkeypatch.setenv("DATABRICKS_TOKEN", "dapi_test")
    config = load_app_config(mcp_config_path=fixtures_dir / "test_mcp_config.yaml")
    assert config.databricks_host == "https://test.databricks.net"
    assert len(config.managed_servers) == 2
    assert len(config.custom_servers) == 0
    # Verify tuples, not lists
    assert isinstance(config.managed_servers, tuple)
    assert isinstance(config.custom_servers, tuple)


def test_frozen_config_immutable(sample_app_config):
    with pytest.raises(AttributeError):
        sample_app_config.databricks_host = "new"


def test_get_server(sample_app_config):
    server = sample_app_config.get_server("genie_test")
    assert server is not None
    assert server.display_name == "Test Genie"
    assert sample_app_config.get_server("nonexistent") is None


def test_load_app_config_has_critic_endpoint(tmp_path, monkeypatch, fixtures_dir):
    monkeypatch.setenv("DATABRICKS_HOST", "https://test.databricks.net")
    monkeypatch.setenv("DATABRICKS_TOKEN", "dapi_test")
    monkeypatch.setenv("CRITIC_LLM_ENDPOINT", "databricks-gpt-5-4")
    config = load_app_config(mcp_config_path=fixtures_dir / "test_mcp_config.yaml")
    assert config.critic_llm_endpoint == "databricks-gpt-5-4"


def test_critic_endpoint_defaults_to_worker(tmp_path, monkeypatch, fixtures_dir):
    monkeypatch.setenv("DATABRICKS_HOST", "https://test.databricks.net")
    monkeypatch.setenv("DATABRICKS_TOKEN", "dapi_test")
    monkeypatch.delenv("CRITIC_LLM_ENDPOINT", raising=False)
    config = load_app_config(mcp_config_path=fixtures_dir / "test_mcp_config.yaml")
    assert config.critic_llm_endpoint == config.llm_endpoint
