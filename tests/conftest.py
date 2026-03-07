"""Shared test fixtures."""

from pathlib import Path

import pytest

from deep_research.config import AppConfig, MCPServerConfig


@pytest.fixture
def sample_mcp_server():
    return MCPServerConfig(
        name="genie_test",
        url="https://test.databricks.net/api/2.0/mcp/genie/test-space",
        display_name="Test Genie",
        server_kind="managed",
        enabled=True,
        risk_tier="safe",
        capability="read",
        managed_type="genie",
        description="Test Genie space",
    )


@pytest.fixture
def sample_app_config(sample_mcp_server):
    return AppConfig(
        databricks_host="https://test.databricks.net",
        databricks_token="dapi_test_token",
        llm_endpoint="databricks-meta-llama-3-1-70b-instruct",
        managed_servers=(sample_mcp_server,),
    )


@pytest.fixture
def fixtures_dir():
    return Path(__file__).parent / "fixtures"
