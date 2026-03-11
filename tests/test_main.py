"""Tests for application factory and startup wiring."""

import pytest
from unittest.mock import patch, MagicMock
from dataclasses import dataclass

from deep_research.config import ConfigError, MCPServerConfig, AppConfig
from deep_research.main import _build_server_map, _create_model


def _make_server(name: str, kind: str = "managed") -> MCPServerConfig:
    return MCPServerConfig(
        name=name,
        url=f"mock://{name}",
        display_name=name.replace("_", " ").title(),
        server_kind=kind,
    )


@dataclass
class FakeConfig:
    managed_servers: tuple
    custom_servers: tuple


class TestBuildServerMap:
    def test_merges_managed_and_custom(self):
        config = FakeConfig(
            managed_servers=(_make_server("genie"),),
            custom_servers=(_make_server("custom_api", "custom"),),
        )
        result = _build_server_map(config)
        assert len(result) == 2
        assert "genie" in result
        assert "custom_api" in result

    def test_duplicate_names_raise(self):
        config = FakeConfig(
            managed_servers=(_make_server("overlap"),),
            custom_servers=(_make_server("overlap", "custom"),),
        )
        with pytest.raises(ConfigError, match="Duplicate MCP server name: overlap"):
            _build_server_map(config)

    def test_empty_servers(self):
        config = FakeConfig(managed_servers=(), custom_servers=())
        result = _build_server_map(config)
        assert result == {}


class TestCreateModel:
    def test_missing_databricks_langchain_raises(self):
        """Without langchain-databricks installed, should raise ConfigError."""

        @dataclass
        class FakeLLMConfig:
            llm_endpoint: str = "test-endpoint"

        # Patch the import inside _create_model to simulate ImportError
        with patch.dict("sys.modules", {"databricks_langchain": None}):
            with pytest.raises(ConfigError, match="langchain-databricks is required"):
                _create_model(FakeLLMConfig())

    def test_model_creation_failure_raises_config_error(self):
        """If ChatDatabricks raises, should wrap in ConfigError."""
        mock_module = MagicMock()
        mock_module.ChatDatabricks.side_effect = RuntimeError("Connection failed")

        @dataclass
        class FakeLLMConfig:
            llm_endpoint: str = "test-endpoint"

        with patch.dict("sys.modules", {"databricks_langchain": mock_module}):
            with pytest.raises(ConfigError, match="Failed to initialize LLM model"):
                _create_model(FakeLLMConfig())

    def test_successful_model_creation(self):
        """With working langchain-databricks, should return model."""
        mock_module = MagicMock()
        mock_model = MagicMock()
        mock_module.ChatDatabricks.return_value = mock_model

        @dataclass
        class FakeLLMConfig:
            llm_endpoint: str = "test-endpoint"

        with patch.dict("sys.modules", {"databricks_langchain": mock_module}):
            result = _create_model(FakeLLMConfig())
            assert result is mock_model
            mock_module.ChatDatabricks.assert_called_once_with(endpoint="test-endpoint")
