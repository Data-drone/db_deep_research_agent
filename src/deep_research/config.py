"""Configuration loading from YAML and environment."""

from __future__ import annotations

import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import Literal

import yaml


@dataclass(frozen=True)
class MCPServerConfig:
    name: str
    url: str
    display_name: str
    server_kind: Literal["managed", "custom"]
    enabled: bool = True
    risk_tier: Literal["safe", "restricted", "privileged"] = "safe"
    capability: Literal["read", "read_write"] = "read"
    managed_type: Literal["genie", "vector_search", "knowledge_assistant"] | None = None
    description: str = ""


@dataclass(frozen=True)
class AppConfig:
    databricks_host: str
    llm_endpoint: str
    critic_llm_endpoint: str = ""
    databricks_token: str = ""
    max_iterations: int = 5
    max_tool_calls: int = 20
    time_cap_seconds: int = 120
    managed_servers: tuple[MCPServerConfig, ...] = ()
    custom_servers: tuple[MCPServerConfig, ...] = ()

    def get_server(self, name: str) -> MCPServerConfig | None:
        """Look up a server by name across managed and custom."""
        for s in self.managed_servers + self.custom_servers:
            if s.name == name:
                return s
        return None

    def all_servers(self) -> tuple[MCPServerConfig, ...]:
        return self.managed_servers + self.custom_servers


class ConfigError(Exception):
    """Raised when configuration is invalid or incomplete."""


def load_mcp_config(path: Path) -> dict[str, MCPServerConfig]:
    """Load MCP server configuration from YAML file."""
    with open(path) as f:
        raw = yaml.safe_load(f)

    servers: dict[str, MCPServerConfig] = {}
    for section in ("managed_servers", "custom_servers"):
        kind: Literal["managed", "custom"] = "managed" if section == "managed_servers" else "custom"
        for name, cfg in (raw.get(section) or {}).items():
            if not isinstance(cfg, dict):
                continue
            if name in servers:
                raise ConfigError(f"Duplicate server name: {name}")
            cfg_copy = dict(cfg)
            cfg_copy["name"] = name
            cfg_copy["server_kind"] = kind
            # Infer managed_type from YAML section context
            if kind == "managed" and "managed_type" not in cfg_copy:
                if "genie" in name.lower():
                    cfg_copy["managed_type"] = "genie"
                elif "vector" in name.lower():
                    cfg_copy["managed_type"] = "vector_search"
                elif "knowledge" in name.lower():
                    cfg_copy["managed_type"] = "knowledge_assistant"
            servers[name] = MCPServerConfig(**cfg_copy)
    return servers


def load_app_config(mcp_config_path: Path | None = None) -> AppConfig:
    """Load full application config from environment + YAML.

    Required env vars: DATABRICKS_HOST.
    Auth: Uses Databricks SDK unified auth — supports DATABRICKS_TOKEN (PAT)
    or DATABRICKS_CLIENT_ID + DATABRICKS_CLIENT_SECRET (OAuth/service principal).
    Optional: LLM_ENDPOINT_NAME, MAX_ITERATIONS, MAX_TOOL_CALLS, TIME_CAP_SECONDS.
    """
    mcp_path = mcp_config_path or Path("mcp_config.yaml")
    servers = load_mcp_config(mcp_path) if mcp_path.exists() else {}

    host = os.environ.get("DATABRICKS_HOST", "")
    if not host:
        raise ConfigError("Missing required environment variable: DATABRICKS_HOST")

    # Token is optional — Databricks Apps use OAuth (CLIENT_ID/CLIENT_SECRET)
    # and the SDK picks those up automatically via unified auth.
    token = os.environ.get("DATABRICKS_TOKEN", "")

    managed = tuple(v for v in servers.values() if v.server_kind == "managed")
    custom = tuple(v for v in servers.values() if v.server_kind == "custom")

    llm_ep = os.environ.get("LLM_ENDPOINT_NAME", "databricks-claude-sonnet-4-6")
    critic_ep = os.environ.get("CRITIC_LLM_ENDPOINT", "") or llm_ep

    return AppConfig(
        databricks_host=host,
        databricks_token=token,
        llm_endpoint=llm_ep,
        critic_llm_endpoint=critic_ep,
        max_iterations=int(os.environ.get("MAX_ITERATIONS", "2")),
        max_tool_calls=int(os.environ.get("MAX_TOOL_CALLS", "20")),
        time_cap_seconds=int(os.environ.get("TIME_CAP_SECONDS", "120")),
        managed_servers=managed,
        custom_servers=custom,
    )
