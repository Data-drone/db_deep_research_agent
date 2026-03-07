"""Configuration loading from YAML and environment."""

from __future__ import annotations

import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import Literal

import yaml


@dataclass(frozen=True)
class MCPServerConfig:
    url: str
    display_name: str
    enabled: bool = True
    risk_tier: Literal["safe", "restricted", "privileged"] = "safe"
    capability: Literal["read", "read_write"] = "read"
    description: str = ""


@dataclass(frozen=True)
class AppConfig:
    databricks_host: str
    databricks_token: str
    llm_endpoint: str
    max_iterations: int = 5
    max_tool_calls: int = 20
    time_cap_seconds: int = 120
    managed_servers: dict[str, MCPServerConfig] = field(default_factory=dict)
    custom_servers: dict[str, MCPServerConfig] = field(default_factory=dict)


def load_mcp_config(path: Path) -> dict[str, MCPServerConfig]:
    """Load MCP server configuration from YAML file."""
    with open(path) as f:
        raw = yaml.safe_load(f)

    servers: dict[str, MCPServerConfig] = {}
    for section in ("managed_servers", "custom_servers"):
        for name, cfg in (raw.get(section) or {}).items():
            if isinstance(cfg, dict):
                servers[name] = MCPServerConfig(**cfg)
    return servers


def load_app_config(mcp_config_path: Path | None = None) -> AppConfig:
    """Load full application config from environment + YAML."""
    mcp_path = mcp_config_path or Path("mcp_config.yaml")
    servers = load_mcp_config(mcp_path) if mcp_path.exists() else {}

    managed = {k: v for k, v in servers.items() if k.startswith("genie") or k.startswith("vector")}
    custom = {k: v for k, v in servers.items() if k not in managed}

    return AppConfig(
        databricks_host=os.environ.get("DATABRICKS_HOST", ""),
        databricks_token=os.environ.get("DATABRICKS_TOKEN", ""),
        llm_endpoint=os.environ.get("LLM_ENDPOINT_NAME", "databricks-meta-llama-3-1-70b-instruct"),
        max_iterations=int(os.environ.get("MAX_ITERATIONS", "5")),
        max_tool_calls=int(os.environ.get("MAX_TOOL_CALLS", "20")),
        time_cap_seconds=int(os.environ.get("TIME_CAP_SECONDS", "120")),
        managed_servers=managed,
        custom_servers=custom,
    )
