"""Application entry point — wires everything together."""

from __future__ import annotations

import asyncio
import logging
import uuid
from pathlib import Path

from deep_research.api.app import create_app
from deep_research.config import load_app_config
from deep_research.graph import build_research_graph
from deep_research.mcp_client import MCPClientManager
from deep_research.state import create_initial_state
from deep_research.tracing import configure_tracing

logger = logging.getLogger(__name__)


def create_production_app():
    """Create the FastAPI app with real dependencies wired up."""
    config = load_app_config()

    # Configure MLflow tracing
    configure_tracing()

    # Build MCP client manager
    all_servers = {}
    for cfg in config.managed_servers:
        all_servers[cfg.name] = cfg
    for cfg in config.custom_servers:
        all_servers[cfg.name] = cfg

    mcp_manager = MCPClientManager(all_servers, token=config.databricks_token)

    # Build the LLM model
    # TODO: Initialize real ChatDatabricks model
    # from langchain_databricks import ChatDatabricks
    # model = ChatDatabricks(endpoint=config.llm_endpoint)
    model = None  # Will be replaced when deploying

    # Build the research graph
    graph = build_research_graph(model=model, mcp_manager=mcp_manager)

    # Create the FastAPI app
    app = create_app()

    # Store graph and manager on app state for endpoint access
    app.state.graph = graph
    app.state.mcp_manager = mcp_manager
    app.state.config = config

    # Serve React build if it exists
    ui_dist = Path(__file__).parent.parent.parent / "ui" / "dist"
    if ui_dist.exists():
        from fastapi.staticfiles import StaticFiles

        app.mount("/", StaticFiles(directory=str(ui_dist), html=True), name="ui")
        logger.info(f"Serving React UI from {ui_dist}")

    return app


if __name__ == "__main__":
    import uvicorn

    app = create_production_app()
    uvicorn.run(app, host="0.0.0.0", port=8000)
