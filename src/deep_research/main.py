"""Application entry point — wires everything together."""

from __future__ import annotations

import logging
from pathlib import Path

from deep_research.api.app import create_app
from deep_research.config import ConfigError, load_app_config
from deep_research.graph import build_research_graph
from deep_research.mcp_client import MCPClientManager
from deep_research.tracing import configure_tracing

logger = logging.getLogger(__name__)


def _build_server_map(config):
    """Merge managed and custom servers, detecting name collisions."""
    all_servers = {}
    for cfg in [*config.managed_servers, *config.custom_servers]:
        if cfg.name in all_servers:
            raise ConfigError(f"Duplicate MCP server name: {cfg.name}")
        all_servers[cfg.name] = cfg
    return all_servers


def _create_model(config):
    """Create the LLM model from config.

    Uses ChatDatabricks when available, otherwise raises a clear error.
    """
    try:
        from langchain_databricks import ChatDatabricks

        model = ChatDatabricks(endpoint=config.llm_endpoint)
        logger.info(f"LLM model initialized: {config.llm_endpoint}")
        return model
    except ImportError:
        raise ConfigError(
            "langchain-databricks is required but not installed. "
            "Install with: pip install databricks-langchain"
        )
    except Exception as e:
        raise ConfigError(f"Failed to initialize LLM model: {e}") from e


def create_production_app():
    """Create the FastAPI app with real dependencies wired up.

    Fails fast if required configuration is missing or invalid.
    """
    logger.info("Starting application initialization...")

    # Load and validate config (fails fast on missing env vars)
    config = load_app_config()
    logger.info("Configuration loaded successfully")

    # Configure MLflow tracing (non-fatal)
    try:
        configure_tracing()
    except Exception:
        logger.warning("MLflow tracing setup failed — continuing without tracing")

    # Build MCP client manager (fail fast on duplicate names)
    all_servers = _build_server_map(config)
    mcp_manager = MCPClientManager(all_servers, token=config.databricks_token)
    logger.info(f"MCP client configured with {len(all_servers)} server(s)")

    # Build LLM model (fail fast if unavailable)
    model = _create_model(config)

    # Build the research graph
    graph = build_research_graph(model=model, mcp_manager=mcp_manager)
    logger.info("Research graph compiled successfully")

    # Create the FastAPI app
    app = create_app()

    # Store graph and manager on app state for endpoint access
    app.state.graph = graph
    app.state.mcp_manager = mcp_manager
    app.state.config = config

    # Serve React build if it exists — mounted AFTER API routes
    # so /api/* and /health are handled by the app first
    ui_dist = Path(__file__).parent.parent.parent / "ui" / "dist"
    if ui_dist.exists():
        from fastapi.staticfiles import StaticFiles

        # Mount under a catch-all that won't interfere with /api or /health
        app.mount("/", StaticFiles(directory=str(ui_dist), html=True), name="ui")
        logger.info(f"Serving React UI from {ui_dist}")
    else:
        logger.info("No UI build found — API-only mode")

    logger.info("Application initialization complete")
    return app


if __name__ == "__main__":
    import uvicorn

    app = create_production_app()
    uvicorn.run(app, host="0.0.0.0", port=8000)
