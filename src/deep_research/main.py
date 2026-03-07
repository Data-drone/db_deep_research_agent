"""Application entry point — wires everything together."""

from __future__ import annotations

import logging
import os
import sys
from pathlib import Path

# Configure root logger early so Databricks Apps captures output
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    stream=sys.stderr,
)
logger = logging.getLogger(__name__)


def _build_server_map(config):
    """Merge managed and custom servers, detecting name collisions."""
    from deep_research.config import ConfigError

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
    from deep_research.config import ConfigError

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
    from deep_research.api.app import create_app
    from deep_research.config import load_app_config
    from deep_research.graph import build_research_graph
    from deep_research.mcp_client import MCPClientManager
    from deep_research.tracing import configure_tracing

    logger.info("Starting application initialization...")
    logger.info(f"CWD: {os.getcwd()}")
    logger.info(f"__file__: {__file__}")
    logger.info(f"PYTHONPATH: {os.environ.get('PYTHONPATH', 'not set')}")
    logger.info(f"DATABRICKS_HOST set: {bool(os.environ.get('DATABRICKS_HOST'))}")
    logger.info(f"DATABRICKS_TOKEN set: {bool(os.environ.get('DATABRICKS_TOKEN'))}")

    # Load and validate config (fails fast on missing env vars)
    config = load_app_config()
    logger.info("Configuration loaded successfully")
    logger.info(f"LLM endpoint: {config.llm_endpoint}")
    logger.info(f"Managed servers: {len(config.managed_servers)}")

    # Configure MLflow tracing (non-fatal)
    try:
        configure_tracing()
    except Exception:
        logger.warning("MLflow tracing setup failed — continuing without tracing",
                       exc_info=True)

    # Build MCP client manager (fail fast on duplicate names)
    all_servers = _build_server_map(config)
    mcp_manager = MCPClientManager(all_servers, token=config.databricks_token)
    logger.info(f"MCP client configured with {len(all_servers)} server(s)")

    # Build LLM model (fail fast if unavailable)
    model = _create_model(config)
    logger.info("Model created successfully")

    # Build the research graph
    graph = build_research_graph(model=model, mcp_manager=mcp_manager)
    logger.info("Research graph compiled successfully")

    # Create the FastAPI app
    app = create_app()
    logger.info("FastAPI app created")

    # Store graph and manager on app state for endpoint access
    app.state.graph = graph
    app.state.mcp_manager = mcp_manager
    app.state.config = config

    # Serve React build if it exists — mounted AFTER API routes
    # so /api/* and /health are handled by the app first
    ui_dist = Path(__file__).parent.parent.parent / "ui" / "dist"
    logger.info(f"Checking for UI at: {ui_dist} (exists: {ui_dist.exists()})")
    if ui_dist.exists():
        from fastapi.staticfiles import StaticFiles

        app.mount("/", StaticFiles(directory=str(ui_dist), html=True), name="ui")
        logger.info(f"Serving React UI from {ui_dist}")
    else:
        logger.info("No UI build found — API-only mode")

    logger.info("Application initialization complete")
    return app


# Module-level app for non-factory uvicorn usage: `uvicorn deep_research.main:app`
# Also works with --factory: `uvicorn deep_research.main:create_production_app --factory`
try:
    app = create_production_app()
except Exception:
    logger.exception("Failed to create app — starting with health-check-only fallback")
    from fastapi import FastAPI
    app = FastAPI(title="Deep Research Agent (startup failed)")

    @app.get("/health")
    async def health():
        return {"status": "error", "message": "App failed to initialize"}


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(app, host="0.0.0.0", port=8000)
