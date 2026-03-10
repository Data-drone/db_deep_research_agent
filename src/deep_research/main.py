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
        from databricks_langchain import ChatDatabricks

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


def _create_critic_model(config):
    """Create the critic LLM model for evaluation/verification."""
    if config.critic_llm_endpoint == config.llm_endpoint:
        return None

    try:
        from databricks_langchain import ChatDatabricks
        model = ChatDatabricks(endpoint=config.critic_llm_endpoint)
        logger.info(f"Critic model initialized: {config.critic_llm_endpoint}")
        return model
    except ImportError:
        logger.warning("databricks-langchain not installed — critic model disabled")
        return None
    except Exception as e:
        logger.warning(f"Failed to initialize critic model: {e} — falling back to worker")
        return None


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
        experiment_name = os.environ.get("MLFLOW_EXPERIMENT_NAME", "deep-research-agent")
        configure_tracing(experiment_name=experiment_name)
    except Exception:
        logger.warning("MLflow tracing setup failed — continuing without tracing",
                       exc_info=True)

    # Build MCP client manager (fail fast on duplicate names)
    all_servers = _build_server_map(config)
    mcp_manager = MCPClientManager(all_servers, token=config.databricks_token)
    logger.info(f"MCP client configured with {len(all_servers)} server(s)")

    # Build LLM model (fail fast if unavailable)
    model = _create_model(config)
    critic_model = _create_critic_model(config)
    logger.info("Model(s) created successfully")

    # Build the research graph
    graph = build_research_graph(model=model, mcp_manager=mcp_manager, critic_model=critic_model)
    logger.info("Research graph compiled successfully")

    # Create the FastAPI app
    app = create_app()
    logger.info("FastAPI app created")

    # Store graph and manager on app state for endpoint access
    app.state.graph = graph
    app.state.mcp_manager = mcp_manager
    app.state.config = config

    # Connect to MCP servers on startup (non-fatal — app works without MCP)
    @app.on_event("startup")
    async def _connect_mcp_servers():
        try:
            await mcp_manager.connect_all()
            logger.info("MCP server connections established")
        except Exception:
            logger.warning(
                "Failed to connect MCP servers — tool calls will fail",
                exc_info=True,
            )

    @app.on_event("shutdown")
    async def _disconnect_mcp_servers():
        await mcp_manager.disconnect_all()

    # ── Debug / spike test endpoints (registered BEFORE StaticFiles mount) ──

    # Temporary debug endpoint to diagnose deployment
    @app.get("/debug/paths")
    async def debug_paths():
        cwd = os.getcwd()
        cwd_contents = []
        try:
            cwd_contents = sorted(os.listdir(cwd))
        except Exception as e:
            cwd_contents = [f"error: {e}"]

        # Check UI candidate paths
        ui_candidates = {}
        for label, p in [
            ("__file__ based", str(Path(__file__).parent.parent.parent / "ui" / "dist")),
            ("cwd based", str(Path(cwd) / "ui" / "dist")),
            ("SOURCE_CODE_PATH", str(Path(os.environ.get("DATABRICKS_SOURCE_CODE_PATH", "")) / "ui" / "dist")),
        ]:
            ui_candidates[label] = {
                "path": p,
                "exists": Path(p).exists(),
                "has_index": (Path(p) / "index.html").exists() if Path(p).exists() else False,
            }

        return {
            "cwd": cwd,
            "cwd_contents": cwd_contents,
            "__file__": __file__,
            "app_initialized": True,
            "ui_candidates": ui_candidates,
            "env_SOURCE_CODE_PATH": os.environ.get("DATABRICKS_SOURCE_CODE_PATH", "not set"),
        }

    # SSE spike test — verifies Databricks Apps proxy streams events incrementally.
    # Remove after confirming SSE works.
    import asyncio as _asyncio
    import json as _json
    from starlette.responses import StreamingResponse as _StreamingResponse

    @app.get("/debug/sse-test")
    async def sse_test():
        async def _generate():
            for i in range(5):
                event = {"seq": i, "msg": f"event {i}"}
                yield f"data: {_json.dumps(event)}\n\n"
                await _asyncio.sleep(1)
            yield f"data: {_json.dumps({'seq': 5, 'msg': 'done'})}\n\n"

        return _StreamingResponse(
            _generate(),
            media_type="text/event-stream",
            headers={
                "X-Accel-Buffering": "no",
                "Cache-Control": "no-cache",
                "Connection": "keep-alive",
            },
        )

    # Serve React build if it exists — mounted LAST so API routes take priority.
    candidates = [
        Path(__file__).parent.parent.parent / "ui" / "dist",
        Path(os.getcwd()) / "ui" / "dist",
    ]
    # Also check the source_code_path used by Databricks Apps
    source_code_path = os.environ.get("DATABRICKS_SOURCE_CODE_PATH", "")
    if source_code_path:
        candidates.insert(0, Path(source_code_path) / "ui" / "dist")

    actual_ui_dist = None
    for candidate in candidates:
        exists = candidate.exists()
        has_index = exists and (candidate / "index.html").exists()
        logger.info(f"Checking for UI at: {candidate} (exists: {exists}, has_index: {has_index})")
        if has_index:
            actual_ui_dist = candidate
            break

    if actual_ui_dist:
        from fastapi.staticfiles import StaticFiles

        app.mount("/", StaticFiles(directory=str(actual_ui_dist), html=True), name="ui")
        logger.info(f"Serving React UI from {actual_ui_dist}")
    else:
        logger.info("No UI build found — API-only mode")
        # Provide a helpful root endpoint instead of 404
        from fastapi.responses import JSONResponse

        @app.get("/")
        async def root():
            return JSONResponse({
                "status": "ok",
                "message": "Deep Research Agent API is running. UI not found.",
                "endpoints": {
                    "health": "/health",
                    "tools": "/api/tools",
                    "research": "POST /api/research",
                    "debug": "/debug/paths",
                },
            })

    logger.info("Application initialization complete")
    return app


# Module-level app for non-factory uvicorn usage: `uvicorn deep_research.main:app`
# Also works with --factory: `uvicorn deep_research.main:create_production_app --factory`
_startup_error = None
try:
    app = create_production_app()
except Exception as exc:
    import traceback
    _startup_error = traceback.format_exc()
    logger.exception("Failed to create app — starting with health-check-only fallback")
    from fastapi import FastAPI
    app = FastAPI(title="Deep Research Agent (startup failed)")

    @app.get("/health")
    async def health():
        return {"status": "error", "message": "App failed to initialize"}

    @app.get("/debug/paths")
    async def debug_paths():
        return {
            "app_initialized": False,
            "cwd": os.getcwd(),
            "__file__": __file__,
            "startup_error": _startup_error,
        }


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(app, host="0.0.0.0", port=8000)
