"""Application entry point for the Pydantic AI deployment target."""

from __future__ import annotations

import asyncio
import logging
import os
import sys
from pathlib import Path
from typing import Any

import httpx
from pydantic_ai import Agent

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    stream=sys.stderr,
)
logger = logging.getLogger(__name__)

_cleanup_tasks: set[asyncio.Task[Any]] = set()


class WorkspaceOAuthAuth(httpx.Auth):
    """Refresh Databricks SDK authentication headers for every request."""

    requires_request_body = True

    def __init__(self, workspace_client: Any) -> None:
        self.workspace_client = workspace_client

    async def async_auth_flow(
        self,
        request: httpx.Request,
    ):
        headers = await asyncio.to_thread(
            self.workspace_client.config.authenticate
        )
        if not headers or not any(
            key.lower() == "authorization"
            for key in headers
        ):
            raise RuntimeError(
                "Databricks SDK returned no authorization header"
            )

        request.headers.update(headers)
        yield request


def _log_cleanup_task_result(task: asyncio.Task[Any]) -> None:
    """Release a cleanup task and log any exception it raised."""

    _cleanup_tasks.discard(task)
    if task.cancelled():
        logger.warning(
            "Model HTTP client cleanup task was cancelled"
        )
        return

    try:
        exception = task.exception()
    except asyncio.CancelledError:
        logger.warning(
            "Model HTTP client cleanup task was cancelled"
        )
        return

    if exception is not None:
        logger.error(
            "Failed to close model HTTP client during initialization",
            exc_info=(
                type(exception),
                exception,
                exception.__traceback__,
            ),
        )


def _close_http_client_after_initialization_failure(
    client: httpx.AsyncClient,
) -> None:
    """Close a partially initialized client in sync or async contexts."""

    try:
        loop = asyncio.get_running_loop()
    except RuntimeError:
        asyncio.run(client.aclose())
        return

    task = loop.create_task(client.aclose())
    _cleanup_tasks.add(task)
    task.add_done_callback(_log_cleanup_task_result)


def _build_server_map(config: Any) -> dict[str, Any]:
    """Merge managed and custom servers, detecting name collisions."""

    from deep_research.config import ConfigError

    all_servers: dict[str, Any] = {}
    for server_config in [
        *config.managed_servers,
        *config.custom_servers,
    ]:
        if server_config.name in all_servers:
            raise ConfigError(
                f"Duplicate MCP server name: {server_config.name}"
            )
        all_servers[server_config.name] = server_config
    return all_servers


def _create_workspace_client(config: Any) -> Any:
    """Create a Databricks SDK client using configured credentials."""

    from databricks.sdk import WorkspaceClient

    from deep_research.config import ConfigError

    client_id = os.environ.get("DATABRICKS_CLIENT_ID", "")
    client_secret = os.environ.get("DATABRICKS_CLIENT_SECRET", "")
    token = getattr(config, "databricks_token", "") or ""

    kwargs: dict[str, Any] = {"host": config.databricks_host}
    if client_id and client_secret:
        kwargs.update(
            client_id=client_id,
            client_secret=client_secret,
        )
        logger.info(
            "Using Databricks OAuth client credentials for authentication"
        )
    elif token:
        kwargs["token"] = token
        logger.info(
            "Using DATABRICKS_TOKEN for Databricks authentication"
        )
    else:
        raise ConfigError(
            "Configure DATABRICKS_TOKEN, or DATABRICKS_CLIENT_ID and "
            "DATABRICKS_CLIENT_SECRET, for Databricks authentication"
        )

    try:
        return WorkspaceClient(**kwargs)
    except Exception as exc:
        raise ConfigError(
            f"Failed to initialize Databricks WorkspaceClient: {exc}"
        ) from exc


def _get_workspace_oauth_token(workspace_client: Any) -> str:
    """Resolve the current OAuth bearer token for the MCP manager."""

    from deep_research.config import ConfigError

    try:
        headers = workspace_client.config.authenticate()
        authorization = (
            headers.get("Authorization")
            or headers.get("authorization")
            or ""
        )
        scheme, separator, token = authorization.partition(" ")
        if (
            not separator
            or scheme.lower() != "bearer"
            or not token.strip()
        ):
            raise ConfigError(
                "Databricks SDK authentication did not return a bearer token"
            )
        return token.strip()
    except ConfigError:
        raise
    except Exception as exc:
        raise ConfigError(
            f"Failed to acquire Databricks OAuth token: {exc}"
        ) from exc


def _create_model(
    config: Any,
    http_client: httpx.AsyncClient,
    endpoint: str | None = None,
) -> Any:
    """Create a Pydantic AI Databricks serving-endpoint model."""

    from deep_research.config import ConfigError
    from deep_research_pydantic.agents import build_model

    endpoint_name = endpoint or config.llm_endpoint
    try:
        model = build_model(
            host=config.databricks_host,
            endpoint=endpoint_name,
            http_client=http_client,
        )
        logger.info("LLM model initialized: %s", endpoint_name)
        return model
    except Exception as exc:
        raise ConfigError(
            f"Failed to initialize LLM model: {exc}"
        ) from exc


def _create_critic_model(
    config: Any,
    http_client: httpx.AsyncClient,
) -> Any | None:
    """Create the critic model when it uses a distinct endpoint."""

    if config.critic_llm_endpoint == config.llm_endpoint:
        return None

    try:
        model = _create_model(
            config,
            http_client,
            endpoint=config.critic_llm_endpoint,
        )
        logger.info(
            "Critic model initialized: %s",
            config.critic_llm_endpoint,
        )
        return model
    except Exception as exc:
        logger.warning(
            "Failed to initialize critic model: %s "
            "— falling back to worker",
            exc,
        )
        return None


def _create_service_agents(model: Any) -> tuple[Agent, Agent]:
    """Create the quick-reply and pre-pipeline clarification agents."""

    from deep_research.prompts import CLARIFIER_SYSTEM
    from deep_research_pydantic.schemas import ClarifierOutput

    quick_reply_agent = Agent(
        model,
        output_type=str,
        name="quick_reply",
    )
    clarifier_agent = Agent(
        model,
        output_type=ClarifierOutput,
        instructions=CLARIFIER_SYSTEM,
        name="pre_graph_clarifier",
    )
    return quick_reply_agent, clarifier_agent


def create_production_app():
    """Create the FastAPI app with production dependencies wired up."""

    from deep_research.config import load_app_config
    from deep_research.mcp_client import MCPClientManager
    from deep_research.tracing import configure_tracing
    from deep_research_pydantic.api.app import create_app
    from deep_research_pydantic.runner import build_research_pipeline

    logger.info("Starting application initialization...")
    logger.info("CWD: %s", os.getcwd())
    logger.info("__file__: %s", __file__)
    logger.info(
        "PYTHONPATH: %s",
        os.environ.get("PYTHONPATH", "not set"),
    )
    logger.info(
        "DATABRICKS_HOST set: %s",
        bool(os.environ.get("DATABRICKS_HOST")),
    )
    logger.info(
        "DATABRICKS_CLIENT_ID set: %s",
        bool(os.environ.get("DATABRICKS_CLIENT_ID")),
    )
    logger.info(
        "DATABRICKS_CLIENT_SECRET set: %s",
        bool(os.environ.get("DATABRICKS_CLIENT_SECRET")),
    )

    config = load_app_config()
    logger.info("Configuration loaded successfully")
    logger.info("LLM endpoint: %s", config.llm_endpoint)
    logger.info("Managed servers: %d", len(config.managed_servers))

    try:
        experiment_name = os.environ.get(
            "MLFLOW_EXPERIMENT_NAME",
            "deep-research-agent",
        )
        configure_tracing(experiment_name=experiment_name)
    except Exception:
        logger.warning(
            "MLflow tracing setup failed — continuing without tracing",
            exc_info=True,
        )

    workspace_client = _create_workspace_client(config)
    oauth_token = _get_workspace_oauth_token(workspace_client)
    logger.info(
        "Databricks OAuth authentication initialized successfully"
    )

    model_http_client = httpx.AsyncClient(
        auth=WorkspaceOAuthAuth(workspace_client),
    )

    try:
        all_servers = _build_server_map(config)
        mcp_manager = MCPClientManager(
            all_servers,
            token=oauth_token,
        )
        logger.info(
            "MCP client configured with %d server(s)",
            len(all_servers),
        )

        model = _create_model(config, model_http_client)
        critic_model = _create_critic_model(
            config,
            model_http_client,
        )
        quick_reply_agent, clarifier_agent = _create_service_agents(
            model
        )

        graph = build_research_pipeline(
            model=model,
            mcp_manager=mcp_manager,
            critic_model=critic_model,
        )
        app = create_app()
    except Exception:
        _close_http_client_after_initialization_failure(
            model_http_client
        )
        raise

    app.state.graph = graph
    app.state.model = model
    app.state.quick_reply_agent = quick_reply_agent
    app.state.clarifier_agent = clarifier_agent
    app.state.mcp_manager = mcp_manager
    app.state.config = config
    app.state.workspace_client = workspace_client
    app.state.model_http_client = model_http_client

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
    async def _disconnect_dependencies():
        try:
            await mcp_manager.disconnect_all()
        finally:
            await model_http_client.aclose()

    @app.get("/debug/paths")
    async def debug_paths():
        cwd = os.getcwd()
        try:
            cwd_contents = sorted(os.listdir(cwd))
        except Exception as exc:
            cwd_contents = [f"error: {exc}"]

        ui_candidates: dict[str, dict[str, Any]] = {}
        for label, path_value in [
            (
                "__file__ based",
                str(
                    Path(__file__).parent.parent.parent
                    / "ui"
                    / "dist"
                ),
            ),
            ("cwd based", str(Path(cwd) / "ui" / "dist")),
            (
                "SOURCE_CODE_PATH",
                str(
                    Path(
                        os.environ.get(
                            "DATABRICKS_SOURCE_CODE_PATH",
                            "",
                        )
                    )
                    / "ui"
                    / "dist"
                ),
            ),
        ]:
            path = Path(path_value)
            ui_candidates[label] = {
                "path": path_value,
                "exists": path.exists(),
                "has_index": (
                    (path / "index.html").exists()
                    if path.exists()
                    else False
                ),
            }

        return {
            "cwd": cwd,
            "cwd_contents": cwd_contents,
            "__file__": __file__,
            "app_initialized": True,
            "ui_candidates": ui_candidates,
            "env_SOURCE_CODE_PATH": os.environ.get(
                "DATABRICKS_SOURCE_CODE_PATH",
                "not set",
            ),
        }

    import json as _json

    from starlette.responses import (
        StreamingResponse as _StreamingResponse,
    )

    @app.get("/debug/sse-test")
    async def sse_test():
        async def generate():
            for sequence in range(5):
                event = {
                    "seq": sequence,
                    "msg": f"event {sequence}",
                }
                yield f"data: {_json.dumps(event)}\n\n"
                await asyncio.sleep(1)
            yield (
                "data: "
                f"{_json.dumps({'seq': 5, 'msg': 'done'})}"
                "\n\n"
            )

        return _StreamingResponse(
            generate(),
            media_type="text/event-stream",
            headers={
                "X-Accel-Buffering": "no",
                "Cache-Control": "no-cache",
                "Connection": "keep-alive",
            },
        )

    candidates = [
        Path(__file__).parent.parent.parent / "ui" / "dist",
        Path(os.getcwd()) / "ui" / "dist",
    ]
    source_code_path = os.environ.get(
        "DATABRICKS_SOURCE_CODE_PATH",
        "",
    )
    if source_code_path:
        candidates.insert(
            0,
            Path(source_code_path) / "ui" / "dist",
        )

    actual_ui_dist: Path | None = None
    for candidate in candidates:
        exists = candidate.exists()
        has_index = exists and (candidate / "index.html").exists()
        logger.info(
            "Checking for UI at: %s (exists: %s, has_index: %s)",
            candidate,
            exists,
            has_index,
        )
        if has_index:
            actual_ui_dist = candidate
            break

    if actual_ui_dist:
        from fastapi.staticfiles import StaticFiles

        app.mount(
            "/",
            StaticFiles(
                directory=str(actual_ui_dist),
                html=True,
            ),
            name="ui",
        )
        logger.info("Serving React UI from %s", actual_ui_dist)
    else:
        from fastapi.responses import JSONResponse

        logger.info("No UI build found — API-only mode")

        @app.get("/")
        async def root():
            return JSONResponse(
                {
                    "status": "ok",
                    "message": (
                        "Deep Research Agent API is running. "
                        "UI not found."
                    ),
                    "endpoints": {
                        "health": "/health",
                        "tools": "/api/tools",
                        "research": "POST /api/research",
                        "debug": "/debug/paths",
                    },
                }
            )

    logger.info("Application initialization complete")
    return app


_startup_error: str | None = None
try:
    app = create_production_app()
except Exception:
    import traceback

    _startup_error = traceback.format_exc()
    logger.exception(
        "Failed to create app "
        "— starting with health-check-only fallback"
    )

    from fastapi import FastAPI

    app = FastAPI(
        title="Deep Research Agent (startup failed)"
    )

    @app.get("/health")
    async def health():
        return {
            "status": "error",
            "message": "App failed to initialize",
        }

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

    port = int(os.environ.get("DATABRICKS_APP_PORT", "8000"))
    uvicorn.run(
        app,
        host="0.0.0.0",
        port=port,
    )
