"""Tests for FastAPI backend."""

import asyncio
import pytest
from unittest.mock import AsyncMock, MagicMock
from httpx import ASGITransport, AsyncClient

from deep_research.api.app import create_app, _run_graph
from deep_research.api.jobs import JobManager


@pytest.fixture
def app():
    return create_app(use_mocks=True)


@pytest.fixture
async def client(app):
    transport = ASGITransport(app=app)
    async with AsyncClient(transport=transport, base_url="http://test") as c:
        yield c


class _MockGraph:
    """Mock graph that supports both ainvoke and astream."""

    def __init__(self, final_output="Test research result"):
        self._final_output = final_output

    async def ainvoke(self, state):
        return {"final_output": self._final_output}

    async def astream(self, state, stream_mode="updates"):
        yield {"verifier": {"final_output": self._final_output}}


@pytest.fixture
def app_with_graph():
    """App with a mock graph that returns final_output."""
    a = create_app()
    a.state.graph = _MockGraph()
    a.state.mcp_manager = None
    a.state.config = None
    return a


@pytest.fixture
async def graph_client(app_with_graph):
    transport = ASGITransport(app=app_with_graph)
    async with AsyncClient(transport=transport, base_url="http://test") as c:
        yield c


async def test_health_check(client):
    response = await client.get("/health")
    assert response.status_code == 200
    assert response.json()["status"] == "ok"


async def test_list_tools(client):
    response = await client.get("/api/tools")
    assert response.status_code == 200
    tools = response.json()
    assert isinstance(tools, list)
    assert len(tools) == 2
    assert tools[0]["name"] == "genie_sales"


async def test_list_tools_from_mcp_manager():
    """Tools endpoint reads from app.state.mcp_manager in production mode."""
    from deep_research.config import MCPServerConfig

    a = create_app(use_mocks=False)
    mock_manager = MagicMock()
    mock_manager.get_available_servers.return_value = {
        "genie_sales": MCPServerConfig(
            name="genie_sales",
            url="mock://genie",
            display_name="Sales Genie",
            server_kind="managed",
            risk_tier="safe",
        ),
    }
    a.state.mcp_manager = mock_manager

    transport = ASGITransport(app=a)
    async with AsyncClient(transport=transport, base_url="http://test") as c:
        response = await c.get("/api/tools")
        assert response.status_code == 200
        tools = response.json()
        assert len(tools) == 1
        assert tools[0]["name"] == "genie_sales"
        assert tools[0]["display_name"] == "Sales Genie"


async def test_submit_research_no_graph(client):
    """Without a graph on app.state, research returns failed."""
    response = await client.post("/api/research", json={
        "query": "What was Q3 revenue?",
        "tools": ["genie_sales"],
        "output_mode": "chat",
    })
    assert response.status_code == 200
    data = response.json()
    assert "job_id" in data
    assert data["status"] == "failed"


async def test_submit_research_with_graph(graph_client):
    """With a graph, research returns pending and runs in background."""
    response = await graph_client.post("/api/research", json={
        "query": "What was Q3 revenue?",
        "tools": [],
        "output_mode": "chat",
    })
    assert response.status_code == 200
    data = response.json()
    assert "job_id" in data
    assert data["status"] == "pending"

    # Let the background task complete
    await asyncio.sleep(0.1)

    status = await graph_client.get(f"/api/research/{data['job_id']}")
    assert status.json()["status"] == "completed"
    assert status.json()["result"] == "Test research result"


async def test_get_research_status(client):
    submit = await client.post("/api/research", json={
        "query": "test",
        "tools": [],
    })
    job_id = submit.json()["job_id"]
    status = await client.get(f"/api/research/{job_id}")
    assert status.status_code == 200
    # Without graph, status is "failed"
    assert status.json()["status"] == "failed"


async def test_cancel_research(graph_client):
    """Cancel a job before it completes."""
    # Use a graph that hangs so we can cancel before completion
    app = graph_client._transport.app  # type: ignore[attr-defined]
    hang_event = asyncio.Event()

    class _SlowGraph:
        async def astream(self, state, stream_mode="updates"):
            await hang_event.wait()
            yield {"verifier": {"final_output": "done"}}

    app.state.graph = _SlowGraph()

    submit = await graph_client.post("/api/research", json={
        "query": "test",
        "tools": [],
    })
    job_id = submit.json()["job_id"]
    cancel = await graph_client.delete(f"/api/research/{job_id}")
    assert cancel.status_code == 200
    status = await graph_client.get(f"/api/research/{job_id}")
    assert status.json()["status"] == "cancelled"
    hang_event.set()  # Clean up


async def test_unknown_job_404(client):
    response = await client.get("/api/research/nonexistent")
    assert response.status_code == 404


async def test_submit_feedback(client):
    response = await client.post("/api/feedback", json={
        "query_id": "test-123",
        "rating": "thumbs_up",
        "comment": "Great!",
    })
    assert response.status_code == 200
    assert response.json()["rating"] == "thumbs_up"


async def test_feedback_invalid_rating(client):
    response = await client.post("/api/feedback", json={
        "query_id": "test",
        "rating": "invalid",
    })
    assert response.status_code == 422


async def test_empty_query_rejected(client):
    response = await client.post("/api/research", json={
        "query": "",
        "tools": [],
    })
    assert response.status_code == 422


async def test_run_graph_failure():
    """Graph exception should mark job as failed."""

    class _FailGraph:
        async def astream(self, state, stream_mode="updates"):
            raise RuntimeError("LLM call failed")
            yield  # make it an async generator  # noqa: unreachable

    jm = JobManager()
    job_id = jm.create_job(query="test", tools=[])

    await _run_graph(_FailGraph(), jm, job_id, {})
    assert jm.get_status(job_id).state == "failed"


async def test_run_graph_empty_output():
    """Graph returning empty final_output should still complete."""
    jm = JobManager()
    job_id = jm.create_job(query="test", tools=[])

    await _run_graph(_MockGraph(final_output=""), jm, job_id, {})
    status = jm.get_status(job_id)
    assert status.state == "completed"
    assert status.result == "Research completed but produced no output."


async def test_job_event_queue():
    """JobManager creates an event queue per job and events can be pushed/consumed."""
    jm = JobManager()
    job_id = jm.create_job(query="test", tools=[])
    queue = jm.get_event_queue(job_id)
    assert queue is not None

    jm.push_event(job_id, {"type": "node_started", "node": "clarifier"})
    event = queue.get_nowait()
    assert event["type"] == "node_started"
    assert event["node"] == "clarifier"


async def test_job_event_queue_unknown_job():
    """get_event_queue raises KeyError for unknown job."""
    jm = JobManager()
    with pytest.raises(KeyError):
        jm.get_event_queue("nonexistent")
