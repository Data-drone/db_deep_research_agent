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

    async def astream(self, state, stream_mode="updates", config=None):
        yield {"synthesizer": {"final_output": self._final_output}}
        yield {"verifier": {"verification_result": None}}


@pytest.fixture
def app_with_graph():
    """App with a mock graph that returns final_output."""
    a = create_app()
    a.state.graph = _MockGraph()
    a.state.model = _MockModel()
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
    assert tools[0]["name"] == "genie_aus_market"


async def test_list_tools_from_mcp_manager():
    """Tools endpoint reads from app.state.mcp_manager in production mode."""
    from deep_research.config import MCPServerConfig

    a = create_app(use_mocks=False)
    mock_manager = MagicMock()
    mock_manager.get_available_servers.return_value = {
        "genie_aus_market": MCPServerConfig(
            name="genie_aus_market",
            url="mock://genie",
            display_name="Australian Economic & Market Data (Genie)",
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
        assert tools[0]["name"] == "genie_aus_market"
        assert tools[0]["display_name"] == "Australian Economic & Market Data (Genie)"


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
    """With a graph, research mode returns pending and runs in background."""
    response = await graph_client.post("/api/research", json={
        "query": "What was Q3 revenue?",
        "tools": [],
        "output_mode": "chat",
        "response_mode": "research",
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
        async def astream(self, state, stream_mode="updates", config=None):
            await hang_event.wait()
            yield {"synthesizer": {"final_output": "done"}}
            yield {"verifier": {"verification_result": None}}

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
        async def astream(self, state, stream_mode="updates", config=None):
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


async def test_sse_stream_endpoint(graph_client):
    """SSE endpoint streams node events then completes."""
    # Submit a job
    response = await graph_client.post("/api/research", json={
        "query": "Test SSE",
        "tools": [],
        "output_mode": "chat",
    })
    job_id = response.json()["job_id"]

    # Let background task complete
    await asyncio.sleep(0.2)

    # Read SSE stream
    response = await graph_client.get(
        f"/api/research/{job_id}/stream",
        headers={"Accept": "text/event-stream"},
    )
    assert response.status_code == 200
    assert "text/event-stream" in response.headers["content-type"]

    # Parse SSE events from response body
    lines = response.text.strip().split("\n")
    data_lines = [l.removeprefix("data: ") for l in lines if l.startswith("data: ")]
    assert len(data_lines) >= 1  # at least the completed event

    import json as _json
    last_event = _json.loads(data_lines[-1])
    assert last_event["type"] in ("completed", "failed")


async def test_sse_stream_unknown_job(client):
    """SSE stream for unknown job returns 404."""
    response = await client.get("/api/research/nonexistent/stream")
    assert response.status_code == 404


async def test_run_graph_injects_job_context():
    """_run_graph adds _job_manager and _job_id to state before calling graph."""
    jm = JobManager()
    job_id = jm.create_job(query="test", tools=[])
    captured_state = {}

    class _CaptureGraph:
        async def astream(self, state, stream_mode="updates", config=None):
            captured_state.update(state)
            yield {"synthesizer": {"final_output": "done"}}
            yield {"verifier": {"verification_result": None}}

    await _run_graph(_CaptureGraph(), jm, job_id, {"user_query": "test"})
    assert captured_state["_job_manager"] is jm
    assert captured_state["_job_id"] == job_id


async def test_run_graph_pushes_events():
    """_run_graph pushes node_started events and a completed event to the queue."""
    jm = JobManager()
    job_id = jm.create_job(query="test", tools=[])

    class _MultiNodeGraph:
        async def astream(self, state, stream_mode="updates", config=None):
            yield {"clarifier": {"clarified_query": "test"}}
            yield {"planner": {"research_plan": []}}
            yield {"synthesizer": {"final_output": "done"}}
            yield {"verifier": {"verification_result": None}}

    await _run_graph(_MultiNodeGraph(), jm, job_id, {})

    events = []
    queue = jm.get_event_queue(job_id)
    while not queue.empty():
        events.append(queue.get_nowait())

    node_events = [e for e in events if e["type"] == "node_started"]
    assert len(node_events) == 4
    assert node_events[0]["node"] == "clarifier"
    assert node_events[1]["node"] == "planner"
    assert node_events[2]["node"] == "synthesizer"
    assert node_events[3]["node"] == "verifier"

    terminal = [e for e in events if e["type"] in ("completed", "failed")]
    assert len(terminal) == 1
    assert terminal[0]["type"] == "completed"
    assert terminal[0]["result"] == "done"


# ── Quick reply tests ──


class _MockModel:
    """Mock LLM model for quick reply tests."""

    def __init__(self, reply="Quick answer"):
        self._reply = reply

    async def ainvoke(self, messages):
        from types import SimpleNamespace
        return SimpleNamespace(content=self._reply)


class _MockMCPManager:
    """Mock MCP manager for quick reply tests."""

    async def call_tool(self, server_name, tool_name, arguments):
        return {"result": f"Data from {server_name}"}


@pytest.fixture
def app_with_model():
    """App with a mock model for quick reply."""
    a = create_app()
    a.state.graph = _MockGraph()
    a.state.model = _MockModel()
    a.state.mcp_manager = _MockMCPManager()
    a.state.config = None
    return a


@pytest.fixture
async def quick_client(app_with_model):
    transport = ASGITransport(app=app_with_model)
    async with AsyncClient(transport=transport, base_url="http://test") as c:
        yield c


async def test_quick_reply_completes(quick_client):
    """Quick reply mode returns a response without running the graph."""
    response = await quick_client.post("/api/research", json={
        "query": "What was CBA's closing price?",
        "tools": ["genie_aus_market"],
        "response_mode": "quick",
    })
    assert response.status_code == 200
    data = response.json()
    assert data["status"] == "pending"
    assert "session_id" in data

    # Let background task complete
    await asyncio.sleep(0.2)

    status = await quick_client.get(f"/api/research/{data['job_id']}")
    assert status.json()["status"] == "completed"
    assert "Quick answer" in status.json()["result"]


async def test_quick_reply_no_tools(quick_client):
    """Quick reply with no tools still works (direct LLM call)."""
    response = await quick_client.post("/api/research", json={
        "query": "What is inflation?",
        "tools": [],
        "response_mode": "quick",
    })
    data = response.json()
    assert data["status"] == "pending"

    await asyncio.sleep(0.2)

    status = await quick_client.get(f"/api/research/{data['job_id']}")
    assert status.json()["status"] == "completed"


async def test_research_mode_uses_graph(quick_client):
    """Research mode still goes through the graph."""
    response = await quick_client.post("/api/research", json={
        "query": "Deep analysis needed",
        "tools": [],
        "response_mode": "research",
        "output_mode": "chat",
    })
    data = response.json()
    assert data["status"] == "pending"

    await asyncio.sleep(0.2)

    status = await quick_client.get(f"/api/research/{data['job_id']}")
    assert status.json()["status"] == "completed"
    assert status.json()["result"] == "Test research result"


async def test_default_response_mode_is_quick(quick_client):
    """Default response_mode is 'quick' when not specified."""
    response = await quick_client.post("/api/research", json={
        "query": "Quick question",
        "tools": [],
    })
    data = response.json()
    assert data["status"] == "pending"

    await asyncio.sleep(0.2)

    status = await quick_client.get(f"/api/research/{data['job_id']}")
    # Should complete via quick reply (model returns "Quick answer")
    assert status.json()["status"] == "completed"
    assert "Quick answer" in status.json()["result"]


async def test_quick_reply_session_continuity(quick_client):
    """Quick reply stores session_id and supports follow-ups."""
    # First query
    r1 = await quick_client.post("/api/research", json={
        "query": "First question",
        "tools": [],
        "response_mode": "quick",
    })
    session_id = r1.json()["session_id"]
    await asyncio.sleep(0.2)

    # Follow-up with same session
    r2 = await quick_client.post("/api/research", json={
        "query": "Follow up",
        "tools": [],
        "response_mode": "quick",
        "session_id": session_id,
    })
    assert r2.json()["session_id"] == session_id
    await asyncio.sleep(0.2)

    status = await quick_client.get(f"/api/research/{r2.json()['job_id']}")
    assert status.json()["status"] == "completed"


# ── Clarification endpoint tests ──


async def test_clarify_unknown_job(client):
    """Clarify endpoint returns 404 for unknown job."""
    response = await client.post(
        "/api/research/nonexistent/clarify",
        json={"clarification_id": "clr-123", "answer": "test"},
    )
    assert response.status_code == 404


async def test_clarify_no_pending(quick_client):
    """Clarify returns 400 if job has no pending clarification."""
    res = await quick_client.post("/api/research", json={
        "query": "test", "tools": [], "response_mode": "quick"
    })
    job_id = res.json()["job_id"]
    await asyncio.sleep(0.1)

    response = await quick_client.post(
        f"/api/research/{job_id}/clarify",
        json={"clarification_id": "clr-123", "answer": "option A"},
    )
    assert response.status_code == 400


async def test_clarify_blank_answer_rejected(client):
    """Clarify endpoint rejects blank answers."""
    response = await client.post(
        "/api/research/nonexistent/clarify",
        json={"clarification_id": "clr-123", "answer": "   "},
    )
    assert response.status_code == 422


async def test_job_status_no_clarification(quick_client):
    """Job status without clarification has no pending_clarification field."""
    res = await quick_client.post("/api/research", json={
        "query": "test", "tools": [], "response_mode": "quick"
    })
    await asyncio.sleep(0.2)
    status = await quick_client.get(f"/api/research/{res.json()['job_id']}")
    assert "pending_clarification" not in status.json()
