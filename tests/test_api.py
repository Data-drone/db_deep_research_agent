"""Tests for FastAPI backend."""

import pytest
from httpx import ASGITransport, AsyncClient

from deep_research.api.app import create_app


@pytest.fixture
def app():
    return create_app(use_mocks=True)


@pytest.fixture
async def client(app):
    transport = ASGITransport(app=app)
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


async def test_submit_research(client):
    response = await client.post("/api/research", json={
        "query": "What was Q3 revenue?",
        "tools": ["genie_sales"],
        "output_mode": "chat",
    })
    assert response.status_code == 200
    data = response.json()
    assert "job_id" in data
    assert data["status"] == "pending"


async def test_get_research_status(client):
    submit = await client.post("/api/research", json={
        "query": "test",
        "tools": [],
    })
    job_id = submit.json()["job_id"]
    status = await client.get(f"/api/research/{job_id}")
    assert status.status_code == 200
    assert status.json()["status"] == "pending"


async def test_cancel_research(client):
    submit = await client.post("/api/research", json={
        "query": "test",
        "tools": [],
    })
    job_id = submit.json()["job_id"]
    cancel = await client.delete(f"/api/research/{job_id}")
    assert cancel.status_code == 200
    status = await client.get(f"/api/research/{job_id}")
    assert status.json()["status"] == "cancelled"


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
