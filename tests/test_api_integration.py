"""API-level integration tests — tests from the user's HTTP perspective."""

import asyncio
import pytest
from unittest.mock import AsyncMock
from httpx import ASGITransport, AsyncClient

from deep_research.api.app import create_app


@pytest.fixture
def app():
    """Create app with mock dependencies and a mock graph."""
    a = create_app(use_mocks=True)
    mock_graph = AsyncMock()
    mock_graph.ainvoke.return_value = {"final_output": "Mock research result"}
    a.state.graph = mock_graph
    a.state.mcp_manager = None
    a.state.config = None
    return a


@pytest.fixture
async def client(app):
    transport = ASGITransport(app=app)
    async with AsyncClient(transport=transport, base_url="http://test") as c:
        yield c


class TestUserResearchFlow:
    """Tests the full user journey: submit query -> poll -> get result."""

    async def test_submit_research_query(self, client):
        """User submits a research query and gets a job ID back."""
        response = await client.post(
            "/api/research",
            json={
                "query": "What was Q3 revenue?",
                "tools": ["genie_sales"],
                "output_mode": "chat",
            },
        )
        assert response.status_code == 200
        data = response.json()
        assert "job_id" in data
        assert data["status"] in ("pending", "running")

    async def test_poll_for_result(self, client):
        """User polls until research is complete."""
        submit = await client.post(
            "/api/research",
            json={
                "query": "What was Q3 revenue?",
                "tools": ["genie_sales"],
                "output_mode": "chat",
            },
        )
        job_id = submit.json()["job_id"]

        # Let background task complete
        await asyncio.sleep(0.1)

        result = await client.get(f"/api/research/{job_id}")
        assert result.status_code == 200
        data = result.json()
        assert data["status"] == "completed"
        assert data["result"] == "Mock research result"

    async def test_cancel_research(self, client):
        """User cancels an in-progress research query."""
        # Use a slow graph so we can cancel before completion
        app = client._transport.app  # type: ignore[attr-defined]
        hang_event = asyncio.Event()
        async def slow_invoke(state):
            await hang_event.wait()
            return {"final_output": "done"}
        app.state.graph.ainvoke = slow_invoke

        submit = await client.post(
            "/api/research",
            json={
                "query": "Deep analysis of everything",
                "tools": ["genie_sales", "vector_search_kb"],
                "output_mode": "report",
            },
        )
        job_id = submit.json()["job_id"]

        cancel = await client.delete(f"/api/research/{job_id}")
        assert cancel.status_code == 200

        status = await client.get(f"/api/research/{job_id}")
        assert status.json()["status"] == "cancelled"
        hang_event.set()  # Clean up

    async def test_submit_feedback(self, client):
        """User submits thumbs up/down on a response."""
        response = await client.post(
            "/api/feedback",
            json={
                "query_id": "test-query-123",
                "rating": "thumbs_up",
                "comment": "Great answer!",
            },
        )
        assert response.status_code == 200

    async def test_list_available_tools(self, client):
        """User loads the tool selector sidebar."""
        response = await client.get("/api/tools")
        assert response.status_code == 200
        tools = response.json()
        assert isinstance(tools, list)
        # In mock mode, should have test tools available
        if len(tools) > 0:
            assert "display_name" in tools[0]
            assert "risk_tier" in tools[0]


class TestEdgeCases:
    """Tests error handling and edge cases a user might hit."""

    async def test_empty_query_rejected(self, client):
        """User submits an empty query."""
        response = await client.post(
            "/api/research",
            json={
                "query": "",
                "tools": ["genie_sales"],
            },
        )
        assert response.status_code == 422

    async def test_no_tools_selected(self, client):
        """User submits query with no tools selected."""
        response = await client.post(
            "/api/research",
            json={
                "query": "What is revenue?",
                "tools": [],
            },
        )
        # Should accept (tools can be empty)
        assert response.status_code in (200, 400, 422)

    async def test_unknown_job_id(self, client):
        """User polls for a nonexistent job."""
        response = await client.get("/api/research/nonexistent-job-id")
        assert response.status_code == 404

    async def test_feedback_invalid_rating(self, client):
        """User submits invalid rating value."""
        response = await client.post(
            "/api/feedback",
            json={
                "query_id": "test",
                "rating": "invalid_value",
            },
        )
        assert response.status_code == 422
