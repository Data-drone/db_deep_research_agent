"""API-level behaviour added while closing out the review findings.

Each test here exists because a client that was not attached to the SSE stream
at exactly the right moment was being told something false: that a completed job
had been cancelled, that a report had no fabricated citations, or nothing at all
about a clarification it was being asked for.
"""

import asyncio
import json

import pytest
from httpx import ASGITransport, AsyncClient

from deep_research.api.app import _recursion_limit, create_app
from deep_research.models import Budget


@pytest.fixture
def app():
    return create_app(use_mocks=True)


@pytest.fixture
async def client(app):
    transport = ASGITransport(app=app)
    async with AsyncClient(transport=transport, base_url="http://test") as c:
        yield c


def _make_job(app):
    """Register a job directly rather than POSTing one.

    A POST also starts the runner, which fails immediately under ``use_mocks``
    (no graph); the job would then be terminal and every state we want to set up
    here would be refused. These tests are about the endpoints, not the runner.
    """
    return app.state.job_manager.create_job(query="q", tools=[])


async def _read_frames(client, job_id, timeout=5.0):
    """Drain the SSE stream to its terminal frame, with a hard timeout.

    The generator loops on keepalives until the job goes terminal, so a test that
    forgets to arrange an ending would otherwise hang rather than fail.
    """
    async def _drain():
        frames = []
        async with client.stream("GET", f"/api/research/{job_id}/stream") as resp:
            async for line in resp.aiter_lines():
                if line.startswith("data: "):
                    frames.append(json.loads(line[len("data: "):]))
        return frames

    return await asyncio.wait_for(_drain(), timeout)


# ── DELETE on a job that already finished ────────────────────────────────────

async def test_delete_on_a_completed_job_reports_completed(client, app):
    """cancel_job is a no-op on a terminal job, so answering "cancelled" told the
    client its finished result had been thrown away."""
    job_id = _make_job(app)
    app.state.job_manager.update_state(job_id, "completed", result="the report")

    resp = await client.delete(f"/api/research/{job_id}")
    assert resp.status_code == 200
    assert resp.json()["status"] == "completed"

    queue = app.state.job_manager.get_event_queue(job_id)
    events = []
    while not queue.empty():
        events.append(queue.get_nowait())
    assert not any(e.get("type") == "cancelled" for e in events)
    assert app.state.job_manager.get_status(job_id).result == "the report"


async def test_delete_on_a_running_job_still_cancels(client, app):
    job_id = _make_job(app)
    app.state.job_manager.update_state(job_id, "running")

    resp = await client.delete(f"/api/research/{job_id}")
    assert resp.json()["status"] == "cancelled"

    queue = app.state.job_manager.get_event_queue(job_id)
    events = []
    while not queue.empty():
        events.append(queue.get_nowait())
    assert any(e.get("type") == "cancelled" for e in events)


# ── Fabricated citations survive a reconnect ─────────────────────────────────

async def test_status_endpoint_exposes_unverified_citations(client, app):
    job_id = _make_job(app)
    status = app.state.job_manager.get_status(job_id)
    status.unverified_citations = ["E7", "E9"]
    status.state = "completed"

    body = (await client.get(f"/api/research/{job_id}")).json()
    assert body["unverified_citations"] == ["E7", "E9"]


async def test_terminal_stream_frame_carries_unverified_citations(client, app):
    """A client that reconnects after the fact must still be told the report
    cites evidence that does not exist."""
    job_id = _make_job(app)
    status = app.state.job_manager.get_status(job_id)
    status.unverified_citations = ["E9"]
    status.token_usage = {"input": 10, "output": 5, "scope": "job"}
    app.state.job_manager.update_state(job_id, "completed", result="r")

    frames = await _read_frames(client, job_id)
    assert frames[0]["unverified_citations"] == ["E9"]
    assert frames[0]["token_usage"]["scope"] == "job"


# ── Reconnecting mid-clarification ───────────────────────────────────────────

async def test_stream_replays_a_pending_clarification(client, app):
    """Queue events are consumed once. Without a replay, a client whose stream
    dropped after the prompt was raised sees nothing until the timeout, and the
    run is parked the whole time."""
    job_id = _make_job(app)
    app.state.job_manager.update_state(job_id, "running")
    store = app.state.clarification_store
    clr = store.create(job_id, "Which market?", ["AU", "US"], "AU")
    # The generator only returns on a terminal frame; queue one so the request
    # ends instead of sitting in the keepalive loop for the rest of the suite.
    app.state.job_manager.push_event(job_id, {"type": "completed", "result": "r"})

    frames = await _read_frames(client, job_id)

    assert frames[0]["type"] == "clarification_needed"
    assert frames[0]["clarification_id"] == clr.clarification_id
    assert frames[0]["options"] == ["AU", "US"]
    assert frames[1]["type"] == "completed"


async def test_stream_does_not_replay_an_answered_clarification(client, app):
    job_id = _make_job(app)
    store = app.state.clarification_store
    clr = store.create(job_id, "Which market?", ["AU", "US"], "AU")
    store.submit_answer(job_id, clr.clarification_id, "AU")
    app.state.job_manager.update_state(job_id, "completed", result="r")

    frames = await _read_frames(client, job_id)
    assert frames[0]["type"] == "completed"


# ── Superstep budget ─────────────────────────────────────────────────────────

def test_recursion_limit_never_drops_below_the_old_fixed_value():
    assert _recursion_limit(Budget()) >= 50
    assert _recursion_limit(None) >= 50


def test_recursion_limit_grows_with_iterations_and_revisions():
    base = _recursion_limit(Budget(max_iterations=5, max_verification_attempts=1))
    more_iters = _recursion_limit(Budget(max_iterations=12, max_verification_attempts=1))
    more_revs = _recursion_limit(Budget(max_iterations=5, max_verification_attempts=3))
    assert more_iters > base
    assert more_revs > base


# ── Clarifier usage is part of the job total ─────────────────────────────────

async def test_clarifier_usage_is_folded_into_the_run_total(monkeypatch, app):
    """The clarifier runs outside the graph, so nothing else picks up its usage.
    Leaving it out made a `"scope": "job"` total quietly exclude a paid call."""
    import deep_research.nodes.clarifier as clarifier_mod
    from deep_research.api.app import _run_research_with_clarification

    async def fake_clarifier(state, model=None):
        return {
            "needs_clarification": False,
            "clarified_query": "q",
            "_token_usage": {"input": 120, "output": 30},
        }

    monkeypatch.setattr(clarifier_mod, "clarifier_node", fake_clarifier)

    seen = {}

    async def fake_run_graph(graph, job_manager, job_id, state, *args, **kwargs):
        seen["usage"] = state.get("_token_usage")

    monkeypatch.setattr("deep_research.api.app._run_graph", fake_run_graph)

    job_manager = app.state.job_manager
    job_id = _make_job(app)
    state = {"user_query": "q", "_token_usage": {"input": 5, "output": 1}}

    await _run_research_with_clarification(
        model=object(),
        graph=object(),
        clarification_store=app.state.clarification_store,
        job_manager=job_manager,
        job_id=job_id,
        initial_state=state,
        session_manager=app.state.session_manager,
        session_id="s1",
    )

    assert seen["usage"] == {"input": 125, "output": 31}
