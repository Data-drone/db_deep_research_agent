"""Tests for real cancellation, task tracking and job lifecycle cleanup."""

import asyncio
from datetime import datetime, timedelta, timezone

import pytest

from deep_research.api.jobs import MAX_QUEUE_SIZE, JobManager


@pytest.fixture
def jm():
    return JobManager()


def drain(jm: JobManager, job_id: str) -> list[dict]:
    queue = jm.get_event_queue(job_id)
    events = []
    while not queue.empty():
        events.append(queue.get_nowait())
    return events


# ── Terminal states are final ────────────────────────────────────────────────

@pytest.mark.parametrize("terminal", ["completed", "cancelled", "failed"])
def test_terminal_state_cannot_be_walked_back(jm, terminal):
    """A runner that hasn't noticed the outcome must not resurrect the job."""
    job_id = jm.create_job(query="q", tools=[])
    jm.update_state(job_id, terminal)
    jm.update_state(job_id, "running")
    jm.update_state(job_id, "completed", result="should not appear")
    status = jm.get_status(job_id)
    assert status.state == terminal
    if terminal != "completed":
        assert status.result is None


def test_finished_at_is_stamped_once(jm):
    job_id = jm.create_job(query="q", tools=[])
    jm.update_state(job_id, "completed")
    first = jm.get_status(job_id).finished_at
    assert first is not None
    jm.update_state(job_id, "completed")
    assert jm.get_status(job_id).finished_at == first


def test_non_terminal_updates_still_apply(jm):
    job_id = jm.create_job(query="q", tools=[])
    jm.update_state(job_id, "running", current_node="planner")
    assert jm.get_status(job_id).current_node == "planner"
    assert jm.get_status(job_id).finished_at is None


# ── Cancellation actually stops the work ─────────────────────────────────────

@pytest.mark.asyncio
async def test_cancel_job_cancels_the_registered_task(jm):
    job_id = jm.create_job(query="q", tools=[])
    started = asyncio.Event()

    async def runner():
        started.set()
        await asyncio.sleep(60)  # would outlive the test if never cancelled

    task = asyncio.create_task(runner())
    jm.register_task(job_id, task)
    await started.wait()

    jm.cancel_job(job_id)
    with pytest.raises(asyncio.CancelledError):
        await task

    assert task.cancelled()
    assert jm.get_status(job_id).state == "cancelled"
    assert jm.is_cancelled(job_id)


@pytest.mark.asyncio
async def test_cancel_is_safe_when_no_task_was_registered(jm):
    """The pre-graph window: cancel arrives before any task exists."""
    job_id = jm.create_job(query="q", tools=[])
    jm.cancel_job(job_id)
    assert jm.is_cancelled(job_id)


@pytest.mark.asyncio
async def test_task_dying_silently_marks_the_job_failed(jm):
    """A runner that returns without a terminal state must not hang the client."""
    job_id = jm.create_job(query="q", tools=[])
    jm.update_state(job_id, "running")

    async def runner():
        return  # never reports an outcome

    task = asyncio.create_task(runner())
    jm.register_task(job_id, task)
    await task
    await asyncio.sleep(0)  # let the done callback run

    assert jm.get_status(job_id).state == "failed"
    assert [e["type"] for e in drain(jm, job_id)] == ["failed"]


@pytest.mark.asyncio
async def test_task_raising_marks_the_job_failed_with_the_reason(jm):
    job_id = jm.create_job(query="q", tools=[])
    jm.update_state(job_id, "running")

    async def runner():
        raise RuntimeError("genie exploded")

    task = asyncio.create_task(runner())
    jm.register_task(job_id, task)
    with pytest.raises(RuntimeError):
        await task
    await asyncio.sleep(0)

    status = jm.get_status(job_id)
    assert status.state == "failed"
    assert "genie exploded" in status.error


@pytest.mark.asyncio
async def test_done_callback_leaves_an_already_terminal_job_alone(jm):
    """The normal path: the runner reported "completed" itself."""
    job_id = jm.create_job(query="q", tools=[])

    async def runner():
        jm.update_state(job_id, "completed", result="done")

    task = asyncio.create_task(runner())
    jm.register_task(job_id, task)
    await task
    await asyncio.sleep(0)

    status = jm.get_status(job_id)
    assert status.state == "completed"
    assert status.result == "done"
    assert drain(jm, job_id) == []


@pytest.mark.asyncio
async def test_registered_task_is_released_when_done(jm):
    """The strong reference must not become a leak of its own."""
    job_id = jm.create_job(query="q", tools=[])

    async def runner():
        jm.update_state(job_id, "completed")

    task = asyncio.create_task(runner())
    jm.register_task(job_id, task)
    await task
    await asyncio.sleep(0)
    assert job_id not in jm._tasks


# ── Queue bounding ───────────────────────────────────────────────────────────

def test_queue_is_bounded(jm):
    job_id = jm.create_job(query="q", tools=[])
    assert jm.get_event_queue(job_id).maxsize == MAX_QUEUE_SIZE


def test_full_queue_drops_oldest_so_terminal_events_survive(jm):
    """A client that never attaches must still be able to learn the outcome."""
    job_id = jm.create_job(query="q", tools=[])
    for i in range(MAX_QUEUE_SIZE):
        jm.push_event(job_id, {"type": "token", "content": str(i)})
    jm.push_event(job_id, {"type": "completed", "result": "final"})

    events = drain(jm, job_id)
    assert len(events) == MAX_QUEUE_SIZE
    assert events[-1]["type"] == "completed"
    # The oldest token was evicted, not the newest event.
    assert events[0] == {"type": "token", "content": "1"}


def test_push_event_to_unknown_job_is_a_noop(jm):
    jm.push_event("job-does-not-exist", {"type": "completed"})  # must not raise


# ── TTL sweep ────────────────────────────────────────────────────────────────

def test_sweep_drops_only_expired_terminal_jobs(jm):
    old = jm.create_job(query="old", tools=[])
    recent = jm.create_job(query="recent", tools=[])
    running = jm.create_job(query="running", tools=[])

    jm.update_state(old, "completed")
    jm.get_status(old).finished_at = datetime.now(timezone.utc) - timedelta(hours=1)
    jm.update_state(recent, "completed")
    jm.update_state(running, "running")

    assert jm.sweep_terminal(ttl_seconds=900) == 1

    with pytest.raises(KeyError):
        jm.get_status(old)
    assert jm.get_status(recent).state == "completed"
    assert jm.get_status(running).state == "running"


def test_sweep_removes_the_queue_too(jm):
    job_id = jm.create_job(query="q", tools=[])
    jm.update_state(job_id, "completed")
    jm.get_status(job_id).finished_at = datetime.now(timezone.utc) - timedelta(hours=1)
    jm.sweep_terminal(ttl_seconds=1)
    with pytest.raises(KeyError):
        jm.get_event_queue(job_id)


def test_sweep_on_an_empty_manager(jm):
    assert jm.sweep_terminal() == 0


@pytest.mark.asyncio
async def test_remove_job_cancels_a_still_running_task(jm):
    job_id = jm.create_job(query="q", tools=[])

    async def runner():
        await asyncio.sleep(60)

    task = asyncio.create_task(runner())
    jm.register_task(job_id, task)
    await asyncio.sleep(0)

    jm.remove_job(job_id)
    with pytest.raises(asyncio.CancelledError):
        await task
    assert task.cancelled()


# ── Stalled (never-terminal) jobs ────────────────────────────────────────────
#
# sweep_terminal only looks at jobs that already finished, so a job whose node
# wedges is skipped by every pass and keeps its task, queue and status forever
# while the UI shows a spinner with no end. sweep_stalled is the backstop.

def test_sweep_stalled_cancels_a_job_that_outran_its_budget(jm):
    job_id = jm.create_job(query="q", tools=[])
    jm.update_state(job_id, "running")
    jm.get_status(job_id).created_at = datetime.now(timezone.utc) - timedelta(seconds=900)

    assert jm.sweep_stalled(600) == 1
    status = jm.get_status(job_id)
    assert status.state == "cancelled"
    assert "time budget" in (status.error or "")
    assert {e["type"] for e in drain(jm, job_id)} == {"cancelled"}


def test_sweep_stalled_leaves_a_young_job_running(jm):
    job_id = jm.create_job(query="q", tools=[])
    jm.update_state(job_id, "running")
    assert jm.sweep_stalled(600) == 0
    assert jm.get_status(job_id).state == "running"


def test_sweep_stalled_ignores_terminal_jobs(jm):
    """A job that completed an hour ago is the TTL sweep's business, not this
    one's — re-cancelling it would throw away a result the client can still GET."""
    job_id = jm.create_job(query="q", tools=[])
    jm.update_state(job_id, "completed", result="done")
    jm.get_status(job_id).created_at = datetime.now(timezone.utc) - timedelta(seconds=3600)
    assert jm.sweep_stalled(600) == 0
    assert jm.get_status(job_id).state == "completed"
    assert jm.get_status(job_id).result == "done"


@pytest.mark.asyncio
async def test_sweep_stalled_cancels_the_underlying_task(jm):
    job_id = jm.create_job(query="q", tools=[])
    started = asyncio.Event()

    async def wedged():
        started.set()
        await asyncio.Event().wait()  # never returns, like a hung poll

    task = asyncio.create_task(wedged())
    jm.register_task(job_id, task)
    await started.wait()
    jm.get_status(job_id).created_at = datetime.now(timezone.utc) - timedelta(seconds=900)

    jm.sweep_stalled(600)
    await asyncio.sleep(0)
    assert task.cancelled() or task.cancelling()


def test_stalled_then_swept_frees_the_job_entirely(jm):
    """The two sweeps have to compose: stalling makes the job terminal, and the
    TTL pass can then collect it. Without the first, nothing ever collects it."""
    job_id = jm.create_job(query="q", tools=[])
    jm.update_state(job_id, "running")
    jm.get_status(job_id).created_at = datetime.now(timezone.utc) - timedelta(seconds=900)

    jm.sweep_stalled(600)
    jm.get_status(job_id).finished_at = datetime.now(timezone.utc) - timedelta(seconds=2000)
    assert jm.sweep_terminal(900) == 1
    with pytest.raises(KeyError):
        jm.get_status(job_id)


# ── Fabricated citations are part of the job's durable status ────────────────

def test_unverified_citations_default_to_empty_and_are_per_job(jm):
    a = jm.create_job(query="q", tools=[])
    b = jm.create_job(query="q", tools=[])
    assert jm.get_status(a).unverified_citations == []
    jm.get_status(a).unverified_citations.append("E9")
    assert jm.get_status(b).unverified_citations == []
