"""Tests for ClarificationStore."""

import asyncio
import pytest

from deep_research.api.clarification import InMemoryClarificationStore


def test_create_and_get():
    store = InMemoryClarificationStore()
    state = store.create("job-1", "Which company?", ["A", "B"], "A")
    assert state.job_id == "job-1"
    assert state.status == "pending"
    assert state.clarification_id.startswith("clr-")

    fetched = store.get("job-1")
    assert fetched is state


def test_submit_answer_first_wins():
    store = InMemoryClarificationStore()
    state = store.create("job-1", "Which?", ["A", "B"], "A")

    result = store.submit_answer("job-1", state.clarification_id, "B")
    assert result is not None
    assert result.status == "answered"
    assert result.answer == "B"

    # Second answer rejected
    result2 = store.submit_answer("job-1", state.clarification_id, "A")
    assert result2 is None


def test_submit_answer_wrong_clarification_id():
    store = InMemoryClarificationStore()
    store.create("job-1", "Which?", ["A", "B"], "A")

    result = store.submit_answer("job-1", "wrong-id", "B")
    assert result is None


def test_submit_answer_unknown_job():
    store = InMemoryClarificationStore()
    result = store.submit_answer("nonexistent", "clr-123", "B")
    assert result is None


def test_mark_timed_out():
    store = InMemoryClarificationStore()
    state = store.create("job-1", "Which?", ["A", "B"], "A")
    store.mark_timed_out("job-1")
    assert state.status == "timed_out"

    # Late answer rejected
    result = store.submit_answer("job-1", state.clarification_id, "B")
    assert result is None


def test_mark_cancelled():
    store = InMemoryClarificationStore()
    state = store.create("job-1", "Which?", ["A", "B"], "A")
    store.mark_cancelled("job-1")
    assert state.status == "cancelled"


def test_event_is_set_on_answer():
    store = InMemoryClarificationStore()
    state = store.create("job-1", "Which?", ["A", "B"], "A")
    event = store._events.get("job-1")
    assert event is not None
    assert not event.is_set()

    store.submit_answer("job-1", state.clarification_id, "B")
    assert event.is_set()


def test_event_is_set_on_cancel():
    store = InMemoryClarificationStore()
    store.create("job-1", "Which?", ["A", "B"], "A")
    event = store._events.get("job-1")
    assert not event.is_set()

    store.mark_cancelled("job-1")
    assert event.is_set()


def test_event_is_set_on_timeout():
    store = InMemoryClarificationStore()
    store.create("job-1", "Which?", ["A", "B"], "A")
    event = store._events.get("job-1")
    assert not event.is_set()

    store.mark_timed_out("job-1")
    assert event.is_set()


def test_get_nonexistent():
    store = InMemoryClarificationStore()
    assert store.get("nonexistent") is None


def test_cleanup():
    store = InMemoryClarificationStore()
    store.create("job-1", "Which?", ["A", "B"], "A")
    store.cleanup("job-1")
    assert store.get("job-1") is None


def test_duplicate_pending_raises():
    store = InMemoryClarificationStore()
    store.create("job-1", "Which?", ["A", "B"], "A")
    with pytest.raises(ValueError, match="Pending clarification already exists"):
        store.create("job-1", "Another?", ["C", "D"], "C")


@pytest.mark.asyncio
async def test_wait_for_resolution_answered():
    store = InMemoryClarificationStore()
    state = store.create("job-1", "Which?", ["A", "B"], "A")

    # Submit answer in background
    async def submit_later():
        await asyncio.sleep(0.05)
        store.submit_answer("job-1", state.clarification_id, "B")

    asyncio.create_task(submit_later())
    result = await store.wait_for_resolution("job-1", timeout=5.0)
    assert result is not None
    assert result.status == "answered"
    assert result.answer == "B"


@pytest.mark.asyncio
async def test_wait_for_resolution_timeout():
    store = InMemoryClarificationStore()
    store.create("job-1", "Which?", ["A", "B"], "A")

    result = await store.wait_for_resolution("job-1", timeout=0.1)
    assert result is not None
    assert result.status == "timed_out"


@pytest.mark.asyncio
async def test_wait_for_resolution_already_answered():
    store = InMemoryClarificationStore()
    state = store.create("job-1", "Which?", ["A", "B"], "A")
    store.submit_answer("job-1", state.clarification_id, "B")

    # Already answered — should return immediately
    result = await store.wait_for_resolution("job-1", timeout=5.0)
    assert result is not None
    assert result.status == "answered"


@pytest.mark.asyncio
async def test_wait_for_resolution_nonexistent():
    store = InMemoryClarificationStore()
    result = await store.wait_for_resolution("nonexistent", timeout=1.0)
    assert result is None
