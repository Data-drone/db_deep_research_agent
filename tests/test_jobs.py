"""Tests for job management."""

import pytest

from deep_research.api.jobs import JobManager


@pytest.fixture
def job_manager():
    return JobManager()


def test_create_job(job_manager):
    job_id = job_manager.create_job(query="What is revenue?", tools=["genie"])
    assert job_id is not None
    status = job_manager.get_status(job_id)
    assert status.state == "pending"


def test_cancel_job(job_manager):
    job_id = job_manager.create_job(query="test", tools=[])
    job_manager.cancel_job(job_id)
    status = job_manager.get_status(job_id)
    assert status.state == "cancelled"


def test_update_state(job_manager):
    job_id = job_manager.create_job(query="test", tools=[])
    job_manager.update_state(job_id, "running")
    assert job_manager.get_status(job_id).state == "running"
    job_manager.update_state(job_id, "completed", result="Answer here")
    status = job_manager.get_status(job_id)
    assert status.state == "completed"
    assert status.result == "Answer here"


def test_unknown_job_raises(job_manager):
    with pytest.raises(KeyError):
        job_manager.get_status("nonexistent")
