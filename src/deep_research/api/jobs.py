"""Job management for research queries."""

from __future__ import annotations

import uuid
from dataclasses import dataclass, field
from typing import Any, Literal


@dataclass
class JobStatus:
    job_id: str
    state: Literal["pending", "running", "completed", "cancelled", "failed"]
    query: str
    tools: list[str]
    output_mode: str = "chat"
    result: str | None = None
    error: str | None = None


class JobManager:
    """In-memory job store. Upgrade to persistent storage later."""

    def __init__(self) -> None:
        self._jobs: dict[str, JobStatus] = {}

    def create_job(
        self,
        query: str,
        tools: list[str],
        output_mode: str = "chat",
    ) -> str:
        job_id = f"job-{uuid.uuid4().hex[:12]}"
        self._jobs[job_id] = JobStatus(
            job_id=job_id,
            state="pending",
            query=query,
            tools=tools,
            output_mode=output_mode,
        )
        return job_id

    def get_status(self, job_id: str) -> JobStatus:
        if job_id not in self._jobs:
            raise KeyError(f"Unknown job: {job_id}")
        return self._jobs[job_id]

    def update_state(
        self,
        job_id: str,
        state: Literal["pending", "running", "completed", "cancelled", "failed"],
        result: str | None = None,
        error: str | None = None,
    ) -> None:
        status = self.get_status(job_id)
        status.state = state
        if result is not None:
            status.result = result
        if error is not None:
            status.error = error

    def cancel_job(self, job_id: str) -> None:
        self.update_state(job_id, "cancelled")
