"""Job management for research queries."""

from __future__ import annotations

import uuid
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Any, Literal


@dataclass
class JobStatus:
    job_id: str
    state: Literal["pending", "running", "completed", "cancelled", "failed"]
    query: str
    tools: list[str]
    output_mode: str = "chat"
    created_by: str = ""
    created_at: datetime | None = None
    current_node: str | None = None
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
        user_id: str = "",
    ) -> str:
        job_id = f"job-{uuid.uuid4().hex[:12]}"
        self._jobs[job_id] = JobStatus(
            job_id=job_id,
            state="pending",
            query=query,
            tools=tools,
            output_mode=output_mode,
            created_by=user_id,
            created_at=datetime.now(timezone.utc),
        )
        return job_id

    def get_status(self, job_id: str, user_id: str = "") -> JobStatus:
        if job_id not in self._jobs:
            raise KeyError(f"Unknown job: {job_id}")
        job = self._jobs[job_id]
        # Enforce ownership if user_id is provided and job has an owner
        if user_id and job.created_by and job.created_by != user_id:
            raise PermissionError(f"Job {job_id} belongs to another user")
        return job

    def update_state(
        self,
        job_id: str,
        state: Literal["pending", "running", "completed", "cancelled", "failed"],
        result: str | None = None,
        error: str | None = None,
        current_node: str | None = None,
    ) -> None:
        status = self.get_status(job_id)
        status.state = state
        if result is not None:
            status.result = result
        if error is not None:
            status.error = error
        if current_node is not None:
            status.current_node = current_node

    def cancel_job(self, job_id: str, user_id: str = "") -> None:
        self.get_status(job_id, user_id=user_id)  # validates ownership
        self.update_state(job_id, "cancelled")
