"""Job management for research queries."""

from __future__ import annotations

import asyncio
import logging
import uuid
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Any, Literal

logger = logging.getLogger(__name__)

JobState = Literal["pending", "running", "completed", "cancelled", "failed"]

#: States from which no further transition is allowed.
TERMINAL_STATES: frozenset[str] = frozenset({"completed", "cancelled", "failed"})

#: Max buffered SSE events per job. Token streaming can produce thousands, so
#: this is generous, but bounded: a client that never attaches must not be able
#: to pin the whole report in memory forever.
MAX_QUEUE_SIZE = 5000

#: How long a terminal job is retained before ``sweep_terminal`` may drop it.
DEFAULT_JOB_TTL_SECONDS = 900

#: Grace period added to the budget's time cap before a still-running job is
#: considered stalled. Nothing in the graph enforces the cap itself, so this is
#: the only thing standing between a wedged node and a job that runs forever.
STALL_GRACE_SECONDS = 120


@dataclass
class JobStatus:
    job_id: str
    state: JobState
    query: str
    tools: list[str]
    output_mode: str = "chat"
    created_by: str = ""
    created_at: datetime | None = None
    current_node: str | None = None
    result: str | None = None
    error: str | None = None
    token_usage: dict[str, int] | None = None
    #: Markers the synthesizer invented. Held on the job, not just pushed on the
    #: live ``completed`` frame, so a client that reconnects afterwards is still
    #: told the report contains citations that do not exist.
    unverified_citations: list[str] = field(default_factory=list)
    finished_at: datetime | None = None


class JobManager:
    """In-memory job store. Upgrade to persistent storage later."""

    def __init__(self) -> None:
        self._jobs: dict[str, JobStatus] = {}
        self._queues: dict[str, asyncio.Queue] = {}
        self._tasks: dict[str, asyncio.Task] = {}

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
        self._queues[job_id] = asyncio.Queue(maxsize=MAX_QUEUE_SIZE)
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
        state: JobState,
        result: str | None = None,
        error: str | None = None,
        current_node: str | None = None,
    ) -> None:
        """Update job state.

        Terminal states are final: once a job is completed, cancelled or failed,
        a later transition is ignored. Without this a cancelled job would be
        walked back to ``running`` and then ``completed`` by a runner that had
        not noticed the cancellation.
        """
        status = self.get_status(job_id)

        if status.state in TERMINAL_STATES and state != status.state:
            logger.info(
                "Ignoring %s -> %s transition for job %s (already terminal)",
                status.state,
                state,
                job_id,
            )
            return

        status.state = state
        if state in TERMINAL_STATES and status.finished_at is None:
            status.finished_at = datetime.now(timezone.utc)
        if result is not None:
            status.result = result
        if error is not None:
            status.error = error
        if current_node is not None:
            status.current_node = current_node

    def register_task(self, job_id: str, task: asyncio.Task) -> None:
        """Hold a strong reference to the job's runner task.

        Two reasons: ``cancel_job`` needs something to cancel, and the event
        loop only keeps weak references to tasks, so a fire-and-forget task can
        be garbage-collected mid-run and leave the job stuck in ``running``.
        """
        self._tasks[job_id] = task
        task.add_done_callback(lambda t, jid=job_id: self._on_task_done(jid, t))

    def _on_task_done(self, job_id: str, task: asyncio.Task) -> None:
        self._tasks.pop(job_id, None)
        status = self._jobs.get(job_id)
        if status is None or status.state in TERMINAL_STATES:
            return
        # The runner finished without reaching a terminal state — don't leave
        # the client waiting on a job nothing is working on any more.
        if task.cancelled():
            self.update_state(job_id, "cancelled")
            self.push_event(job_id, {"type": "cancelled"})
            return
        exc = task.exception()
        message = f"Research task ended unexpectedly: {exc}" if exc else "Research task ended without producing a result"
        logger.error("Job %s: %s", job_id, message)
        self.update_state(job_id, "failed", error=message)
        self.push_event(job_id, {"type": "failed", "error": message})

    def cancel_job(self, job_id: str, user_id: str = "") -> None:
        """Cancel a job for real: mark it, then cancel the task running it."""
        self.get_status(job_id, user_id=user_id)  # validates ownership
        self.update_state(job_id, "cancelled")
        task = self._tasks.get(job_id)
        if task is not None and not task.done():
            task.cancel()

    def is_cancelled(self, job_id: str) -> bool:
        status = self._jobs.get(job_id)
        return status is not None and status.state == "cancelled"

    def remove_job(self, job_id: str) -> None:
        """Drop all state for a job."""
        self._jobs.pop(job_id, None)
        self._queues.pop(job_id, None)
        task = self._tasks.pop(job_id, None)
        if task is not None and not task.done():
            task.cancel()

    def sweep_terminal(self, ttl_seconds: int = DEFAULT_JOB_TTL_SECONDS) -> int:
        """Drop terminal jobs that finished more than ``ttl_seconds`` ago."""
        now = datetime.now(timezone.utc)
        stale = [
            job_id
            for job_id, status in self._jobs.items()
            if status.state in TERMINAL_STATES
            and status.finished_at is not None
            and (now - status.finished_at).total_seconds() > ttl_seconds
        ]
        for job_id in stale:
            self.remove_job(job_id)
        if stale:
            logger.info("Swept %d terminal job(s)", len(stale))
        return len(stale)

    def sweep_stalled(self, max_runtime_seconds: int) -> int:
        """Cancel non-terminal jobs that have been alive too long.

        ``sweep_terminal`` only ever looks at jobs that already finished, so a
        job whose node wedges — a Genie poll that never returns, say — is skipped
        by every pass and keeps its task, queue and status forever while the UI
        shows a spinner with no end. This is the backstop: cancelling the job
        makes it terminal, after which the normal TTL sweep can collect it.
        """
        now = datetime.now(timezone.utc)
        stalled = [
            job_id
            for job_id, status in self._jobs.items()
            if status.state not in TERMINAL_STATES
            and status.created_at is not None
            and (now - status.created_at).total_seconds() > max_runtime_seconds
        ]
        for job_id in stalled:
            logger.warning(
                "Job %s exceeded %ds without finishing - cancelling as stalled",
                job_id,
                max_runtime_seconds,
            )
            self.update_state(
                job_id, "cancelled", error="Job exceeded its time budget and was stopped"
            )
            task = self._tasks.get(job_id)
            if task is not None and not task.done():
                task.cancel()
            self.push_event(job_id, {"type": "cancelled"})
        return len(stalled)

    def get_event_queue(self, job_id: str) -> asyncio.Queue:
        if job_id not in self._queues:
            raise KeyError(f"Unknown job: {job_id}")
        return self._queues[job_id]

    def push_event(self, job_id: str, event: dict) -> None:
        queue = self._queues.get(job_id)
        if queue is None:
            logger.warning(f"push_event: no queue for job {job_id}, event dropped")
            return
        try:
            queue.put_nowait(event)
        except asyncio.QueueFull:
            # Nobody is draining fast enough (usually: no client attached).
            # Drop the oldest event rather than the newest, so terminal events
            # are never the ones lost.
            try:
                queue.get_nowait()
            except asyncio.QueueEmpty:
                pass
            try:
                queue.put_nowait(event)
            except asyncio.QueueFull:
                logger.warning(f"push_event: queue full for job {job_id}, event dropped")
