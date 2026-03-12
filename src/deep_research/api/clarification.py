"""Clarification state management.

Uses in-memory storage (single-process). To support multi-worker deployments,
swap InMemoryClarificationStore for a Redis-backed implementation.
"""

from __future__ import annotations

import asyncio
import logging
import uuid
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Literal

logger = logging.getLogger(__name__)

ClarificationStatus = Literal["pending", "answered", "timed_out", "cancelled"]


@dataclass
class ClarificationState:
    clarification_id: str
    job_id: str
    question: str
    options: list[str]
    best_guess: str
    status: ClarificationStatus = "pending"
    answer: str | None = None
    created_at: datetime = field(default_factory=lambda: datetime.now(timezone.utc))


class ClarificationStore(ABC):
    """Abstract base for clarification state storage."""

    @abstractmethod
    def create(self, job_id: str, question: str, options: list[str], best_guess: str) -> ClarificationState:
        ...

    @abstractmethod
    def get(self, job_id: str) -> ClarificationState | None:
        ...

    @abstractmethod
    def submit_answer(self, job_id: str, clarification_id: str, answer: str) -> ClarificationState | None:
        """Submit answer. Returns updated state if accepted, None if expired/invalid.
        First-answer-wins: subsequent answers are rejected."""
        ...

    @abstractmethod
    def mark_timed_out(self, job_id: str) -> None:
        ...

    @abstractmethod
    def mark_cancelled(self, job_id: str) -> None:
        ...

    @abstractmethod
    async def wait_for_resolution(self, job_id: str, timeout: float) -> ClarificationState | None:
        """Wait for the clarification to be resolved (answered, timed out, or cancelled).
        Returns the final state, or None if not found."""
        ...

    @abstractmethod
    def cleanup(self, job_id: str) -> None:
        """Remove all state for a job. Call after job completion."""
        ...


class InMemoryClarificationStore(ClarificationStore):
    """In-memory implementation. Works for single-process deployments.

    Limitation: Not safe across multiple workers/replicas.
    For multi-worker, implement a Redis-backed ClarificationStore.
    """

    def __init__(self) -> None:
        self._states: dict[str, ClarificationState] = {}
        self._events: dict[str, asyncio.Event] = {}

    def create(self, job_id: str, question: str, options: list[str], best_guess: str) -> ClarificationState:
        if job_id in self._states and self._states[job_id].status == "pending":
            raise ValueError(f"Pending clarification already exists for job {job_id}")
        cid = f"clr-{uuid.uuid4().hex[:12]}"
        state = ClarificationState(
            clarification_id=cid,
            job_id=job_id,
            question=question,
            options=options,
            best_guess=best_guess,
        )
        self._states[job_id] = state
        self._events[job_id] = asyncio.Event()
        return state

    def get(self, job_id: str) -> ClarificationState | None:
        return self._states.get(job_id)

    def submit_answer(self, job_id: str, clarification_id: str, answer: str) -> ClarificationState | None:
        state = self._states.get(job_id)
        if state is None:
            return None
        if state.clarification_id != clarification_id:
            return None
        if state.status != "pending":
            return None  # First-answer-wins / already resolved
        state.status = "answered"
        state.answer = answer
        event = self._events.get(job_id)
        if event:
            event.set()
        return state

    def mark_timed_out(self, job_id: str) -> None:
        state = self._states.get(job_id)
        if state and state.status == "pending":
            state.status = "timed_out"
        # Set event BEFORE cleanup so waiting coroutine wakes up
        event = self._events.get(job_id)
        if event:
            event.set()
        self._events.pop(job_id, None)

    def mark_cancelled(self, job_id: str) -> None:
        state = self._states.get(job_id)
        if state and state.status == "pending":
            state.status = "cancelled"
        # Set event BEFORE cleanup so waiting coroutine wakes up
        event = self._events.get(job_id)
        if event:
            event.set()
        self._events.pop(job_id, None)

    async def wait_for_resolution(self, job_id: str, timeout: float) -> ClarificationState | None:
        """Wait up to timeout seconds for clarification resolution."""
        state = self._states.get(job_id)
        if state is None:
            return None
        # Check if already resolved (answer arrived before we started waiting)
        if state.status != "pending":
            return state
        event = self._events.get(job_id)
        if event is None:
            return state
        try:
            await asyncio.wait_for(event.wait(), timeout=timeout)
        except asyncio.TimeoutError:
            self.mark_timed_out(job_id)
        return self._states.get(job_id)

    def cleanup(self, job_id: str) -> None:
        """Remove all state for a job."""
        self._states.pop(job_id, None)
        self._events.pop(job_id, None)
