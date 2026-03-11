"""In-memory session manager with TTL expiry.

All mutation goes through SessionManager methods to ensure thread safety.
"""

from __future__ import annotations

import logging
import threading
import uuid
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Any

from deep_research.models import Evidence, ToolCall

logger = logging.getLogger(__name__)


@dataclass
class Session:
    """Immutable-ish session data. Mutate only via SessionManager methods."""

    session_id: str
    created_at: datetime
    last_active: datetime
    conversation_history: list[dict[str, str]] = field(default_factory=list)
    accumulated_evidence: list[Evidence] = field(default_factory=list)
    accumulated_tool_calls: list[ToolCall] = field(default_factory=list)
    ttl_minutes: int = 30

    @property
    def is_expired(self) -> bool:
        now = datetime.now(timezone.utc)
        elapsed = (now - self.last_active).total_seconds()
        return elapsed > self.ttl_minutes * 60

    def to_dict(self) -> dict[str, Any]:
        """Serialize session metadata (not full evidence)."""
        return {
            "session_id": self.session_id,
            "created_at": self.created_at.isoformat(),
            "last_active": self.last_active.isoformat(),
            "turn_count": len(self.conversation_history),
            "evidence_count": len(self.accumulated_evidence),
            "tool_call_count": len(self.accumulated_tool_calls),
            "is_expired": self.is_expired,
        }


class SessionManager:
    """Thread-safe in-memory session store with TTL expiry.

    All session mutations go through this manager to guarantee
    thread safety via a single lock.
    """

    def __init__(self, default_ttl_minutes: int = 30) -> None:
        self._sessions: dict[str, Session] = {}
        self._lock = threading.Lock()
        self._default_ttl = default_ttl_minutes

    def create_session(self, ttl_minutes: int | None = None) -> Session:
        """Create a new session and return it."""
        ttl = self._default_ttl if ttl_minutes is None else ttl_minutes
        now = datetime.now(timezone.utc)
        session = Session(
            session_id=uuid.uuid4().hex,
            created_at=now,
            last_active=now,
            ttl_minutes=ttl,
        )
        with self._lock:
            self._sessions[session.session_id] = session
        logger.info("Created session %s (TTL=%dm)", session.session_id, session.ttl_minutes)
        return session

    def get_session(self, session_id: str) -> Session | None:
        """Get a session by ID. Returns None if not found or expired."""
        with self._lock:
            session = self._sessions.get(session_id)
            if session is None:
                return None
            if session.is_expired:
                del self._sessions[session_id]
                logger.info("Session %s expired, removed", session_id)
                return None
            return session

    def get_or_create(self, session_id: str | None) -> Session:
        """Get existing session or create a new one."""
        if session_id:
            session = self.get_session(session_id)
            if session:
                return session
        return self.create_session()

    def add_turn(self, session_id: str, role: str, content: str) -> bool:
        """Append a conversation turn. Returns False if session not found/expired."""
        with self._lock:
            session = self._sessions.get(session_id)
            if session is None or session.is_expired:
                return False
            session.conversation_history.append({"role": role, "content": content})
            session.last_active = datetime.now(timezone.utc)
            return True

    def add_evidence(self, session_id: str, evidence: list[Evidence]) -> bool:
        """Accumulate evidence. Returns False if session not found/expired."""
        with self._lock:
            session = self._sessions.get(session_id)
            if session is None or session.is_expired:
                return False
            session.accumulated_evidence.extend(evidence)
            session.last_active = datetime.now(timezone.utc)
            return True

    def add_tool_calls(self, session_id: str, tool_calls: list[ToolCall]) -> bool:
        """Accumulate tool calls. Returns False if session not found/expired."""
        with self._lock:
            session = self._sessions.get(session_id)
            if session is None or session.is_expired:
                return False
            session.accumulated_tool_calls.extend(tool_calls)
            session.last_active = datetime.now(timezone.utc)
            return True

    def delete_session(self, session_id: str) -> bool:
        """Delete a session. Returns True if it existed."""
        with self._lock:
            return self._sessions.pop(session_id, None) is not None

    def cleanup_expired(self) -> int:
        """Remove all expired sessions. Returns count removed."""
        with self._lock:
            expired = [
                sid for sid, s in self._sessions.items() if s.is_expired
            ]
            for sid in expired:
                del self._sessions[sid]
        if expired:
            logger.info("Cleaned up %d expired sessions", len(expired))
        return len(expired)

    @property
    def active_count(self) -> int:
        """Number of non-expired sessions."""
        with self._lock:
            return sum(1 for s in self._sessions.values() if not s.is_expired)
