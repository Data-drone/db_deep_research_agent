"""Tests for session manager."""

import time
from datetime import datetime, timezone

from deep_research.models import Evidence, ToolCall
from deep_research.session import Session, SessionManager


def _make_evidence(eid: str = "e1") -> Evidence:
    return Evidence(
        evidence_id=eid,
        source_id="src1",
        source_type="genie",
        title="Test",
        uri=None,
        snippet="test snippet",
        confidence=0.9,
        freshness="fresh",
        tool_that_produced_it="genie_sales",
        tool_call_id="tc1",
        iteration=0,
    )


def _make_tool_call(tcid: str = "tc1") -> ToolCall:
    return ToolCall(
        tool_call_id=tcid,
        tool_name="query",
        server_name="genie_sales",
        input_data={"question": "test"},
        output_data={"result": "ok"},
        latency_ms=100,
    )


# --- Session dataclass ---

def test_session_not_expired():
    now = datetime.now(timezone.utc)
    s = Session(session_id="abc", created_at=now, last_active=now, ttl_minutes=30)
    assert not s.is_expired


def test_session_expired():
    from datetime import timedelta
    old = datetime.now(timezone.utc) - timedelta(minutes=31)
    s = Session(session_id="abc", created_at=old, last_active=old, ttl_minutes=30)
    assert s.is_expired


def test_session_to_dict():
    now = datetime.now(timezone.utc)
    s = Session(session_id="abc", created_at=now, last_active=now)
    d = s.to_dict()
    assert d["session_id"] == "abc"
    assert d["turn_count"] == 0
    assert d["evidence_count"] == 0
    assert d["tool_call_count"] == 0
    assert d["is_expired"] is False


# --- SessionManager ---

def test_create_session():
    sm = SessionManager()
    s = sm.create_session()
    assert len(s.session_id) == 32  # uuid hex
    assert s.ttl_minutes == 30


def test_create_session_custom_ttl():
    sm = SessionManager(default_ttl_minutes=60)
    s = sm.create_session()
    assert s.ttl_minutes == 60


def test_create_session_zero_ttl():
    sm = SessionManager()
    s = sm.create_session(ttl_minutes=0)
    assert s.ttl_minutes == 0  # Should NOT fall back to default


def test_get_session():
    sm = SessionManager()
    s = sm.create_session()
    retrieved = sm.get_session(s.session_id)
    assert retrieved is not None
    assert retrieved.session_id == s.session_id


def test_get_session_not_found():
    sm = SessionManager()
    assert sm.get_session("nonexistent") is None


def test_get_session_expired():
    sm = SessionManager()
    s = sm.create_session(ttl_minutes=0)
    # TTL=0 means already expired
    time.sleep(0.01)
    assert sm.get_session(s.session_id) is None


def test_get_or_create_existing():
    sm = SessionManager()
    s = sm.create_session()
    s2 = sm.get_or_create(s.session_id)
    assert s2.session_id == s.session_id


def test_get_or_create_new():
    sm = SessionManager()
    s = sm.get_or_create(None)
    assert s is not None
    assert len(s.session_id) == 32


def test_get_or_create_expired():
    sm = SessionManager()
    s = sm.create_session(ttl_minutes=0)
    time.sleep(0.01)
    s2 = sm.get_or_create(s.session_id)
    assert s2.session_id != s.session_id  # New session created


def test_add_turn():
    sm = SessionManager()
    s = sm.create_session()
    result = sm.add_turn(s.session_id, "user", "hello")
    assert result is True
    retrieved = sm.get_session(s.session_id)
    assert len(retrieved.conversation_history) == 1
    assert retrieved.conversation_history[0] == {"role": "user", "content": "hello"}


def test_add_turn_nonexistent():
    sm = SessionManager()
    result = sm.add_turn("nonexistent", "user", "hello")
    assert result is False


def test_add_evidence():
    sm = SessionManager()
    s = sm.create_session()
    ev = [_make_evidence("e1"), _make_evidence("e2")]
    result = sm.add_evidence(s.session_id, ev)
    assert result is True
    retrieved = sm.get_session(s.session_id)
    assert len(retrieved.accumulated_evidence) == 2


def test_add_tool_calls():
    sm = SessionManager()
    s = sm.create_session()
    tcs = [_make_tool_call("tc1")]
    result = sm.add_tool_calls(s.session_id, tcs)
    assert result is True
    retrieved = sm.get_session(s.session_id)
    assert len(retrieved.accumulated_tool_calls) == 1


def test_delete_session():
    sm = SessionManager()
    s = sm.create_session()
    assert sm.delete_session(s.session_id) is True
    assert sm.get_session(s.session_id) is None


def test_delete_nonexistent():
    sm = SessionManager()
    assert sm.delete_session("nonexistent") is False


def test_cleanup_expired():
    sm = SessionManager()
    sm.create_session(ttl_minutes=0)
    sm.create_session(ttl_minutes=0)
    sm.create_session(ttl_minutes=30)
    time.sleep(0.01)
    removed = sm.cleanup_expired()
    assert removed == 2
    assert sm.active_count == 1


def test_active_count():
    sm = SessionManager()
    sm.create_session()
    sm.create_session()
    assert sm.active_count == 2


def test_multiple_turns_accumulate():
    sm = SessionManager()
    s = sm.create_session()
    sm.add_turn(s.session_id, "user", "q1")
    sm.add_turn(s.session_id, "assistant", "a1")
    sm.add_turn(s.session_id, "user", "q2")
    retrieved = sm.get_session(s.session_id)
    assert len(retrieved.conversation_history) == 3
    assert retrieved.conversation_history[0]["role"] == "user"
    assert retrieved.conversation_history[1]["role"] == "assistant"
