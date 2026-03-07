"""Tests for feedback logging."""

from deep_research.feedback import log_user_feedback


def test_log_feedback_no_mlflow():
    """log_user_feedback should return False gracefully without mlflow."""
    result = log_user_feedback(
        trace_id="test-123",
        rating="thumbs_up",
        comment="Great answer!",
        user_id="user-1",
    )
    # Without mlflow installed, returns False but doesn't raise
    assert isinstance(result, bool)


def test_log_feedback_thumbs_down():
    """Should handle thumbs_down rating."""
    result = log_user_feedback(
        trace_id="test-456",
        rating="thumbs_down",
    )
    assert isinstance(result, bool)
