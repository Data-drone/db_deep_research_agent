"""Feedback logging to MLflow."""

from __future__ import annotations

import logging
from typing import Any, Literal

logger = logging.getLogger(__name__)

try:
    import mlflow

    MLFLOW_AVAILABLE = True
except ImportError:
    MLFLOW_AVAILABLE = False
    mlflow = None  # type: ignore[assignment]


def log_user_feedback(
    trace_id: str,
    rating: Literal["thumbs_up", "thumbs_down"],
    comment: str = "",
    user_id: str = "",
) -> bool:
    """Log user feedback (thumbs up/down) to MLflow.

    Returns True if feedback was logged successfully.
    """
    if not MLFLOW_AVAILABLE:
        logger.warning("mlflow not available — feedback not logged")
        return False

    try:
        # Map rating to numeric score for MLflow
        score = 1.0 if rating == "thumbs_up" else 0.0

        mlflow.log_metrics(
            {
                f"feedback_score_{trace_id[:8]}": score,
            }
        )
        mlflow.log_params(
            {
                f"feedback_rating_{trace_id[:8]}": rating,
                f"feedback_user_{trace_id[:8]}": user_id or "anonymous",
            }
        )

        if comment:
            mlflow.log_text(comment, f"feedback/{trace_id}.txt")

        logger.info(
            f"Logged feedback for trace {trace_id}: {rating}"
            + (f" (comment: {comment[:50]})" if comment else "")
        )
        return True

    except Exception:
        logger.exception(f"Failed to log feedback for trace {trace_id}")
        return False
