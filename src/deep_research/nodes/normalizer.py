"""Normalizer node — standardizes and deduplicates evidence."""

from __future__ import annotations

import hashlib
import logging
from typing import Any

from deep_research.state import ResearchState

logger = logging.getLogger(__name__)


async def normalizer_node(state: ResearchState, *, model: Any) -> dict:
    """Deduplicate and normalize evidence entries."""
    evidence = state.get("evidence", [])
    if not evidence:
        return {}

    seen_hashes: set[str] = set()
    deduplicated = []

    for ev in evidence:
        # Compute content hash if not set
        if not ev.content_hash:
            content = f"{ev.source_id}:{ev.snippet}"
            ev.content_hash = hashlib.sha256(content.encode()).hexdigest()[:16]

        if ev.content_hash not in seen_hashes:
            seen_hashes.add(ev.content_hash)
            deduplicated.append(ev)
        else:
            logger.debug(f"Dedup: skipping duplicate evidence {ev.evidence_id}")

    return {"evidence": deduplicated}
