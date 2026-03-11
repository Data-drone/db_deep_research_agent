"""Normalizer node — standardizes and deduplicates evidence."""

from __future__ import annotations

import hashlib
import logging
import re
from typing import Any

from deep_research.state import ResearchState

logger = logging.getLogger(__name__)


def _normalize_text(text: str) -> str:
    """Normalize text for dedup comparison: lowercase, strip, collapse whitespace."""
    text = text.lower().strip()
    text = re.sub(r"\s+", " ", text)
    # Remove common markdown noise
    text = re.sub(r"[*_`#>]", "", text)
    return text


def _compute_content_hash(source_id: str, snippet: str) -> str:
    """Compute a stable content hash from normalized source and snippet."""
    normalized_source = source_id.lower().strip()
    normalized_snippet = _normalize_text(snippet)
    content = f"{normalized_source}\n{normalized_snippet}"
    return hashlib.sha256(content.encode()).hexdigest()[:16]


async def normalizer_node(state: ResearchState, *, model: Any) -> dict:
    """Deduplicate and normalize evidence entries."""
    evidence = state.get("evidence", [])
    if not evidence:
        return {}

    seen_hashes: set[str] = set()
    deduplicated = []

    for ev in evidence:
        # Compute content hash with text normalization
        if not ev.content_hash:
            ev.content_hash = _compute_content_hash(ev.source_id, ev.snippet)

        if ev.content_hash not in seen_hashes:
            seen_hashes.add(ev.content_hash)
            deduplicated.append(ev)
        else:
            logger.debug(f"Dedup: skipping duplicate evidence {ev.evidence_id}")

    return {"evidence": deduplicated}
