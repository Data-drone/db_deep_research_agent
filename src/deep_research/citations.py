"""Enumerated evidence markers, and validation of the model's citations.

The synthesizer used to hand the model free-text ``[Source: Tool — description]``
markers and never parse what came back, so a citation naming a table that was
never queried read exactly like a real one. Evidence is now enumerated as
``[E1]``…``[En]``; only markers that resolve to a real evidence item survive,
and anything else is rewritten in place so the reader can see it was made up.
"""

from __future__ import annotations

import logging
import re
from typing import Iterable

from deep_research.models import Citation, Evidence

logger = logging.getLogger(__name__)

#: Matches a lone evidence marker, e.g. ``[E3]``. Kept because it is the shape
#: the prompt asks for and the shape the UI links, but it is *not* what
#: validation keys off: a model that writes ``[E1, E9]`` or ``[E1-E3]`` would
#: slip past a marker-only pattern entirely and ship the fabricated ``E9``.
MARKER_RE = re.compile(r"\[\s*E(\d+)\s*\]", re.IGNORECASE)

#: Any bracketed run with no nested brackets. Every citation shape the model can
#: produce is one of these, so validation starts here and then decides whether a
#: given group is a citation attempt at all.
_GROUP_RE = re.compile(r"\[([^\[\]]*)\]")

#: A number inside a group, with or without the ``E`` prefix and with leading
#: zeros tolerated, so ``E01`` resolves to ``E1`` instead of looking fabricated.
_TOKEN_RE = re.compile(r"(E\s*)?0*(\d+)", re.IGNORECASE)

#: Everything allowed *between* tokens in a citation group. If anything else is
#: left over, the group is prose or a markdown link, not a citation.
_SEPARATOR_RE = re.compile(r"(?:[\s,;&/+·-]|--|\u2013|\u2014|\band\b|\bto\b)+", re.IGNORECASE)

#: The old free-text format the model was previously told to use. Still worth
#: recognising: a model that falls back to it is naming an unverifiable source.
_LEGACY_SOURCE_RE = re.compile(r"^\s*sources?\s*[:\u2014-]", re.IGNORECASE)

#: Guards against a range like ``[E1-E99999]`` expanding into a huge list.
_MAX_RANGE = 100

#: What a marker with no matching evidence item is replaced with.
UNVERIFIED_LABEL = "[unverified]"

#: Sentence-ish split. Deliberately crude: it only has to group a claim with
#: the markers that follow it, not parse prose correctly.
_CLAIM_SPLIT_RE = re.compile(r"(?<=[.!?])\s+|\n+")

#: Tidies the gap a removed marker leaves behind ("rose 12% ." -> "rose 12%.").
_ORPHAN_SPACE_RE = re.compile(r"\s+([.!?,;:])")


def build_evidence_index(evidence: Iterable[Evidence]) -> dict[str, Evidence]:
    """Map ``E1``…``En`` to evidence items, numbered as the model will see them."""
    return {f"E{i}": e for i, e in enumerate(evidence, start=1)}


def render_evidence_block(index: dict[str, Evidence]) -> list[str]:
    """Render the evidence list for the prompt, one line per item.

    The tool and title are shown for the model's benefit but are *not* what it
    cites — it cites the marker, which is the only thing we can verify.
    """
    lines = []
    for marker, e in index.items():
        tool = (e.tool_that_produced_it or "unknown tool").replace("]", ")")
        title = (e.title or e.source_id or "untitled").replace("]", ")")
        lines.append(f"[{marker}] ({tool} — {title}) {e.snippet}")
    return lines


def _dash_between(inner: str, left: re.Match[str], right: re.Match[str]) -> bool:
    """True if the gap between two tokens is a range dash and nothing else."""
    gap = inner[left.end():right.start()].strip()
    return gap in {"-", "\u2013", "\u2014", "--", "to"}


def _group_markers(inner: str) -> list[str] | None:
    """The markers a bracketed group cites, or ``None`` if it isn't a citation.

    Returning ``None`` matters as much as returning markers: prose in brackets
    and markdown link text must be left exactly as the model wrote it, or a
    report gets quietly mangled. A group only counts as a citation attempt if it
    is made of numbers and separators *and* at least one of them carries the
    ``E`` prefix, or if it uses the legacy ``[Source: ...]`` shape.
    """
    tokens = list(_TOKEN_RE.finditer(inner))

    if _LEGACY_SOURCE_RE.match(inner):
        # A free-text source name can never be checked against the evidence set,
        # so treat the whole group as one unresolvable marker.
        return [m for m in _numbers(inner, tokens)] or ["<free-text source>"]

    if not tokens or not any(m.group(1) for m in tokens):
        return None
    if _SEPARATOR_RE.sub("", _TOKEN_RE.sub("", inner)) != "":
        return None
    return _numbers(inner, tokens)


def _numbers(inner: str, tokens: list[re.Match[str]]) -> list[str]:
    """Expand a group's tokens into markers, resolving ``E1-E3`` style ranges."""
    markers: list[str] = []
    i = 0
    while i < len(tokens):
        start = int(tokens[i].group(2))
        if i + 1 < len(tokens) and _dash_between(inner, tokens[i], tokens[i + 1]):
            end = int(tokens[i + 1].group(2))
            if start <= end and end - start < _MAX_RANGE:
                markers.extend(f"E{n}" for n in range(start, end + 1))
                i += 2
                continue
        markers.append(f"E{start}")
        i += 1
    return markers


def _rewrite_group(markers: list[str], known: set[str]) -> str:
    """Render a validated group: keep what resolves, flag what doesn't."""
    valid = [m for m in markers if m in known]
    invalid = [m for m in markers if m not in known]
    parts = [f"[{m}]" for m in valid]
    if invalid:
        parts.append(UNVERIFIED_LABEL)
    return " ".join(parts)


def extract_citations(
    text: str, index: dict[str, Evidence]
) -> tuple[str, list[Citation], list[str]]:
    """Validate the citations in ``text`` against ``index``.

    Returns ``(cleaned_text, citations, unknown_markers)``:

    - ``cleaned_text`` has every citation group normalised to individual
      ``[En]`` markers, with :data:`UNVERIFIED_LABEL` standing in for anything
      that does not resolve to a real evidence item. Non-citation brackets are
      untouched.
    - ``citations`` is one :class:`Citation` per claim that cited at least one
      real evidence item, with ``confidence`` taken as the weakest supporting
      item — a claim is only as good as its shakiest source.
    - ``unknown_markers`` are the markers the model invented.
    """
    normalized = {k.upper(): v for k, v in index.items()}
    known = set(normalized)
    unknown: list[str] = []

    citations: list[Citation] = []
    for n, raw_claim in enumerate(_CLAIM_SPLIT_RE.split(text), start=1):
        markers: list[str] = []
        for group in _GROUP_RE.finditer(raw_claim):
            found = _group_markers(group.group(1))
            if found:
                markers.extend(found)
        if not markers:
            continue
        resolved = [normalized[m] for m in markers if m in known]
        unknown.extend(m for m in markers if m not in known)
        if not resolved:
            continue
        claim = _strip_citations(raw_claim)
        if not claim:
            continue
        citations.append(
            Citation(
                claim_id=f"c-{n}",
                claim=claim,
                evidence_ids=[e.evidence_id for e in resolved],
                confidence=min(e.confidence for e in resolved),
            )
        )

    def _clean(match: re.Match[str]) -> str:
        found = _group_markers(match.group(1))
        if found is None:
            return match.group(0)
        return _rewrite_group(found, known)

    cleaned = _GROUP_RE.sub(_clean, text)

    if unknown:
        logger.warning(
            "Synthesizer cited %d marker(s) with no matching evidence: %s",
            len(unknown), sorted(set(unknown)),
        )

    return cleaned, citations, unknown


def _strip_citations(claim: str) -> str:
    """The claim text with its citation groups removed, for the ``Citation``."""
    stripped = _GROUP_RE.sub(
        lambda m: "" if _group_markers(m.group(1)) is not None else m.group(0), claim
    )
    stripped = _ORPHAN_SPACE_RE.sub(r"\1", stripped)
    return re.sub(r"\s{2,}", " ", stripped).strip()
