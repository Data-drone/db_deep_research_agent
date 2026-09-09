"""Tests for evidence-marker citations and validation."""

import pytest

from deep_research.citations import (
    UNVERIFIED_LABEL,
    build_evidence_index,
    extract_citations,
    render_evidence_block,
)
from deep_research.models import Evidence


def make_evidence(n: int, confidence: float = 0.9, **kw) -> Evidence:
    defaults = dict(
        evidence_id=f"ev-{n}",
        source_id=f"src-{n}",
        source_type="document",
        title=f"Title {n}",
        uri=None,
        snippet=f"snippet {n}",
        confidence=confidence,
        freshness="fresh",
        tool_that_produced_it="genie_test",
        tool_call_id=f"tc-{n}",
        iteration=0,
    )
    defaults.update(kw)
    return Evidence(**defaults)


@pytest.fixture
def index():
    return build_evidence_index([make_evidence(1), make_evidence(2, confidence=0.4)])


def test_index_is_one_based_and_ordered():
    idx = build_evidence_index([make_evidence(1), make_evidence(2)])
    assert list(idx) == ["E1", "E2"]
    assert idx["E1"].evidence_id == "ev-1"


def test_empty_evidence_yields_empty_index():
    assert build_evidence_index([]) == {}


def test_render_block_one_line_per_item(index):
    lines = render_evidence_block(index)
    assert lines == [
        "[E1] (genie_test — Title 1) snippet 1",
        "[E2] (genie_test — Title 2) snippet 2",
    ]


def test_render_block_neutralizes_brackets_in_labels():
    """A title containing ] must not be able to forge a marker boundary."""
    idx = build_evidence_index([make_evidence(1, title="Bad] [E9")])
    line = render_evidence_block(idx)[0]
    assert "Bad)" in line
    assert "[E9" in line  # kept verbatim, but the closing bracket was defanged
    assert line.startswith("[E1] ")


def test_valid_markers_are_kept_and_become_citations(index):
    text = "Revenue rose 12% [E1]. Margins fell [E2]."
    cleaned, citations, unknown = extract_citations(text, index)
    assert cleaned == text
    assert unknown == []
    assert [c.claim for c in citations] == ["Revenue rose 12%.", "Margins fell."]
    assert [c.evidence_ids for c in citations] == [["ev-1"], ["ev-2"]]


def test_multiple_markers_on_one_claim(index):
    _, citations, _ = extract_citations("Both agree [E1] [E2].", index)
    assert len(citations) == 1
    assert citations[0].evidence_ids == ["ev-1", "ev-2"]


def test_claim_confidence_is_the_weakest_supporting_item(index):
    """A claim is only as trustworthy as its shakiest source."""
    _, citations, _ = extract_citations("Both agree [E1] [E2].", index)
    assert citations[0].confidence == 0.4


def test_fabricated_marker_is_replaced_and_reported(index):
    cleaned, citations, unknown = extract_citations(
        "Revenue tripled [E9].", index
    )
    assert UNVERIFIED_LABEL in cleaned
    assert "[E9]" not in cleaned
    assert unknown == ["E9"]
    # No real evidence backed the claim, so it produces no citation.
    assert citations == []


def test_claim_with_mixed_real_and_fake_markers_keeps_only_the_real(index):
    cleaned, citations, unknown = extract_citations("Mixed [E1] [E9].", index)
    assert "[E1]" in cleaned and UNVERIFIED_LABEL in cleaned
    assert unknown == ["E9"]
    assert len(citations) == 1
    assert citations[0].evidence_ids == ["ev-1"]


def test_uncited_prose_produces_no_citations(index):
    _, citations, unknown = extract_citations("Just an opinion, no sources.", index)
    assert citations == [] and unknown == []


def test_markers_are_case_and_space_tolerant(index):
    cleaned, citations, unknown = extract_citations("Claim [ e1 ].", index)
    assert unknown == []
    assert citations[0].evidence_ids == ["ev-1"]
    # Normalised to the canonical marker so the UI has one shape to link.
    assert cleaned == "Claim [E1]."


def test_marker_removal_does_not_leave_orphaned_whitespace(index):
    _, citations, _ = extract_citations("Revenue rose 12% [E1] , then fell.", index)
    assert citations[0].claim == "Revenue rose 12%, then fell."


def test_newline_separated_claims_are_split(index):
    text = "- First point [E1]\n- Second point [E2]"
    _, citations, _ = extract_citations(text, index)
    assert len(citations) == 2


def test_every_marker_is_unverified_when_there_is_no_evidence():
    cleaned, citations, unknown = extract_citations("Claim [E1].", {})
    assert cleaned == f"Claim {UNVERIFIED_LABEL}."
    assert citations == []
    assert unknown == ["E1"]


# --- Citation shapes the model actually produces -------------------------------
#
# Validation used to key off a marker-only pattern, which meant a model writing
# its markers any other way bypassed validation entirely and shipped whatever it
# had invented. These cover the shapes that got through.


def test_comma_grouped_markers_are_each_validated(index):
    cleaned, citations, unknown = extract_citations("Revenue grew 12% [E1, E9].", index)
    assert unknown == ["E9"]
    assert cleaned == f"Revenue grew 12% [E1] {UNVERIFIED_LABEL}."
    assert citations[0].evidence_ids == ["ev-1"]


def test_semicolon_grouped_markers_both_resolve(index):
    cleaned, citations, _ = extract_citations("Both [E1; E2] agree.", index)
    assert cleaned == "Both [E1] [E2] agree."
    assert citations[0].evidence_ids == ["ev-1", "ev-2"]


def test_marker_range_is_expanded(index):
    idx = build_evidence_index([make_evidence(1), make_evidence(2), make_evidence(3)])
    cleaned, citations, unknown = extract_citations("Range [E1-E3] holds.", idx)
    assert unknown == []
    assert cleaned == "Range [E1] [E2] [E3] holds."
    assert citations[0].evidence_ids == ["ev-1", "ev-2", "ev-3"]


def test_absurd_range_does_not_expand(index):
    """A range wide enough to be nonsense is treated as two markers, not
    expanded into thousands of them."""
    cleaned, _, unknown = extract_citations("Wide [E1-E9999] range.", index)
    assert unknown == ["E9999"]
    assert cleaned == f"Wide [E1] {UNVERIFIED_LABEL} range."


def test_leading_zeros_resolve_instead_of_looking_fabricated(index):
    cleaned, citations, unknown = extract_citations("Padded [E01].", index)
    assert unknown == []
    assert cleaned == "Padded [E1]."
    assert citations[0].evidence_ids == ["ev-1"]


def test_legacy_free_text_source_is_flagged(index):
    """The format the model used to be told to use names a source that cannot be
    checked against anything, so it is never allowed to read as a citation."""
    cleaned, citations, unknown = extract_citations(
        "Profit doubled [Source: Genie — quarterly_revenue].", index
    )
    assert unknown == ["<free-text source>"]
    assert cleaned == f"Profit doubled {UNVERIFIED_LABEL}."
    assert citations == []


def test_markdown_links_are_left_alone(index):
    text = "See [the filing](https://example.com/8-K) for detail [E1]."
    cleaned, citations, unknown = extract_citations(text, index)
    assert cleaned == "See [the filing](https://example.com/8-K) for detail [E1]."
    assert unknown == []
    assert citations[0].claim == "See [the filing](https://example.com/8-K) for detail."


def test_bracketed_prose_is_left_alone(index):
    text = "Growth was strong [note: excludes FX] overall [E1]."
    cleaned, _, unknown = extract_citations(text, index)
    assert cleaned == "Growth was strong [note: excludes FX] overall [E1]."
    assert unknown == []


def test_bare_bracketed_number_is_not_treated_as_a_citation(index):
    """A footnote or list marker must not be rewritten - destroying real report
    content is as bad as passing a fake citation."""
    cleaned, _, unknown = extract_citations("Step [1] of the method.", index)
    assert cleaned == "Step [1] of the method."
    assert unknown == []


def test_bare_number_inside_a_marker_group_is_read_as_a_marker(index):
    cleaned, citations, unknown = extract_citations("Both [E1, 2] agree.", index)
    assert unknown == []
    assert cleaned == "Both [E1] [E2] agree."
    assert citations[0].evidence_ids == ["ev-1", "ev-2"]


def test_existing_unverified_label_survives_a_second_pass(index):
    """extract_citations must be idempotent: the synthesizer's output can be
    re-validated (e.g. after a verifier revision) without cascading damage."""
    once, _, _ = extract_citations("Claim [E9].", index)
    twice, _, unknown = extract_citations(once, index)
    assert twice == once == f"Claim {UNVERIFIED_LABEL}."
    assert unknown == []
