"""Tests for shared token accounting."""

import pytest

from deep_research.token_usage import (
    accumulate_usage,
    add_usage,
    charge,
    empty_usage,
    extract_stream_usage,
    extract_usage,
)


class _Response:
    """Stand-in for a LangChain response with arbitrary metadata shapes."""

    def __init__(self, response_metadata=None, usage_metadata=None):
        if response_metadata is not None:
            self.response_metadata = response_metadata
        if usage_metadata is not None:
            self.usage_metadata = usage_metadata


def test_empty_usage_is_not_shared():
    a, b = empty_usage(), empty_usage()
    a["input"] = 5
    assert b["input"] == 0


@pytest.mark.parametrize(
    "response,expected",
    [
        # response_metadata["usage"] with OpenAI-style names
        (_Response({"usage": {"prompt_tokens": 10, "completion_tokens": 4}}), (10, 4)),
        # response_metadata["usage"] with LangChain-style names
        (_Response({"usage": {"input_tokens": 7, "output_tokens": 3}}), (7, 3)),
        # response_metadata["token_usage"] — the other Databricks shape
        (_Response({"token_usage": {"prompt_tokens": 2, "completion_tokens": 1}}), (2, 1)),
        # usage_metadata attribute — the only shape populated on stream chunks
        (_Response({}, {"input_tokens": 8, "output_tokens": 9}), (8, 9)),
        # nothing reported at all
        (_Response({}), (0, 0)),
        (_Response(), (0, 0)),
    ],
)
def test_extract_usage_handles_every_metadata_shape(response, expected):
    usage = extract_usage(response)
    assert (usage["input"], usage["output"]) == expected


def test_extract_usage_prefers_metadata_over_attribute():
    """response_metadata is the richer source; usage_metadata is the fallback."""
    r = _Response({"usage": {"input_tokens": 100, "output_tokens": 100}},
                  {"input_tokens": 1, "output_tokens": 1})
    assert extract_usage(r) == {"input": 100, "output": 100}


def test_extract_usage_survives_junk_metadata():
    assert extract_usage(_Response({"usage": "not-a-dict"})) == {"input": 0, "output": 0}
    assert extract_usage(_Response({"usage": {"input_tokens": None}})) == {"input": 0, "output": 0}


def test_add_usage_tolerates_none_and_partials():
    assert add_usage(None, None) == {"input": 0, "output": 0}
    assert add_usage({"input": 1}, {"output": 2}) == {"input": 1, "output": 2}
    assert add_usage({"input": 1, "output": 2}, {"input": 3, "output": 4}) == {
        "input": 4, "output": 6
    }


def test_accumulate_usage_adds_to_the_running_state_total():
    state = {"_token_usage": {"input": 50, "output": 20}}
    r = _Response({"usage": {"input_tokens": 5, "output_tokens": 2}})
    assert accumulate_usage(state, r) == {"input": 55, "output": 22}


def test_accumulate_usage_from_empty_state():
    r = _Response({"usage": {"input_tokens": 5, "output_tokens": 2}})
    assert accumulate_usage({}, r) == {"input": 5, "output": 2}


def test_charge_accumulates_in_place_across_a_loop():
    """The shape used by nodes whose LLM calls sit inside a helper."""
    sink = empty_usage()
    for _ in range(3):
        charge(sink, _Response({"usage": {"input_tokens": 10, "output_tokens": 1}}))
    assert sink == {"input": 30, "output": 3}


# ── Stream chunks ────────────────────────────────────────────────────────────
#
# extract_stream_usage is deliberately narrower than extract_usage: only
# usage_metadata is defined by LangChain as per-chunk and additive. Summing
# response_metadata across chunks would multiply a provider that repeats a
# cumulative total on every chunk by the chunk count.

class _Chunk:
    def __init__(self, usage_metadata=None, response_metadata=None):
        if usage_metadata is not None:
            self.usage_metadata = usage_metadata
        self.response_metadata = response_metadata or {}


def test_stream_usage_reads_usage_metadata():
    chunk = _Chunk(usage_metadata={"input_tokens": 40, "output_tokens": 7})
    assert extract_stream_usage(chunk) == {"input": 40, "output": 7}


def test_stream_usage_ignores_response_metadata():
    """The shape that would over-count if it were summed per chunk."""
    chunk = _Chunk(response_metadata={"usage": {"prompt_tokens": 100, "completion_tokens": 50}})
    assert extract_stream_usage(chunk) == {"input": 0, "output": 0}


def test_stream_usage_on_a_content_only_chunk_is_zero():
    assert extract_stream_usage(_Chunk()) == {"input": 0, "output": 0}


def test_stream_usage_sums_to_the_trailing_usage_chunk():
    """databricks_langchain accumulates internally and emits usage once, on a
    trailing empty chunk, so the total over a stream is that one chunk's value."""
    stream = [_Chunk() for _ in range(20)]
    stream.append(_Chunk(usage_metadata={"input_tokens": 512, "output_tokens": 300}))
    total = empty_usage()
    for chunk in stream:
        total = add_usage(total, extract_stream_usage(chunk))
    assert total == {"input": 512, "output": 300}


def test_stream_usage_tolerates_junk_metadata():
    chunk = _Chunk()
    chunk.usage_metadata = "not a dict"
    assert extract_stream_usage(chunk) == {"input": 0, "output": 0}
