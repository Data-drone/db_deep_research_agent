"""Shared token accounting for LLM calls.

Every LLM-calling node funnels its response through :func:`accumulate_usage`,
so the running total in ``state["_token_usage"]`` covers the whole pipeline
rather than the synthesizer alone. Keeping the extraction in one place also
stops the streaming and non-streaming paths from disagreeing about the same
model's metadata.
"""

from __future__ import annotations

from typing import Any


def empty_usage() -> dict[str, int]:
    """A fresh zeroed counter. Returned as a new dict so callers can't alias it."""
    return {"input": 0, "output": 0}


def extract_usage(response: Any) -> dict[str, int]:
    """Pull an ``{input, output}`` token count out of an LLM response or chunk.

    Handles the three shapes LangChain chat models use: ``response_metadata``
    keyed by ``usage`` or ``token_usage``, and the standardised
    ``usage_metadata`` attribute — the last being the only one populated on
    stream chunks. Returns zeros when the model reports nothing, which is the
    honest answer for providers that omit usage on streamed responses.
    """
    meta = getattr(response, "response_metadata", None) or {}
    usage: Any = None
    if isinstance(meta, dict):
        usage = meta.get("usage") or meta.get("token_usage")
    if not usage:
        usage = getattr(response, "usage_metadata", None)
    if not isinstance(usage, dict):
        return empty_usage()
    return {
        "input": int(usage.get("input_tokens", usage.get("prompt_tokens", 0)) or 0),
        "output": int(usage.get("output_tokens", usage.get("completion_tokens", 0)) or 0),
    }


def extract_stream_usage(chunk: Any) -> dict[str, int]:
    """Pull an ``{input, output}`` count out of a single *stream chunk*.

    Deliberately narrower than :func:`extract_usage`: it reads only
    ``usage_metadata``, the field LangChain defines as per-chunk and additive
    (``AIMessageChunk.__add__`` sums it). ``response_metadata["usage"]`` carries
    no such guarantee on a chunk — a provider that repeats a cumulative total on
    every chunk would be multiplied by the chunk count if we summed it. Verified
    against ``databricks_langchain``: it accumulates usage internally and emits
    it once, as ``usage_metadata`` on a trailing empty chunk.
    """
    usage = getattr(chunk, "usage_metadata", None)
    if not isinstance(usage, dict):
        return empty_usage()
    return {
        "input": int(usage.get("input_tokens", usage.get("prompt_tokens", 0)) or 0),
        "output": int(usage.get("output_tokens", usage.get("completion_tokens", 0)) or 0),
    }


def add_usage(prior: dict[str, int] | None, extra: dict[str, int] | None) -> dict[str, int]:
    """Sum two counters, tolerating None or partially-populated dicts."""
    prior = prior or {}
    extra = extra or {}
    return {
        "input": int(prior.get("input", 0) or 0) + int(extra.get("input", 0) or 0),
        "output": int(prior.get("output", 0) or 0) + int(extra.get("output", 0) or 0),
    }


def accumulate_usage(state: Any, response: Any) -> dict[str, int]:
    """Running total after charging ``response`` against the state's counter.

    Nodes run sequentially and each returns the new total, so plain
    last-write-wins state merging still accumulates correctly — including when
    the verifier sends work back to the planner and nodes run a second time.
    """
    prior = state.get("_token_usage") if hasattr(state, "get") else None
    return add_usage(prior, extract_usage(response))


def charge(sink: dict[str, int], response: Any) -> None:
    """Add ``response``'s usage into ``sink``, in place.

    For nodes whose LLM calls happen inside helpers invoked in a loop, where
    passing one accumulator down is quieter than threading a usage tuple back
    through every call site.
    """
    sink.update(add_usage(sink, extract_usage(response)))
