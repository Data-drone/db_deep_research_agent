# Citations, Clarification & Token Display — Implementation Plan (v2)

> **For Claude:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.

**Goal:** Add enhanced text citations as styled badges, interactive user clarification with pre-graph pause/resume, and token usage display on completed messages.

**Architecture:** Citations are prompt + UI-only. Clarification uses a pre-graph pattern: clarifier runs standalone before the LangGraph pipeline, with a `ClarificationStore` abstraction (in-memory, swappable to Redis later) and `clarification_id` for idempotent answers. Token display extracts usage metadata from LLM responses (quick reply path; research path accumulation is best-effort since streaming doesn't expose usage). Single-process deployment; multi-worker is a documented future upgrade.

**Tech Stack:** Python 3.11, FastAPI, LangGraph, ChatDatabricks, React 19, TypeScript, Tailwind CSS 4, Vite 7.

**Revision notes (from GPT-5-4 review):**
- Clarification is now pre-graph (not mid-graph pause) — cleaner, fewer LangGraph compatibility issues
- Added `clarification_id` for idempotent answers and race-condition safety
- Added `ClarificationStore` abstraction with explicit status transitions (pending → answered/timed_out/cancelled)
- First-answer-wins semantics; late answers return `expired` status
- Max 1 clarification round per job
- Token usage scoped to "final answer" for quick reply; "research run total" for research mode, labeled accordingly
- Single-process limitation documented; `ClarificationStore` is an abstract base for future Redis swap

---

## Task 1: Enhanced Citations — Update Synthesizer Prompts

**Files:**
- Modify: `src/deep_research/prompts.py:110-127`

**Step 1: Update SYNTHESIZER_CHAT_SYSTEM prompt**

In `src/deep_research/prompts.py`, replace lines 110-115:

```python
SYNTHESIZER_CHAT_SYSTEM = """You are a research synthesizer. Given compressed findings and evidence, produce a clear, concise answer to the user's query.

- Lead with the direct answer
- Support with evidence
- Note uncertainties and limitations
- Cite sources inline using [Source: Tool — description] format, where Tool is the tool display name and description identifies the specific data (e.g. [Source: Genie — quarterly_revenue table] or [Source: Knowledge Assistant — ANZ FY2024 Annual Report])
- Every factual claim must have at least one citation
- Place citations immediately after the claim they support"""
```

**Step 2: Update SYNTHESIZER_REPORT_SYSTEM prompt**

Replace lines 117-127:

```python
SYNTHESIZER_REPORT_SYSTEM = """You are a research report writer. Given compressed findings and evidence, produce a structured report.

Report structure:
1. Executive Summary (2-3 sentences)
2. Key Findings (bulleted, with citations)
3. Detailed Analysis (organized by sub-topic)
4. Limitations & Uncertainties
5. What Would Change This Conclusion
6. Sources

Use [Source: Tool — description] format for inline citations, where Tool is the tool display name and description identifies the data (e.g. [Source: Genie — quarterly_revenue table] or [Source: Knowledge Assistant — ANZ FY2024 Annual Report]).
Every factual claim must have at least one citation. Be thorough but concise."""
```

**Step 3: Verify tests still pass**

```bash
cd /tmp/db_deep_research_agent && PYTHONPATH=src pytest tests/ -v --timeout=30 2>&1 | tail -20
```

**Step 4: Commit**

```bash
cd /tmp/db_deep_research_agent
git add src/deep_research/prompts.py
git commit -m "feat: improve citation format in synthesizer prompts"
```

---

## Task 2: Enhanced Citations — Enrich Evidence Context in Synthesizer

**Files:**
- Modify: `src/deep_research/nodes/synthesizer.py:58-61`

**Step 1: Update evidence formatting in synthesizer_node**

In `synthesizer_node`, replace the evidence block (lines 58-61):

```python
    if evidence:
        context_parts.append("Evidence:")
        for e in evidence:
            context_parts.append(f"- [{e.source_id}] {e.snippet}")
```

With:

```python
    if evidence:
        context_parts.append("Evidence (cite using [Source: Tool — description] format):")
        for e in evidence:
            tool_label = e.tool_that_produced_it or "Unknown"
            title_label = e.title or e.source_id
            context_parts.append(f"- [Source: {tool_label} — {title_label}] {e.snippet}")
```

**Step 2: Verify tests still pass**

```bash
cd /tmp/db_deep_research_agent && PYTHONPATH=src pytest tests/test_nodes/ -v --timeout=30
```

**Step 3: Commit**

```bash
cd /tmp/db_deep_research_agent
git add src/deep_research/nodes/synthesizer.py
git commit -m "feat: enrich evidence context with tool names for citations"
```

---

## Task 3: Enhanced Citations — Style Citation Badges in UI

**Files:**
- Modify: `ui/src/components/MessageBubble.tsx`
- Modify: `ui/src/types.ts`

**Step 1: Add tokenUsage to Message type**

In `ui/src/types.ts`, update the `Message` interface to add `tokenUsage`:

```typescript
export interface Message {
  id: string;
  role: "user" | "assistant";
  content: string;
  timestamp: string;
  jobId?: string;
  rating?: "thumbs_up" | "thumbs_down";
  streaming?: boolean;
  tableData?: TableData;
  tokenUsage?: { input: number; output: number; scope: "answer" | "job" };
}
```

**Step 2: Update MessageBubble to render citation badges and token usage**

Replace the entire content of `ui/src/components/MessageBubble.tsx` with:

```tsx
import ReactMarkdown from "react-markdown";
import remarkGfm from "remark-gfm";
import rehypeSanitize from "rehype-sanitize";
import type { Message } from "../types";
import { DataViz } from "./DataViz";

interface Props {
  message: Message;
  onRate: (id: string, rating: "thumbs_up" | "thumbs_down") => void;
}

const CITATION_RE = /\[Source:\s*([^\]]+)\]/g;

export function MessageBubble({ message, onRate }: Props) {
  const isUser = message.role === "user";

  // Convert [Source: X] markers to bold inline-code for badge styling
  const displayContent = !isUser
    ? message.content.replace(CITATION_RE, "**`📎 $1`**")
    : message.content;

  return (
    <div
      className={`flex animate-fade-in ${
        isUser ? "justify-end" : "items-start gap-3"
      }`}
    >
      {!isUser && (
        <div className="w-8 h-8 rounded-full bg-warm-sage flex items-center justify-center flex-shrink-0 mt-1">
          <span className="text-white text-xs font-semibold">RA</span>
        </div>
      )}
      <div className={`max-w-[720px] ${isUser ? "max-w-[85%]" : "flex-1 max-w-[720px]"}`}>
        <div
          className={`px-4 py-3 ${
            isUser
              ? "bg-warm-user rounded-xl rounded-br-sm text-warm-text"
              : "bg-warm-card shadow-sm rounded-xl rounded-bl-sm text-warm-text"
          }`}
        >
          {isUser ? (
            <p className="leading-relaxed text-[15px]">{message.content}</p>
          ) : (
            <>
              <div className="leading-relaxed text-[15px] prose prose-sm max-w-none prose-headings:text-warm-text prose-p:text-warm-text prose-strong:text-warm-text prose-a:text-warm-accent prose-code:bg-warm-sage/10 prose-code:text-warm-sage prose-code:text-xs prose-code:px-1.5 prose-code:py-0.5 prose-code:rounded-md prose-code:font-medium prose-code:before:content-none prose-code:after:content-none">
                <ReactMarkdown
                  remarkPlugins={[remarkGfm]}
                  rehypePlugins={[rehypeSanitize]}
                >
                  {displayContent}
                </ReactMarkdown>
                {message.streaming && (
                  <span className="streaming-cursor" aria-hidden="true">
                    &#9610;
                  </span>
                )}
              </div>
              {message.tableData && <DataViz data={message.tableData} />}
            </>
          )}
        </div>
        {!isUser && !message.streaming && (
          <div className="mt-1.5 flex items-center gap-1">
            <button
              className={`px-2 py-0.5 rounded-md text-sm transition-colors duration-150 ${
                message.rating === "thumbs_up"
                  ? "bg-warm-accent/10 border border-warm-accent/30"
                  : "border border-transparent hover:border-warm-border text-warm-text-secondary hover:text-warm-text"
              }`}
              onClick={() => onRate(message.id, "thumbs_up")}
              aria-label="Rate response as helpful"
              title="Helpful"
            >
              👍
            </button>
            <button
              className={`px-2 py-0.5 rounded-md text-sm transition-colors duration-150 ${
                message.rating === "thumbs_down"
                  ? "bg-warm-rose/10 border border-warm-rose/30"
                  : "border border-transparent hover:border-warm-border text-warm-text-secondary hover:text-warm-text"
              }`}
              onClick={() => onRate(message.id, "thumbs_down")}
              aria-label="Rate response as not helpful"
              title="Not helpful"
            >
              👎
            </button>
            {message.tokenUsage && (
              <span className="ml-auto text-[11px] text-warm-text-secondary tabular-nums" title={message.tokenUsage.scope === "job" ? "Total tokens for full research run" : "Tokens for this answer"}>
                {(message.tokenUsage.input + message.tokenUsage.output).toLocaleString()} tokens
              </span>
            )}
          </div>
        )}
      </div>
    </div>
  );
}
```

Key changes:
- Citations `[Source: Genie — table]` become `**\`📎 Genie — table\`**` before markdown
- Tailwind prose-code styles make inline code look like citation badges (sage green bg, rounded)
- Token usage label with scope-aware tooltip
- `prose-code:before:content-none prose-code:after:content-none` removes backtick rendering

**Step 3: Verify TypeScript compiles**

```bash
cd /tmp/db_deep_research_agent/ui && npx tsc -b 2>&1
```

**Step 4: Commit**

```bash
cd /tmp/db_deep_research_agent
git add ui/src/components/MessageBubble.tsx ui/src/types.ts
git commit -m "feat(ui): style citation markers as badges, add tokenUsage type"
```

---

## Task 4: Clarification — ClarificationStore + Updated Clarifier

**Files:**
- Create: `src/deep_research/api/clarification.py`
- Modify: `src/deep_research/prompts.py:3-17`
- Modify: `src/deep_research/nodes/clarifier.py`

**Step 1: Create ClarificationStore abstraction**

Create `src/deep_research/api/clarification.py`:

```python
"""Clarification state management.

Uses in-memory storage (single-process). To support multi-worker deployments,
swap InMemoryClarificationStore for a Redis-backed implementation.
"""

from __future__ import annotations

import asyncio
import logging
import uuid
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Literal

logger = logging.getLogger(__name__)

ClarificationStatus = Literal["pending", "answered", "timed_out", "cancelled"]


@dataclass
class ClarificationState:
    clarification_id: str
    job_id: str
    question: str
    options: list[str]
    best_guess: str
    status: ClarificationStatus = "pending"
    answer: str | None = None
    created_at: datetime = field(default_factory=lambda: datetime.now(timezone.utc))


class ClarificationStore(ABC):
    """Abstract base for clarification state storage."""

    @abstractmethod
    def create(self, job_id: str, question: str, options: list[str], best_guess: str) -> ClarificationState:
        ...

    @abstractmethod
    def get(self, job_id: str) -> ClarificationState | None:
        ...

    @abstractmethod
    def submit_answer(self, job_id: str, clarification_id: str, answer: str) -> ClarificationState | None:
        """Submit answer. Returns updated state if accepted, None if expired/invalid.
        First-answer-wins: subsequent answers are rejected."""
        ...

    @abstractmethod
    def mark_timed_out(self, job_id: str) -> None:
        ...

    @abstractmethod
    def mark_cancelled(self, job_id: str) -> None:
        ...


class InMemoryClarificationStore(ClarificationStore):
    """In-memory implementation. Works for single-process deployments.

    Limitation: Not safe across multiple workers/replicas.
    For multi-worker, implement a Redis-backed ClarificationStore.
    """

    def __init__(self) -> None:
        self._states: dict[str, ClarificationState] = {}
        self._events: dict[str, asyncio.Event] = {}

    def create(self, job_id: str, question: str, options: list[str], best_guess: str) -> ClarificationState:
        cid = f"clr-{uuid.uuid4().hex[:12]}"
        state = ClarificationState(
            clarification_id=cid,
            job_id=job_id,
            question=question,
            options=options,
            best_guess=best_guess,
        )
        self._states[job_id] = state
        self._events[job_id] = asyncio.Event()
        return state

    def get(self, job_id: str) -> ClarificationState | None:
        return self._states.get(job_id)

    def submit_answer(self, job_id: str, clarification_id: str, answer: str) -> ClarificationState | None:
        state = self._states.get(job_id)
        if state is None:
            return None
        if state.clarification_id != clarification_id:
            return None
        if state.status != "pending":
            return None  # First-answer-wins / already resolved
        state.status = "answered"
        state.answer = answer
        event = self._events.get(job_id)
        if event:
            event.set()
        return state

    def mark_timed_out(self, job_id: str) -> None:
        state = self._states.get(job_id)
        if state and state.status == "pending":
            state.status = "timed_out"
        self._cleanup(job_id)

    def mark_cancelled(self, job_id: str) -> None:
        state = self._states.get(job_id)
        if state and state.status == "pending":
            state.status = "cancelled"
        self._cleanup(job_id)

    def get_event(self, job_id: str) -> asyncio.Event | None:
        return self._events.get(job_id)

    def _cleanup(self, job_id: str) -> None:
        self._events.pop(job_id, None)
```

**Step 2: Update CLARIFIER_SYSTEM prompt for dual-mode output**

In `src/deep_research/prompts.py`, replace the `CLARIFIER_SYSTEM` string (lines 3-17):

```python
CLARIFIER_SYSTEM = """You are a research query refiner. Analyze the user's query and decide if it needs clarification.

If conversation history is present, the user may be asking a follow-up question. Resolve pronouns and references using the prior conversation context.

DECISION RULES:
- If the query is clear enough to research (even if it could be more specific), return a clarified version
- If the query is genuinely ambiguous (multiple distinct interpretations that would lead to very different research paths), ask for clarification
- Err on the side of NOT asking — only ask when the ambiguity would waste significant research effort

For CLEAR queries, output:
{{"clarified_query": "<the refined query>"}}

For AMBIGUOUS queries, output:
{{"needs_clarification": true, "question": "<a short, specific question>", "options": ["<option 1>", "<option 2>", "<option 3 if needed>"], "best_guess": "<your best interpretation as a clarified query>"}}

Rules for clarification questions:
- Maximum 3 options
- Options must be distinct and cover the likely interpretations
- best_guess is used if the user doesn't respond in time
- question should be one sentence

Output valid JSON only."""
```

**Step 3: Update clarifier_node for dual-mode + short-circuit**

Replace the content of `src/deep_research/nodes/clarifier.py`:

```python
"""Clarifier node — refines and focuses user queries for research."""

from __future__ import annotations

import json
import logging
from typing import Any

from deep_research.prompts import CLARIFIER_SYSTEM
from deep_research.state import ResearchState

logger = logging.getLogger(__name__)


async def clarifier_node(state: ResearchState, *, model: Any) -> dict:
    """Refine and focus the user query. May request clarification for ambiguous queries.

    When called with clarified_query already set (pre-graph clarification resolved it),
    short-circuits and returns the existing value.

    Returns either:
    - {"clarified_query": "..."} for clear queries or pre-resolved
    - {"clarified_query": best_guess, "needs_clarification": True,
       "clarification_question": "...", "clarification_options": [...]}
    """
    # Short-circuit if pre-graph clarification already resolved this
    if state.get("clarified_query"):
        return {"clarified_query": state["clarified_query"]}

    messages: list[dict[str, str]] = [{"role": "system", "content": CLARIFIER_SYSTEM}]

    _MAX_HISTORY_TURNS = 10
    for msg in state.get("conversation_history", [])[-_MAX_HISTORY_TURNS:]:
        role = msg.get("role")
        content = msg.get("content")
        if role in ("user", "assistant") and isinstance(content, str):
            messages.append({"role": role, "content": content})

    messages.append({"role": "user", "content": state["user_query"]})

    response = await model.ainvoke(messages)

    try:
        result = json.loads(response.content)
    except (json.JSONDecodeError, AttributeError):
        return {"clarified_query": state["user_query"]}

    if result.get("needs_clarification"):
        return {
            "clarified_query": result.get("best_guess", state["user_query"]),
            "needs_clarification": True,
            "clarification_question": result.get("question", "Could you clarify your question?"),
            "clarification_options": result.get("options", []),
        }

    return {"clarified_query": result.get("clarified_query", state["user_query"])}
```

**Step 4: Add clarification fields to ResearchState**

In `src/deep_research/state.py`, add these fields to `ResearchState` after `clarified_query: str | None` (line 29):

```python
    needs_clarification: bool
    clarification_question: str
    clarification_options: list[str]
```

And update `create_initial_state` to include defaults (add after `clarified_query=None,` on line 88):

```python
        needs_clarification=False,
        clarification_question="",
        clarification_options=[],
```

**Step 5: Write tests for ClarificationStore**

Create `tests/test_clarification.py`:

```python
"""Tests for ClarificationStore."""

import asyncio
import pytest

from deep_research.api.clarification import InMemoryClarificationStore


def test_create_and_get():
    store = InMemoryClarificationStore()
    state = store.create("job-1", "Which company?", ["A", "B"], "A")
    assert state.job_id == "job-1"
    assert state.status == "pending"
    assert state.clarification_id.startswith("clr-")

    fetched = store.get("job-1")
    assert fetched is state


def test_submit_answer_first_wins():
    store = InMemoryClarificationStore()
    state = store.create("job-1", "Which?", ["A", "B"], "A")

    result = store.submit_answer("job-1", state.clarification_id, "B")
    assert result is not None
    assert result.status == "answered"
    assert result.answer == "B"

    # Second answer rejected
    result2 = store.submit_answer("job-1", state.clarification_id, "A")
    assert result2 is None


def test_submit_answer_wrong_clarification_id():
    store = InMemoryClarificationStore()
    store.create("job-1", "Which?", ["A", "B"], "A")

    result = store.submit_answer("job-1", "wrong-id", "B")
    assert result is None


def test_submit_answer_unknown_job():
    store = InMemoryClarificationStore()
    result = store.submit_answer("nonexistent", "clr-123", "B")
    assert result is None


def test_mark_timed_out():
    store = InMemoryClarificationStore()
    state = store.create("job-1", "Which?", ["A", "B"], "A")
    store.mark_timed_out("job-1")
    assert state.status == "timed_out"

    # Late answer rejected
    result = store.submit_answer("job-1", state.clarification_id, "B")
    assert result is None


def test_mark_cancelled():
    store = InMemoryClarificationStore()
    state = store.create("job-1", "Which?", ["A", "B"], "A")
    store.mark_cancelled("job-1")
    assert state.status == "cancelled"


def test_event_is_set_on_answer():
    store = InMemoryClarificationStore()
    state = store.create("job-1", "Which?", ["A", "B"], "A")
    event = store.get_event("job-1")
    assert event is not None
    assert not event.is_set()

    store.submit_answer("job-1", state.clarification_id, "B")
    assert event.is_set()


def test_get_nonexistent():
    store = InMemoryClarificationStore()
    assert store.get("nonexistent") is None
```

**Step 6: Run tests**

```bash
cd /tmp/db_deep_research_agent && PYTHONPATH=src pytest tests/test_clarification.py -v --timeout=30
```

**Step 7: Run all tests**

```bash
cd /tmp/db_deep_research_agent && PYTHONPATH=src pytest tests/ -v --timeout=30 2>&1 | tail -20
```

**Step 8: Commit**

```bash
cd /tmp/db_deep_research_agent
git add src/deep_research/api/clarification.py src/deep_research/prompts.py src/deep_research/nodes/clarifier.py src/deep_research/state.py tests/test_clarification.py
git commit -m "feat: add ClarificationStore, dual-mode clarifier, pre-graph pattern"
```

---

## Task 5: Clarification — Backend Pre-Graph Flow + /clarify Endpoint

**Files:**
- Modify: `src/deep_research/api/app.py`
- Modify: `src/deep_research/api/jobs.py`

**Step 1: Add token_usage to JobStatus**

In `src/deep_research/api/jobs.py`, add to the `JobStatus` dataclass (after `error: str | None = None`):

```python
    token_usage: dict[str, int] | None = None
```

**Step 2: Add _run_research_with_clarification + /clarify endpoint + token extraction to app.py**

In `src/deep_research/api/app.py`:

Add import at the top (after existing imports):

```python
from deep_research.api.clarification import InMemoryClarificationStore
```

Add `_extract_token_usage` helper before `create_app`:

```python
def _extract_token_usage(response: Any) -> dict[str, int]:
    """Extract token usage from an LLM response's metadata."""
    meta = getattr(response, "response_metadata", {}) or {}
    usage = meta.get("usage", {})
    return {
        "input": usage.get("input_tokens", 0) or usage.get("prompt_tokens", 0),
        "output": usage.get("output_tokens", 0) or usage.get("completion_tokens", 0),
    }
```

Add `_run_research_with_clarification` after `_run_quick_reply`:

```python
async def _run_research_with_clarification(
    model: Any,
    graph: Any,
    clarification_store: InMemoryClarificationStore,
    job_manager: JobManager,
    job_id: str,
    initial_state: dict,
    session_manager: SessionManager,
    session_id: str,
) -> None:
    """Run clarifier pre-graph, handle clarification if needed, then run full graph."""
    from deep_research.nodes.clarifier import clarifier_node

    try:
        job_manager.update_state(job_id, "running", current_node="clarifier")
        job_manager.push_event(job_id, {"type": "node_started", "node": "clarifier"})

        # Run clarifier standalone (pre-graph)
        clarifier_result = await clarifier_node(initial_state, model=model)

        if clarifier_result.get("needs_clarification"):
            question = clarifier_result.get("clarification_question", "")
            options = clarifier_result.get("clarification_options", [])
            best_guess = clarifier_result.get("clarified_query", initial_state["user_query"])

            # Create persisted clarification state BEFORE emitting SSE
            clr_state = clarification_store.create(job_id, question, options, best_guess)

            # Push event to UI with clarification_id
            job_manager.push_event(job_id, {
                "type": "clarification_needed",
                "clarification_id": clr_state.clarification_id,
                "question": question,
                "options": options,
            })

            # Wait for user answer (30s timeout)
            wait_event = clarification_store.get_event(job_id)
            if wait_event:
                try:
                    await asyncio.wait_for(wait_event.wait(), timeout=30.0)
                    updated = clarification_store.get(job_id)
                    if updated and updated.status == "answered" and updated.answer:
                        initial_state["clarified_query"] = updated.answer
                        job_manager.push_event(job_id, {
                            "type": "clarification_resolved",
                            "answer": updated.answer,
                        })
                    else:
                        initial_state["clarified_query"] = best_guess
                except asyncio.TimeoutError:
                    clarification_store.mark_timed_out(job_id)
                    initial_state["clarified_query"] = best_guess
                    job_manager.push_event(job_id, {
                        "type": "clarification_timeout",
                        "best_guess": best_guess,
                    })
        else:
            initial_state["clarified_query"] = clarifier_result.get(
                "clarified_query", initial_state["user_query"]
            )

        # Ensure graph's clarifier short-circuits
        initial_state["needs_clarification"] = False

        # Run the full graph
        await _run_graph(
            graph, job_manager, job_id, initial_state,
            session_manager=session_manager,
            session_id=session_id,
        )
    except Exception as exc:
        logger.exception(f"Research with clarification failed for job {job_id}")
        job_manager.update_state(
            job_id, "failed", error=f"Research execution failed: {exc}"
        )
        job_manager.push_event(job_id, {"type": "failed", "error": f"Research execution failed: {exc}"})
```

Inside `create_app`, add the clarification store after `session_manager`:

```python
    clarification_store = InMemoryClarificationStore()
```

Add the `/clarify` endpoint inside `create_app` (after `cancel_research`):

```python
    class ClarifyRequest(BaseModel):
        clarification_id: str = Field(..., min_length=1)
        answer: str = Field(..., min_length=1)

    @app.post("/api/research/{job_id}/clarify")
    async def submit_clarification(job_id: str, req: ClarifyRequest):
        try:
            job_manager.get_status(job_id)
        except KeyError:
            raise HTTPException(status_code=404, detail="Job not found")

        clr_state = clarification_store.get(job_id)
        if clr_state is None:
            raise HTTPException(status_code=400, detail="No pending clarification for this job")

        if clr_state.status != "pending":
            return {"status": "expired", "used_answer": False}

        result = clarification_store.submit_answer(job_id, req.clarification_id, req.answer)
        if result is None:
            return {"status": "expired", "used_answer": False}

        return {"status": "accepted", "job_id": job_id}
```

Add pending clarification to the job status endpoint. In `get_research_status`, add after the existing return dict:

Replace the return statement in `get_research_status`:

```python
        response = {
            "job_id": job_id,
            "status": status.state,
            "result": status.result,
            "current_node": status.current_node,
            "error": status.error,
        }
        # Include pending clarification for reconnect recovery
        clr_state = clarification_store.get(job_id)
        if clr_state and clr_state.status == "pending":
            response["pending_clarification"] = {
                "clarification_id": clr_state.clarification_id,
                "question": clr_state.question,
                "options": clr_state.options,
            }
        return response
```

**Step 3: Update research path to use _run_research_with_clarification**

In `submit_research`, replace the `asyncio.create_task(_run_graph(...))` block in the research path (the `else:` branch at line 278) with:

```python
            asyncio.create_task(
                _run_research_with_clarification(
                    model=getattr(request.app.state, "model", None),
                    graph=graph,
                    clarification_store=clarification_store,
                    job_manager=job_manager,
                    job_id=job_id,
                    initial_state=initial_state,
                    session_manager=session_manager,
                    session_id=session.session_id,
                )
            )
```

Note: This requires `model` on `app.state`, which is already stored in `main.py:123`.

**Step 4: Update _run_quick_reply to include token usage**

In `_run_quick_reply`, after the `response = await model.ainvoke(...)` line and `reply = response.content...`:

```python
            token_usage = _extract_token_usage(response)
```

And replace the completed event push:

```python
            completed_event: dict[str, Any] = {"type": "completed", "result": reply}
            if token_usage and (token_usage["input"] > 0 or token_usage["output"] > 0):
                completed_event["token_usage"] = {**token_usage, "scope": "answer"}
            job_manager.push_event(job_id, completed_event)
```

(Remove the old `job_manager.push_event(job_id, {"type": "completed", "result": reply})` line.)

**Step 5: Write tests for /clarify endpoint**

Add to `tests/test_api.py`:

```python
async def test_clarify_unknown_job(client):
    """Clarify endpoint returns 404 for unknown job."""
    response = await client.post(
        "/api/research/nonexistent/clarify",
        json={"clarification_id": "clr-123", "answer": "test"},
    )
    assert response.status_code == 404


async def test_clarify_no_pending(quick_client):
    """Clarify returns 400 if job has no pending clarification."""
    res = await quick_client.post("/api/research", json={
        "query": "test", "tools": [], "response_mode": "quick"
    })
    job_id = res.json()["job_id"]
    await asyncio.sleep(0.1)

    response = await quick_client.post(
        f"/api/research/{job_id}/clarify",
        json={"clarification_id": "clr-123", "answer": "option A"},
    )
    assert response.status_code == 400


async def test_job_status_no_clarification(quick_client):
    """Job status without clarification has no pending_clarification field."""
    res = await quick_client.post("/api/research", json={
        "query": "test", "tools": [], "response_mode": "quick"
    })
    await asyncio.sleep(0.2)
    status = await quick_client.get(f"/api/research/{res.json()['job_id']}")
    assert "pending_clarification" not in status.json()
```

**Step 6: Run tests**

```bash
cd /tmp/db_deep_research_agent && PYTHONPATH=src pytest tests/test_api.py -v --timeout=30
```

**Step 7: Run all tests**

```bash
cd /tmp/db_deep_research_agent && PYTHONPATH=src pytest tests/ -v --timeout=30 2>&1 | tail -20
```

**Step 8: Commit**

```bash
cd /tmp/db_deep_research_agent
git add src/deep_research/api/app.py src/deep_research/api/jobs.py tests/test_api.py
git commit -m "feat: pre-graph clarification with ClarificationStore, /clarify endpoint, token extraction"
```

---

## Task 6: Clarification — Frontend ClarificationPrompt Component + API

**Files:**
- Create: `ui/src/components/ClarificationPrompt.tsx`
- Modify: `ui/src/types.ts`
- Modify: `ui/src/api.ts`

**Step 1: Add ClarificationRequest type to types.ts**

In `ui/src/types.ts`, add:

```typescript
export interface ClarificationRequest {
  jobId: string;
  clarificationId: string;
  question: string;
  options: string[];
}
```

**Step 2: Add submitClarification to api.ts**

In `ui/src/api.ts`, add:

```typescript
export async function submitClarification(
  jobId: string,
  clarificationId: string,
  answer: string
): Promise<void> {
  await client.post(`/api/research/${jobId}/clarify`, {
    clarification_id: clarificationId,
    answer,
  });
}
```

And update the `streamJob` `onEvent` type to include clarification fields:

```typescript
  onEvent: (event: {
    type: string;
    node?: string;
    result?: string;
    error?: string;
    content?: string;
    columns?: { name: string; type: string }[];
    rows?: string[][];
    sql?: string;
    chart?: { type: string; x: string; y: string[]; title?: string };
    clarification_id?: string;
    question?: string;
    options?: string[];
    best_guess?: string;
    answer?: string;
    token_usage?: { input: number; output: number; scope: string };
  }) => void,
```

**Step 3: Create ClarificationPrompt component**

Create `ui/src/components/ClarificationPrompt.tsx`:

```tsx
import { useState } from "react";
import type { ClarificationRequest } from "../types";

interface Props {
  request: ClarificationRequest;
  onAnswer: (answer: string) => void;
}

export function ClarificationPrompt({ request, onAnswer }: Props) {
  const [customInput, setCustomInput] = useState("");

  return (
    <div className="flex items-start gap-3 animate-fade-in">
      <div className="w-8 h-8 rounded-full bg-warm-amber flex items-center justify-center flex-shrink-0 mt-1">
        <span className="text-white text-xs font-semibold">?</span>
      </div>
      <div className="flex-1 max-w-[720px]">
        <div className="px-4 py-3 bg-warm-card shadow-sm rounded-xl rounded-bl-sm border border-warm-amber/30">
          <p className="text-[15px] font-medium text-warm-text mb-3">
            {request.question}
          </p>
          <div className="flex flex-wrap gap-2 mb-3">
            {request.options.map((option) => (
              <button
                key={option}
                className="px-3 py-1.5 rounded-lg border border-warm-border bg-warm-bg text-warm-text text-sm transition-all duration-150 hover:border-warm-accent hover:bg-warm-accent/5 hover:shadow-sm"
                onClick={() => onAnswer(option)}
              >
                {option}
              </button>
            ))}
          </div>
          <div className="flex gap-2">
            <input
              type="text"
              value={customInput}
              onChange={(e) => setCustomInput(e.target.value)}
              onKeyDown={(e) => {
                if (e.key === "Enter" && customInput.trim()) {
                  onAnswer(customInput.trim());
                }
              }}
              placeholder="Or type your own answer..."
              className="flex-1 px-3 py-1.5 border border-warm-border rounded-lg bg-warm-input text-warm-text text-sm placeholder:text-warm-placeholder focus:outline-none focus:border-warm-border-focus focus:ring-2 focus:ring-warm-accent/20 transition-colors duration-150"
            />
            <button
              className="px-3 py-1.5 rounded-lg bg-warm-accent text-white text-sm font-medium transition-colors duration-150 hover:bg-warm-accent-hover disabled:opacity-40"
              disabled={!customInput.trim()}
              onClick={() => {
                if (customInput.trim()) onAnswer(customInput.trim());
              }}
            >
              Send
            </button>
          </div>
        </div>
      </div>
    </div>
  );
}
```

**Step 4: Verify build**

```bash
cd /tmp/db_deep_research_agent/ui && npx tsc -b 2>&1
```

**Step 5: Commit**

```bash
cd /tmp/db_deep_research_agent
git add ui/src/components/ClarificationPrompt.tsx ui/src/types.ts ui/src/api.ts
git commit -m "feat(ui): ClarificationPrompt component, API function, types"
```

---

## Task 7: Clarification + Token Usage — Wire into useResearch, ChatPanel, App

**Files:**
- Modify: `ui/src/hooks/useResearch.ts`
- Modify: `ui/src/components/ChatPanel.tsx`
- Modify: `ui/src/App.tsx`

**Step 1: Update useResearch hook**

Replace the full content of `ui/src/hooks/useResearch.ts`:

```typescript
import { useCallback, useEffect, useRef, useState } from "react";
import { submitResearch, streamJob, cancelJob, submitFeedback, submitClarification } from "../api";
import type { Message, JobStatus, OutputMode, ResponseMode, TableData, ClarificationRequest } from "../types";

export function useResearch() {
  const [messages, setMessages] = useState<Message[]>([]);
  const [currentJob, setCurrentJob] = useState<JobStatus | null>(null);
  const [isLoading, setIsLoading] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const [clarificationRequest, setClarificationRequest] = useState<ClarificationRequest | null>(null);

  const activeJobIdRef = useRef<string | null>(null);
  const cleanupRef = useRef<(() => void) | null>(null);
  const mountedRef = useRef(true);
  const sessionIdRef = useRef<string | null>(null);
  const pendingTableDataRef = useRef<TableData | null>(null);

  useEffect(() => {
    return () => {
      mountedRef.current = false;
      cleanupRef.current?.();
      cleanupRef.current = null;
    };
  }, []);

  const stopStream = useCallback(() => {
    activeJobIdRef.current = null;
    cleanupRef.current?.();
    cleanupRef.current = null;
  }, []);

  const startStream = useCallback(
    (jobId: string) => {
      if (!mountedRef.current) return;

      const cleanup = streamJob(
        jobId,
        (event) => {
          if (!mountedRef.current || activeJobIdRef.current !== jobId) return;

          if (event.type === "node_started" && event.node) {
            setCurrentJob((prev) =>
              prev ? { ...prev, status: "running", current_node: event.node } : prev
            );
          } else if (event.type === "token" && event.content) {
            const tokenText = event.content;
            setMessages((prev) => {
              const lastMsg = prev[prev.length - 1];
              if (lastMsg?.streaming && lastMsg.jobId === jobId) {
                return [
                  ...prev.slice(0, -1),
                  { ...lastMsg, content: lastMsg.content + tokenText },
                ];
              } else {
                return [
                  ...prev,
                  {
                    id: crypto.randomUUID(),
                    role: "assistant" as const,
                    content: tokenText,
                    timestamp: new Date().toISOString(),
                    jobId,
                    streaming: true,
                  },
                ];
              }
            });
          } else if (event.type === "table_data") {
            pendingTableDataRef.current = {
              columns: event.columns || [],
              rows: event.rows || [],
              sql: event.sql,
              chart: event.chart as TableData["chart"],
            };
          } else if (event.type === "clarification_needed") {
            setClarificationRequest({
              jobId,
              clarificationId: event.clarification_id || "",
              question: event.question || "Could you clarify your question?",
              options: event.options || [],
            });
          } else if (event.type === "clarification_resolved" || event.type === "clarification_timeout") {
            setClarificationRequest(null);
            if (event.type === "clarification_timeout" && event.best_guess) {
              setMessages((prev) => [
                ...prev,
                {
                  id: crypto.randomUUID(),
                  role: "assistant" as const,
                  content: `No response received — proceeding with: "${event.best_guess}"`,
                  timestamp: new Date().toISOString(),
                  jobId,
                },
              ]);
            }
          } else if (event.type === "completed") {
            const pendingTableData = pendingTableDataRef.current;
            pendingTableDataRef.current = null;
            const tokenUsage = event.token_usage
              ? { input: event.token_usage.input, output: event.token_usage.output, scope: (event.token_usage.scope || "job") as "answer" | "job" }
              : undefined;
            stopStream();
            setIsLoading(false);
            setCurrentJob(null);
            setClarificationRequest(null);
            setMessages((prev) => {
              const lastMsg = prev[prev.length - 1];
              if (lastMsg?.streaming && lastMsg.jobId === jobId) {
                return [
                  ...prev.slice(0, -1),
                  {
                    ...lastMsg,
                    content: event.result || lastMsg.content,
                    streaming: false,
                    tableData: pendingTableData ?? undefined,
                    tokenUsage,
                  },
                ];
              } else if (event.result) {
                return [
                  ...prev,
                  {
                    id: crypto.randomUUID(),
                    role: "assistant" as const,
                    content: event.result,
                    timestamp: new Date().toISOString(),
                    jobId,
                    tableData: pendingTableData ?? undefined,
                    tokenUsage,
                  },
                ];
              }
              return prev;
            });
          } else if (event.type === "failed") {
            stopStream();
            setIsLoading(false);
            setCurrentJob(null);
            setClarificationRequest(null);
            setMessages((prev) => [
              ...prev,
              {
                id: crypto.randomUUID(),
                role: "assistant",
                content: event.error || "Research failed. Please try again.",
                timestamp: new Date().toISOString(),
                jobId,
              },
            ]);
          }
        },
        () => {
          if (!mountedRef.current || activeJobIdRef.current !== jobId) return;
          stopStream();
          setIsLoading(false);
          setCurrentJob(null);
          setError("Lost connection to research stream.");
        }
      );

      cleanupRef.current = cleanup;
    },
    [stopStream]
  );

  const send = useCallback(
    async (query: string, tools: string[], outputMode: OutputMode, responseMode: ResponseMode) => {
      if (isLoading) return;

      setError(null);

      const userMsg: Message = {
        id: crypto.randomUUID(),
        role: "user",
        content: query,
        timestamp: new Date().toISOString(),
      };
      setMessages((prev) => [...prev, userMsg]);
      setIsLoading(true);

      stopStream();

      try {
        const { job_id, session_id } = await submitResearch(
          query, tools, outputMode, responseMode, sessionIdRef.current ?? undefined
        );
        if (!mountedRef.current) return;

        sessionIdRef.current = session_id;
        activeJobIdRef.current = job_id;
        setCurrentJob({ job_id, status: "pending" });

        startStream(job_id);
      } catch {
        if (!mountedRef.current) return;
        setIsLoading(false);
        setError("Failed to submit research query.");
        setMessages((prev) => [
          ...prev,
          {
            id: crypto.randomUUID(),
            role: "assistant",
            content: "Failed to submit research query.",
            timestamp: new Date().toISOString(),
          },
        ]);
      }
    },
    [isLoading, stopStream, startStream]
  );

  const cancel = useCallback(async () => {
    const jobId = activeJobIdRef.current;
    if (!jobId) return;

    stopStream();
    setIsLoading(false);
    setCurrentJob(null);
    setClarificationRequest(null);

    try {
      await cancelJob(jobId);
    } catch {
      // Best-effort cancellation
    }
  }, [stopStream]);

  const answerClarification = useCallback(async (answer: string) => {
    const req = clarificationRequest;
    if (!req) return;

    setClarificationRequest(null);

    // Show the user's answer as a message
    setMessages((prev) => [
      ...prev,
      {
        id: crypto.randomUUID(),
        role: "user",
        content: answer,
        timestamp: new Date().toISOString(),
      },
    ]);

    try {
      await submitClarification(req.jobId, req.clarificationId, answer);
    } catch {
      // Best-effort — the pipeline will timeout and use best_guess
    }
  }, [clarificationRequest]);

  const rate = useCallback(
    async (messageId: string, rating: "thumbs_up" | "thumbs_down") => {
      let targetJobId: string | undefined;

      setMessages((prev) =>
        prev.map((m) => {
          if (m.id === messageId) {
            targetJobId = m.jobId;
            return { ...m, rating };
          }
          return m;
        })
      );

      if (targetJobId) {
        try {
          await submitFeedback(targetJobId, rating);
        } catch {
          // Feedback submission is best-effort
        }
      }
    },
    []
  );

  return { messages, currentJob, isLoading, error, clarificationRequest, send, cancel, answerClarification, rate };
}
```

**Step 2: Update ChatPanel to render ClarificationPrompt**

In `ui/src/components/ChatPanel.tsx`, add imports:

```typescript
import { ClarificationPrompt } from "./ClarificationPrompt";
import type { ClarificationRequest } from "../types";
```

Add props to Props interface:

```typescript
  clarificationRequest: ClarificationRequest | null;
  onAnswerClarification: (answer: string) => void;
```

Destructure new props in the component signature.

Add in JSX after the messages map and before `{showTypingIndicator && ...}`:

```tsx
          {clarificationRequest && (
            <ClarificationPrompt
              request={clarificationRequest}
              onAnswer={onAnswerClarification}
            />
          )}
```

**Step 3: Update App.tsx**

In `ui/src/App.tsx`, destructure new values from `useResearch()`:

```typescript
  const { messages, currentJob, isLoading, error, clarificationRequest, send, cancel, answerClarification, rate } =
    useResearch();
```

Pass new props to ChatPanel:

```tsx
              clarificationRequest={clarificationRequest}
              onAnswerClarification={answerClarification}
```

**Step 4: Verify build**

```bash
cd /tmp/db_deep_research_agent/ui && npx tsc -b 2>&1
```

**Step 5: Commit**

```bash
cd /tmp/db_deep_research_agent
git add ui/src/hooks/useResearch.ts ui/src/components/ChatPanel.tsx ui/src/App.tsx
git commit -m "feat(ui): wire clarification + token usage into useResearch, ChatPanel, App"
```

---

## Task 8: Token Usage — Backend Accumulation for Research Path

**Files:**
- Modify: `src/deep_research/nodes/synthesizer.py`
- Modify: `src/deep_research/state.py`
- Modify: `src/deep_research/api/app.py` (_run_graph)

**Step 1: Add _token_usage to ResearchState**

In `src/deep_research/state.py`, add to `ResearchState` (in the "Runtime-only" section):

```python
    _token_usage: dict[str, int]
```

And in `create_initial_state`, add (at end, before closing paren):

```python
        _token_usage={"input": 0, "output": 0},
```

**Step 2: Accumulate token usage in synthesizer**

In `src/deep_research/nodes/synthesizer.py`, update the return to include token usage.

Replace the streaming and non-streaming paths at the bottom of the function:

```python
    # Try streaming for token-level output
    if job_manager and job_id and hasattr(model, "astream"):
        chunks = []
        async for chunk in model.astream(messages):
            token = chunk.content if hasattr(chunk, "content") else str(chunk)
            if token:
                chunks.append(token)
                job_manager.push_event(job_id, {"type": "token", "content": token})
        final_output = "".join(chunks)
        # Token usage not available from streaming; carry forward prior accumulation
        token_usage = state.get("_token_usage", {"input": 0, "output": 0})
    else:
        response = await model.ainvoke(messages)
        final_output = response.content
        # Extract and accumulate token usage
        meta = getattr(response, "response_metadata", {}) or {}
        usage = meta.get("usage", {})
        prior = state.get("_token_usage", {"input": 0, "output": 0})
        token_usage = {
            "input": prior["input"] + (usage.get("input_tokens", 0) or usage.get("prompt_tokens", 0)),
            "output": prior["output"] + (usage.get("output_tokens", 0) or usage.get("completion_tokens", 0)),
        }

    return {"final_output": final_output, "_token_usage": token_usage}
```

**Step 3: Pass token usage through _run_graph completed event**

In `src/deep_research/api/app.py`, in `_run_graph`, replace the completed event push block (lines 90-93):

```python
            # Extract token usage if accumulated through graph
            token_usage = final_state.get("_token_usage")

            job_manager.update_state(job_id, "completed", result=final_output)
            completed_event: dict[str, Any] = {"type": "completed", "result": final_output}
            if token_usage and (token_usage.get("input", 0) > 0 or token_usage.get("output", 0) > 0):
                completed_event["token_usage"] = {**token_usage, "scope": "job"}
                status = job_manager.get_status(job_id)
                status.token_usage = token_usage
            job_manager.push_event(job_id, completed_event)
            trace_ctx["output"] = final_output
            trace_ctx["status"] = "completed"
```

**Step 4: Verify tests pass**

```bash
cd /tmp/db_deep_research_agent && PYTHONPATH=src pytest tests/ -v --timeout=30 2>&1 | tail -20
```

**Step 5: Commit**

```bash
cd /tmp/db_deep_research_agent
git add src/deep_research/nodes/synthesizer.py src/deep_research/state.py src/deep_research/api/app.py
git commit -m "feat: accumulate and emit token usage from research pipeline"
```

---

## Task 9: Verify End-to-End + Push

**Step 1: Full backend test suite**

```bash
cd /tmp/db_deep_research_agent && PYTHONPATH=src pytest tests/ -v --timeout=30
```

All tests must pass.

**Step 2: Full frontend build**

```bash
cd /tmp/db_deep_research_agent/ui && npm run build 2>&1 | tail -10
```

Build must succeed with no errors.

**Step 3: Push all commits**

```bash
cd /tmp/db_deep_research_agent && git push origin feat/implementation
```

---

## Task 10: Build, Upload, and Deploy

**Step 1: Full production build**

```bash
cd /tmp/db_deep_research_agent/ui && npm run build
```

**Step 2: Upload to Databricks workspace**

Upload all source files (ui/dist, ui/src, src/) to `/Workspace/Users/brian.law@databricks.com/deep-research-agent`:

```python
from databricks.sdk import WorkspaceClient
from databricks.sdk.service.workspace import ImportFormat
import os, base64

w = WorkspaceClient()
ws_base = "/Workspace/Users/brian.law@databricks.com/deep-research-agent"
local_base = "/tmp/db_deep_research_agent"

uploaded = 0
all_files = []

for subdir in ["ui/dist", "ui/src", "src"]:
    for root, dirs, files in os.walk(f"{local_base}/{subdir}"):
        for f in files:
            lp = os.path.join(root, f)
            all_files.append((lp, os.path.relpath(lp, local_base)))

for lp, rel in all_files:
    ws_path = f"{ws_base}/{rel}"
    with open(lp, "rb") as fh:
        content = fh.read()
    w.workspace.import_(path=ws_path, content=base64.b64encode(content).decode(), format=ImportFormat.AUTO, overwrite=True)
    uploaded += 1

print(f"Uploaded {uploaded} files")
```

**Step 3: Deploy**

```python
from databricks.sdk import WorkspaceClient
from databricks.sdk.service.apps import AppDeployment

w = WorkspaceClient()
deploy = w.apps.deploy("deep-research-agent", AppDeployment(source_code_path="/Workspace/Users/brian.law@databricks.com/deep-research-agent"))
print(f"Deploy: {deploy.response.deployment_id}")
```

**Step 4: Verify**

1. Open https://deep-research-agent-984752964297111.11.azure.databricksapps.com
2. Test citations: Ask "What was CBA's closing price?" in Deep Research mode with Genie selected — should see styled `📎` citation badges
3. Test clarification: Ask something ambiguous like "Compare revenue" in Deep Research mode — should see clarification prompt (if clarifier detects ambiguity). Select an option or type custom answer.
4. Test clarification timeout: Ask ambiguous question, wait 30s — should see "No response received — proceeding with: ..." message
5. Test token count: Check that completed messages show a token count label (quick reply shows "X tokens", research shows "X tokens" with "research run" tooltip)
6. Test session continuity: Follow-up question references prior answer
