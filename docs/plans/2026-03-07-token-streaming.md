# Token-Level Answer Streaming Implementation Plan

> **For Claude:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.

**Goal:** Stream the synthesizer's LLM output token-by-token to the frontend, so users see the answer appear incrementally instead of as one big block.

**Architecture:** The synthesizer node currently calls `model.ainvoke()` which waits for the full response. We change it to `model.astream()` which yields token chunks. Each chunk is pushed as a `token` SSE event via the job's event queue. The frontend accumulates tokens into a streaming message bubble. The verifier still runs after synthesis completes (it needs the full text), and a final `completed` event carries the full output for the status endpoint.

**Tech Stack:** ChatDatabricks (`model.astream()`), asyncio.Queue, SSE (EventSource), React state

---

## Design Decisions

1. **Token events bypass LangGraph streaming.** LangGraph's `astream(stream_mode="updates")` only fires when a node *finishes*. To stream mid-node, the synthesizer pushes token events directly to the job's event queue. This requires passing `job_manager` and `job_id` into the synthesizer node.

2. **New SSE event type: `token`.** Format: `{"type": "token", "content": "chunk text"}`. The frontend accumulates these into a growing message. When `completed` arrives, it replaces the streaming message with the final full text.

3. **Verifier runs after synthesis completes.** No change to graph edges. The synthesizer still sets `final_output` in state — just builds it from streamed chunks instead of `ainvoke`.

4. **Graceful fallback.** If `model.astream()` is unavailable (mock or test), fall back to `model.ainvoke()` and push the full result as a single token event. No streaming, same behavior as today.

---

### Task 1: Pass job context to synthesizer node

**Files:**
- Modify: `src/deep_research/api/app.py:47-57` (pass job_manager/job_id via state)
- Modify: `src/deep_research/state.py:20-62` (add `_job_manager` and `_job_id` to initial state creation — these are runtime-only, not part of ResearchState TypedDict)

**Step 1: Write the failing test**

In `tests/test_api.py`, add a test that verifies `_run_graph` injects `_job_manager` and `_job_id` into the state passed to `graph.astream`:

```python
async def test_run_graph_injects_job_context():
    """_run_graph adds _job_manager and _job_id to state before calling graph."""
    jm = JobManager()
    job_id = jm.create_job(query="test", tools=[])
    captured_state = {}

    class _CaptureGraph:
        async def astream(self, state, stream_mode="updates"):
            captured_state.update(state)
            yield {"synthesizer": {"final_output": "done"}}
            yield {"verifier": {"verification_result": None}}

    await _run_graph(_CaptureGraph(), jm, job_id, {"user_query": "test"})
    assert captured_state["_job_manager"] is jm
    assert captured_state["_job_id"] == job_id
```

**Step 2: Run test to verify it fails**

Run: `PYTHONPATH=src pytest tests/test_api.py::test_run_graph_injects_job_context -v`
Expected: FAIL — `_job_manager` not in captured_state

**Step 3: Write minimal implementation**

In `src/deep_research/api/app.py`, in `_run_graph`, inject job context into the state dict before passing to graph:

```python
# In _run_graph, before calling graph.astream:
initial_state["_job_manager"] = job_manager
initial_state["_job_id"] = job_id
```

**Step 4: Run test to verify it passes**

Run: `PYTHONPATH=src pytest tests/test_api.py::test_run_graph_injects_job_context -v`
Expected: PASS

**Step 5: Commit**

```bash
git add src/deep_research/api/app.py tests/test_api.py
git commit -m "feat: inject job context into graph state for token streaming"
```

---

### Task 2: Stream tokens from synthesizer node

**Files:**
- Modify: `src/deep_research/nodes/synthesizer.py` (use `model.astream()`, push token events)
- Test: `tests/test_nodes/test_all_nodes.py` (add streaming test)

**Step 1: Write the failing test**

In `tests/test_nodes/test_all_nodes.py`, add a test that verifies the synthesizer pushes token events when a job_manager is available in state:

```python
async def test_synthesizer_streams_tokens():
    """Synthesizer pushes token events to job_manager when available."""
    from deep_research.api.jobs import JobManager

    jm = JobManager()
    job_id = jm.create_job(query="test", tools=[])

    # Mock model with astream that yields chunks
    class _StreamingModel:
        async def astream(self, messages):
            for word in ["Hello", " world", "!"]:
                chunk = MagicMock()
                chunk.content = word
                yield chunk

        async def ainvoke(self, messages):
            resp = MagicMock()
            resp.content = "Hello world!"
            return resp

    model = _StreamingModel()
    state = create_initial_state(user_query="test", selected_tools=[])
    state["_job_manager"] = jm
    state["_job_id"] = job_id

    result = await synthesizer_node(state, model=model)
    assert result["final_output"] == "Hello world!"

    # Check token events were pushed
    queue = jm.get_event_queue(job_id)
    events = []
    while not queue.empty():
        events.append(queue.get_nowait())
    token_events = [e for e in events if e["type"] == "token"]
    assert len(token_events) == 3
    assert token_events[0]["content"] == "Hello"
    assert token_events[1]["content"] == " world"
    assert token_events[2]["content"] == "!"
```

**Step 2: Run test to verify it fails**

Run: `PYTHONPATH=src pytest tests/test_nodes/test_all_nodes.py::test_synthesizer_streams_tokens -v`
Expected: FAIL — synthesizer doesn't call astream or push events

**Step 3: Write implementation**

Replace `synthesizer.py` with streaming-aware version:

```python
"""Synthesizer node — produces final answer or report from findings."""

from __future__ import annotations

import logging
from typing import Any

from deep_research.prompts import SYNTHESIZER_CHAT_SYSTEM, SYNTHESIZER_REPORT_SYSTEM
from deep_research.state import ResearchState

logger = logging.getLogger(__name__)


async def synthesizer_node(state: ResearchState, *, model: Any) -> dict:
    """Generate final output from compressed findings, streaming tokens when possible."""
    findings = state.get("compressed_findings")
    evidence = state.get("evidence", [])
    output_mode = state.get("output_mode", "chat")
    query = state.get("clarified_query") or state["user_query"]

    # Job context for token streaming (injected by _run_graph)
    job_manager = state.get("_job_manager")
    job_id = state.get("_job_id")

    system_prompt = (
        SYNTHESIZER_REPORT_SYSTEM if output_mode == "report"
        else SYNTHESIZER_CHAT_SYSTEM
    )

    # Build context for synthesis
    context_parts = [f"User query: {query}"]

    if findings:
        context_parts.append(f"Key findings: {', '.join(findings.key_findings)}")
        if findings.open_questions:
            context_parts.append(f"Open questions: {', '.join(findings.open_questions)}")
        if findings.uncertainties:
            context_parts.append(f"Uncertainties: {', '.join(findings.uncertainties)}")
        if findings.contradictions:
            context_parts.append(f"Contradictions: {', '.join(findings.contradictions)}")

    if evidence:
        context_parts.append("Evidence:")
        for e in evidence:
            context_parts.append(f"- [{e.source_id}] {e.snippet}")

    messages = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": "\n".join(context_parts)},
    ]

    # Try streaming for token-level output
    if job_manager and job_id and hasattr(model, "astream"):
        chunks = []
        async for chunk in model.astream(messages):
            token = chunk.content if hasattr(chunk, "content") else str(chunk)
            if token:
                chunks.append(token)
                job_manager.push_event(job_id, {"type": "token", "content": token})
        final_output = "".join(chunks)
    else:
        response = await model.ainvoke(messages)
        final_output = response.content

    return {"final_output": final_output}
```

**Step 4: Run test to verify it passes**

Run: `PYTHONPATH=src pytest tests/test_nodes/test_all_nodes.py::test_synthesizer_streams_tokens -v`
Expected: PASS

**Step 5: Run existing synthesizer tests to ensure no regression**

Run: `PYTHONPATH=src pytest tests/test_nodes/test_all_nodes.py::test_synthesizer_chat_mode tests/test_nodes/test_all_nodes.py::test_synthesizer_report_mode -v`
Expected: PASS (existing tests use `ainvoke`-only mocks without `_job_manager`, so they hit the fallback path)

**Step 6: Commit**

```bash
git add src/deep_research/nodes/synthesizer.py tests/test_nodes/test_all_nodes.py
git commit -m "feat: stream tokens from synthesizer via job event queue"
```

---

### Task 3: Frontend — accumulate token events into streaming message

**Files:**
- Modify: `ui/src/hooks/useResearch.ts` (handle `token` events, create streaming message)
- Modify: `ui/src/types.ts` (add `streaming` flag to Message)
- Modify: `ui/src/components/MessageBubble.tsx` (show cursor for streaming messages)

**Step 1: Add `streaming` flag to Message type**

In `ui/src/types.ts`, add optional `streaming` boolean:

```typescript
export interface Message {
  id: string;
  role: "user" | "assistant";
  content: string;
  timestamp: string;
  jobId?: string;
  rating?: "thumbs_up" | "thumbs_down";
  streaming?: boolean;  // true while tokens are still arriving
}
```

**Step 2: Handle `token` events in useResearch hook**

In `ui/src/hooks/useResearch.ts`, add a `token` case to the event handler. When the first token arrives, create a new streaming message. On subsequent tokens, append to it. On `completed`, finalize it:

```typescript
// Inside startStream's onEvent callback, add before the "completed" case:

if (event.type === "token" && event.content) {
  setMessages((prev) => {
    const lastMsg = prev[prev.length - 1];
    if (lastMsg?.streaming && lastMsg.jobId === jobId) {
      // Append to existing streaming message
      return [
        ...prev.slice(0, -1),
        { ...lastMsg, content: lastMsg.content + event.content },
      ];
    } else {
      // Create new streaming message
      return [
        ...prev,
        {
          id: crypto.randomUUID(),
          role: "assistant" as const,
          content: event.content,
          timestamp: new Date().toISOString(),
          jobId,
          streaming: true,
        },
      ];
    }
  });
}
```

Update the `completed` handler: instead of always creating a new message, finalize the streaming message if one exists:

```typescript
} else if (event.type === "completed") {
  stopStream();
  setIsLoading(false);
  setCurrentJob(null);
  setMessages((prev) => {
    const lastMsg = prev[prev.length - 1];
    if (lastMsg?.streaming && lastMsg.jobId === jobId) {
      // Finalize streaming message with full result
      return [
        ...prev.slice(0, -1),
        {
          ...lastMsg,
          content: event.result || lastMsg.content,
          streaming: false,
        },
      ];
    } else if (event.result) {
      // No streaming happened — create message from full result
      return [
        ...prev,
        {
          id: crypto.randomUUID(),
          role: "assistant" as const,
          content: event.result,
          timestamp: new Date().toISOString(),
          jobId,
        },
      ];
    }
    return prev;
  });
}
```

**Step 3: Add streaming cursor to MessageBubble**

In `ui/src/components/MessageBubble.tsx`, show a blinking cursor when `message.streaming` is true:

```tsx
{!isUser && message.streaming && (
  <span className="streaming-cursor" aria-hidden="true">▊</span>
)}
```

Add CSS for the cursor (in the existing CSS file):

```css
.streaming-cursor {
  display: inline;
  animation: blink 1s step-end infinite;
  color: var(--accent, #3b82f6);
}
@keyframes blink {
  50% { opacity: 0; }
}
```

**Step 4: Build UI**

Run: `cd ui && npm run build`
Expected: Build succeeds

**Step 5: Commit**

```bash
git add ui/src/hooks/useResearch.ts ui/src/types.ts ui/src/components/MessageBubble.tsx
git commit -m "feat: accumulate token events into streaming message bubble"
```

---

### Task 4: Run full test suite, deploy, verify

**Files:**
- No new files

**Step 1: Run full Python test suite**

Run: `PYTHONPATH=src pytest tests/ --ignore=tests/test_graph.py --ignore=tests/test_integration.py -v`
Expected: All tests pass (93+)

**Step 2: Upload changed files to Databricks workspace**

Files to upload:
- `src/deep_research/api/app.py`
- `src/deep_research/nodes/synthesizer.py`
- `ui/dist/*` (rebuilt)

Use the REST API upload pattern from previous deployments.

**Step 3: Deploy**

```bash
curl -s -X POST "${DATABRICKS_HOST}/api/2.0/apps/deep-research-agent/deployments" \
  -H "Authorization: Bearer ${DATABRICKS_TOKEN}" \
  -H "Content-Type: application/json" \
  -d '{"source_code_path": "/Workspace/Users/brian.law@databricks.com/deep-research-agent"}'
```

**Step 4: Monitor deployment**

Poll deployment status until SUCCEEDED.

**Step 5: Verify in browser**

Open the app URL, submit a query, confirm:
- Progress bar still updates per-node
- Answer text streams in token-by-token during synthesizer phase
- Blinking cursor appears while streaming
- Final message is complete after `completed` event

**Step 6: Commit and push**

```bash
git push origin feat/implementation
```

---

## Risk Notes

- **ChatDatabricks `astream()` chunk format:** Each chunk should have a `.content` attribute with the token text. If the Databricks endpoint returns chunks differently (e.g. AIMessageChunk with empty content for metadata), the synthesizer handles this with the `if token:` guard.
- **Queue pressure:** Token events are small and fast. The asyncio.Queue is unbounded so no backpressure issues. If a query generates a very long response, many events accumulate — but SSE drains them fast enough.
- **ReactMarkdown re-rendering:** Updating content on every token triggers ReactMarkdown re-parse. For very long outputs this could lag. If observed, throttle token updates (batch every ~50ms). Not implementing upfront — optimize only if needed.
