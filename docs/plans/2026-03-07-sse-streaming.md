# SSE Streaming for Research Progress

> **For Claude:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.

**Goal:** Replace the frontend's polling loop with Server-Sent Events (SSE) so the UI receives real-time node progress updates and the final result via a persistent HTTP stream.

**Architecture:** Add a `GET /api/research/{job_id}/stream` SSE endpoint that pushes JSON events as the graph executes. The frontend replaces its `setTimeout`-based polling with a native `EventSource` connection. The existing polling endpoint stays as a fallback. The backend uses `asyncio.Queue` per job so `_run_graph` pushes events and the SSE generator consumes them.

**Tech Stack:** FastAPI `StreamingResponse` (starlette), `asyncio.Queue`, native browser `EventSource` API (no new npm deps).

---

## Current State

- Backend: `POST /api/research` creates job + launches `_run_graph` background task. `GET /api/research/{job_id}` returns current state snapshot.
- Frontend: `useResearch` hook submits via POST, then polls GET every 1.5s via `setTimeout`. `ProgressBar` reads `current_node` from poll response.
- Problem: Polling has 1.5s latency, wastes requests when idle, and can't deliver partial text.

## Design Decisions

1. **SSE over WebSocket** — SSE is simpler (unidirectional server→client), works with FastAPI out of the box, auto-reconnects, and the use case is purely server push.
2. **asyncio.Queue per job** — The `_run_graph` task pushes events to a queue. The SSE generator reads from it. This decouples graph execution from the HTTP connection.
3. **Keep polling endpoint** — `GET /api/research/{job_id}` stays unchanged as fallback for clients that don't support SSE.
4. **Event types:** `node_started`, `completed`, `failed`, `cancelled`. Future: `token` for partial LLM output.

---

### Task 1: Add event queue to JobManager

**Files:**
- Modify: `src/deep_research/api/jobs.py`
- Test: `tests/test_api.py`

**Step 1: Write the failing test**

Add to `tests/test_api.py`:

```python
async def test_job_event_queue():
    """JobManager creates an event queue per job and events can be pushed/consumed."""
    jm = JobManager()
    job_id = jm.create_job(query="test", tools=[])
    queue = jm.get_event_queue(job_id)
    assert queue is not None

    jm.push_event(job_id, {"type": "node_started", "node": "clarifier"})
    event = queue.get_nowait()
    assert event["type"] == "node_started"
    assert event["node"] == "clarifier"


async def test_job_event_queue_unknown_job():
    """get_event_queue raises KeyError for unknown job."""
    jm = JobManager()
    with pytest.raises(KeyError):
        jm.get_event_queue("nonexistent")
```

**Step 2: Run test to verify it fails**

Run: `PYTHONPATH=src python3 -m pytest tests/test_api.py::test_job_event_queue tests/test_api.py::test_job_event_queue_unknown_job -v`
Expected: FAIL — `JobManager` has no `get_event_queue` or `push_event` method.

**Step 3: Write minimal implementation**

In `src/deep_research/api/jobs.py`, add `asyncio` import and modify `JobManager`:

```python
import asyncio

# In JobManager.__init__, add:
self._queues: dict[str, asyncio.Queue] = {}

# In create_job, after creating the JobStatus, add:
self._queues[job_id] = asyncio.Queue()

# Add two new methods:
def get_event_queue(self, job_id: str) -> asyncio.Queue:
    if job_id not in self._queues:
        raise KeyError(f"Unknown job: {job_id}")
    return self._queues[job_id]

def push_event(self, job_id: str, event: dict) -> None:
    if job_id in self._queues:
        self._queues[job_id].put_nowait(event)
```

**Step 4: Run test to verify it passes**

Run: `PYTHONPATH=src python3 -m pytest tests/test_api.py::test_job_event_queue tests/test_api.py::test_job_event_queue_unknown_job -v`
Expected: PASS

**Step 5: Commit**

```bash
git add src/deep_research/api/jobs.py tests/test_api.py
git commit -m "feat: add event queue to JobManager for SSE streaming"
```

---

### Task 2: Push events from `_run_graph`

**Files:**
- Modify: `src/deep_research/api/app.py`
- Test: `tests/test_api.py`

**Step 1: Write the failing test**

Add to `tests/test_api.py`:

```python
async def test_run_graph_pushes_events():
    """_run_graph pushes node_started events and a completed event to the queue."""
    jm = JobManager()
    job_id = jm.create_job(query="test", tools=[])

    class _MultiNodeGraph:
        async def astream(self, state, stream_mode="updates"):
            yield {"clarifier": {"clarified_query": "test"}}
            yield {"planner": {"research_plan": []}}
            yield {"verifier": {"final_output": "done"}}

    await _run_graph(_MultiNodeGraph(), jm, job_id, {})

    events = []
    queue = jm.get_event_queue(job_id)
    while not queue.empty():
        events.append(queue.get_nowait())

    node_events = [e for e in events if e["type"] == "node_started"]
    assert len(node_events) == 3
    assert node_events[0]["node"] == "clarifier"
    assert node_events[1]["node"] == "planner"
    assert node_events[2]["node"] == "verifier"

    terminal = [e for e in events if e["type"] in ("completed", "failed")]
    assert len(terminal) == 1
    assert terminal[0]["type"] == "completed"
    assert terminal[0]["result"] == "done"
```

**Step 2: Run test to verify it fails**

Run: `PYTHONPATH=src python3 -m pytest tests/test_api.py::test_run_graph_pushes_events -v`
Expected: FAIL — `_run_graph` doesn't accept or use a job_manager with events.

**Step 3: Write minimal implementation**

In `src/deep_research/api/app.py`, update `_run_graph` to push events. Inside the `async for event in graph.astream(...)` loop, after `job_manager.update_state(...)`, add:

```python
job_manager.push_event(job_id, {"type": "node_started", "node": node_name})
```

After `job_manager.update_state(job_id, "completed", ...)`, add:

```python
job_manager.push_event(job_id, {"type": "completed", "result": final_output})
```

In the `except` block after `job_manager.update_state(job_id, "failed", ...)`, add:

```python
job_manager.push_event(job_id, {"type": "failed", "error": f"Research execution failed: {exc}"})
```

Also in the `else` branch (non-streaming fallback), add the same completed push after update_state.

**Step 4: Run test to verify it passes**

Run: `PYTHONPATH=src python3 -m pytest tests/test_api.py::test_run_graph_pushes_events -v`
Expected: PASS

**Step 5: Run full test suite**

Run: `PYTHONPATH=src python3 -m pytest tests/ -x -q`
Expected: All 93+ tests pass (existing tests unaffected since they don't read the queue).

**Step 6: Commit**

```bash
git add src/deep_research/api/app.py tests/test_api.py
git commit -m "feat: push SSE events from _run_graph during execution"
```

---

### Task 3: Add SSE streaming endpoint

**Files:**
- Modify: `src/deep_research/api/app.py`
- Test: `tests/test_api.py`

**Step 1: Write the failing test**

Add to `tests/test_api.py`:

```python
async def test_sse_stream_endpoint(graph_client):
    """SSE endpoint streams node events then completes."""
    # Submit a job
    response = await graph_client.post("/api/research", json={
        "query": "Test SSE",
        "tools": [],
        "output_mode": "chat",
    })
    job_id = response.json()["job_id"]

    # Let background task complete
    await asyncio.sleep(0.2)

    # Read SSE stream
    response = await graph_client.get(
        f"/api/research/{job_id}/stream",
        headers={"Accept": "text/event-stream"},
    )
    assert response.status_code == 200
    assert "text/event-stream" in response.headers["content-type"]

    # Parse SSE events from response body
    lines = response.text.strip().split("\n")
    data_lines = [l.removeprefix("data: ") for l in lines if l.startswith("data: ")]
    assert len(data_lines) >= 1  # at least the completed event

    import json as _json
    last_event = _json.loads(data_lines[-1])
    assert last_event["type"] in ("completed", "failed")


async def test_sse_stream_unknown_job(client):
    """SSE stream for unknown job returns 404."""
    response = await client.get("/api/research/nonexistent/stream")
    assert response.status_code == 404
```

**Step 2: Run test to verify it fails**

Run: `PYTHONPATH=src python3 -m pytest tests/test_api.py::test_sse_stream_endpoint tests/test_api.py::test_sse_stream_unknown_job -v`
Expected: FAIL — 404 because the route doesn't exist yet.

**Step 3: Write minimal implementation**

Add to `src/deep_research/api/app.py`, inside `create_app()`, after the existing `get_research_status` endpoint:

```python
import json as _json
from starlette.responses import StreamingResponse

@app.get("/api/research/{job_id}/stream")
async def stream_research(job_id: str):
    try:
        job_manager.get_status(job_id)
    except KeyError:
        raise HTTPException(status_code=404, detail="Job not found")

    async def event_generator():
        queue = job_manager.get_event_queue(job_id)
        while True:
            try:
                event = await asyncio.wait_for(queue.get(), timeout=30.0)
                yield f"data: {_json.dumps(event)}\n\n"
                if event.get("type") in ("completed", "failed", "cancelled"):
                    break
            except asyncio.TimeoutError:
                # Send keepalive
                yield f": keepalive\n\n"
                # Check if job is in terminal state (e.g. cancelled externally)
                status = job_manager.get_status(job_id)
                if status.state in ("completed", "failed", "cancelled"):
                    yield f"data: {_json.dumps({'type': status.state, 'result': status.result, 'error': status.error})}\n\n"
                    break

    return StreamingResponse(event_generator(), media_type="text/event-stream")
```

Move the `import json` to the top of the file (rename the module-level one if needed, or use `json` directly since it's already effectively available).

**Step 4: Run test to verify it passes**

Run: `PYTHONPATH=src python3 -m pytest tests/test_api.py::test_sse_stream_endpoint tests/test_api.py::test_sse_stream_unknown_job -v`
Expected: PASS

**Step 5: Run full test suite**

Run: `PYTHONPATH=src python3 -m pytest tests/ -x -q`
Expected: All tests pass.

**Step 6: Commit**

```bash
git add src/deep_research/api/app.py tests/test_api.py
git commit -m "feat: add GET /api/research/{job_id}/stream SSE endpoint"
```

---

### Task 4: Add `streamJob` to frontend API client

**Files:**
- Modify: `ui/src/api.ts`

**Step 1: Add the streamJob function**

```typescript
export function streamJob(
  jobId: string,
  onEvent: (event: { type: string; node?: string; result?: string; error?: string }) => void,
  onError: (err: Error) => void
): () => void {
  const es = new EventSource(`/api/research/${jobId}/stream`);

  es.onmessage = (msg) => {
    try {
      const event = JSON.parse(msg.data);
      onEvent(event);
      if (["completed", "failed", "cancelled"].includes(event.type)) {
        es.close();
      }
    } catch {
      // Ignore unparseable messages (keepalives)
    }
  };

  es.onerror = () => {
    es.close();
    onError(new Error("SSE connection lost"));
  };

  // Return cleanup function
  return () => es.close();
}
```

**Step 2: Commit**

```bash
git add ui/src/api.ts
git commit -m "feat: add streamJob SSE client to frontend API layer"
```

---

### Task 5: Replace polling with SSE in useResearch hook

**Files:**
- Modify: `ui/src/hooks/useResearch.ts`

**Step 1: Replace the polling implementation**

Key changes to `useResearch.ts`:

1. Remove `POLL_INTERVAL_MS` constant
2. Replace `timeoutRef` with `cleanupRef` (stores the SSE close function)
3. Replace `stopPolling` → `stopStream` (calls the cleanup function)
4. Replace the `poll` callback with an `startStream` callback that uses `streamJob`
5. In `send`, after `submitResearch`, call `startStream(job_id)` instead of `setTimeout(() => poll(...))`

```typescript
import { useCallback, useEffect, useRef, useState } from "react";
import { submitResearch, streamJob, cancelJob, submitFeedback } from "../api";
import type { Message, JobStatus, OutputMode } from "../types";

export function useResearch() {
  const [messages, setMessages] = useState<Message[]>([]);
  const [currentJob, setCurrentJob] = useState<JobStatus | null>(null);
  const [isLoading, setIsLoading] = useState(false);
  const [error, setError] = useState<string | null>(null);

  const activeJobIdRef = useRef<string | null>(null);
  const cleanupRef = useRef<(() => void) | null>(null);
  const mountedRef = useRef(true);

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
          } else if (event.type === "completed") {
            stopStream();
            setIsLoading(false);
            setCurrentJob(null);
            if (event.result) {
              setMessages((prev) => [
                ...prev,
                {
                  id: crypto.randomUUID(),
                  role: "assistant",
                  content: event.result!,
                  timestamp: new Date().toISOString(),
                  jobId,
                },
              ]);
            }
          } else if (event.type === "failed") {
            stopStream();
            setIsLoading(false);
            setCurrentJob(null);
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
          // SSE error — fall back to showing an error
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
    async (query: string, tools: string[], outputMode: OutputMode) => {
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
        const { job_id } = await submitResearch(query, tools, outputMode);
        if (!mountedRef.current) return;

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

    try {
      await cancelJob(jobId);
    } catch {
      // Best-effort cancellation
    }
  }, [stopStream]);

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

  return { messages, currentJob, isLoading, error, send, cancel, rate };
}
```

**Step 2: Commit**

```bash
git add ui/src/hooks/useResearch.ts ui/src/api.ts
git commit -m "feat: replace polling with SSE streaming in useResearch hook"
```

---

### Task 6: Build UI, run full tests, deploy

**Files:**
- No new files

**Step 1: Build the React UI**

Run: `cd ui && npm run build`
Expected: Build succeeds, output in `ui/dist/`

**Step 2: Run full Python test suite**

Run: `PYTHONPATH=src python3 -m pytest tests/ -x -v`
Expected: All tests pass.

**Step 3: Upload changed files to Databricks workspace**

Upload these files via the workspace import API:
- `src/deep_research/api/app.py`
- `src/deep_research/api/jobs.py`
- `ui/dist/*` (the built frontend)

**Step 4: Deploy**

```bash
curl -s -X POST "$DATABRICKS_HOST/api/2.0/apps/deep-research-agent/deployments" \
  -H "Authorization: Bearer $DATABRICKS_TOKEN" \
  -H "Content-Type: application/json" \
  -d '{"source_code_path": "/Workspace/Users/brian.law@databricks.com/deep-research-agent"}'
```

**Step 5: Verify deployment succeeds**

Poll the app status until `RUNNING` + deployment `SUCCEEDED`.

**Step 6: Commit build output and push**

```bash
git add -A
git commit -m "feat: SSE streaming for research progress - build and deploy"
git push origin feat/implementation
```
