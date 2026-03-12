# Citations, Clarification & Token Display — Implementation Plan

> **For Claude:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.

**Goal:** Add enhanced text citations as styled badges, interactive user clarification with pipeline pause/resume, and token usage display on completed messages.

**Architecture:** Citations are prompt + UI-only. Clarification adds asyncio.Event-based pause/resume to _run_graph + a new /clarify endpoint + a new UI component. Token display adds metadata extraction from LLM responses accumulated through the graph state.

**Tech Stack:** Python 3.11, FastAPI, LangGraph, ChatDatabricks, React 19, TypeScript, Tailwind CSS 4, Vite 7.

---

## Task 1: Enhanced Citations — Update Synthesizer Prompts

**Files:**
- Modify: `src/deep_research/prompts.py`

**Step 1: Update SYNTHESIZER_CHAT_SYSTEM prompt**

In `src/deep_research/prompts.py`, replace the `SYNTHESIZER_CHAT_SYSTEM` string (lines 110-115) with:

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

Replace the `SYNTHESIZER_REPORT_SYSTEM` string (lines 117-127) with:

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
- Modify: `src/deep_research/nodes/synthesizer.py`

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

**Step 1: Add citation rendering to MessageBubble**

Replace the entire content of `ui/src/components/MessageBubble.tsx` with:

```tsx
import type { ComponentPropsWithoutRef } from "react";
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

function renderWithCitations(text: string): string {
  return text.replace(
    CITATION_RE,
    '<span class="inline-flex items-center px-1.5 py-0.5 mx-0.5 rounded-md bg-warm-sage/15 text-warm-sage text-xs font-medium whitespace-nowrap align-baseline">📎 $1</span>'
  );
}

function MarkdownWithCitations({ children }: { children: string }) {
  const processed = renderWithCitations(children);
  return (
    <ReactMarkdown
      remarkPlugins={[remarkGfm]}
      rehypePlugins={[rehypeSanitize]}
      components={{
        p: ({ children: pChildren, ...props }: ComponentPropsWithoutRef<"p">) => (
          <p {...props} dangerouslySetInnerHTML={undefined}>
            {pChildren}
          </p>
        ),
      }}
    >
      {processed}
    </ReactMarkdown>
  );
}

export function MessageBubble({ message, onRate }: Props) {
  const isUser = message.role === "user";

  // Pre-process content: replace citation markers with HTML spans
  // We do this before passing to ReactMarkdown so citations render as badges
  const displayContent = !isUser
    ? message.content.replace(
        CITATION_RE,
        "**`📎 $1`**"
      )
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
              <span className="ml-auto text-[11px] text-warm-text-secondary tabular-nums">
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
- Citations like `[Source: Genie — table]` are converted to `**\`📎 Genie — table\`**` before markdown rendering
- Tailwind prose styles `prose-code:bg-warm-sage/10 prose-code:text-warm-sage` make inline code blocks look like citation badges
- Token usage label added (will work once Task 9 provides the data)
- Removed unused `renderWithCitations` and `MarkdownWithCitations` (we use the simpler markdown approach)

**Step 2: Verify TypeScript compiles**

```bash
cd /tmp/db_deep_research_agent/ui && npx tsc -b 2>&1
```

Note: This will show an error about `tokenUsage` not existing on Message yet. That's expected — Task 9 adds it. For now, add the type to `ui/src/types.ts` early.

**Step 3: Add tokenUsage to Message type (needed for compile)**

In `ui/src/types.ts`, add `tokenUsage` to the Message interface:

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
  tokenUsage?: { input: number; output: number };
}
```

**Step 4: Verify build**

```bash
cd /tmp/db_deep_research_agent/ui && npx tsc -b 2>&1
```

**Step 5: Commit**

```bash
cd /tmp/db_deep_research_agent
git add ui/src/components/MessageBubble.tsx ui/src/types.ts
git commit -m "feat(ui): style citation markers as badges, add tokenUsage type"
```

---

## Task 4: Clarification — Update Clarifier Prompt and Node

**Files:**
- Modify: `src/deep_research/prompts.py`
- Modify: `src/deep_research/nodes/clarifier.py`

**Step 1: Update CLARIFIER_SYSTEM prompt**

Replace the `CLARIFIER_SYSTEM` string in `src/deep_research/prompts.py` with:

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

**Step 2: Update clarifier_node to return needs_clarification**

Replace the content of `src/deep_research/nodes/clarifier.py` with:

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

    Returns either:
    - {"clarified_query": "..."} for clear queries
    - {"clarified_query": best_guess, "needs_clarification": True,
       "clarification_question": "...", "clarification_options": [...]}
    """
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

**Step 3: Add clarification fields to ResearchState**

In `src/deep_research/state.py`, add these fields to `ResearchState` (after `clarified_query`):

```python
    # Clarification
    clarified_query: str | None
    needs_clarification: bool
    clarification_question: str
    clarification_options: list[str]
```

And update `create_initial_state` to include defaults:

```python
        needs_clarification=False,
        clarification_question="",
        clarification_options=[],
```

**Step 4: Verify tests pass**

```bash
cd /tmp/db_deep_research_agent && PYTHONPATH=src pytest tests/test_nodes/test_all_nodes.py -v --timeout=30
```

**Step 5: Commit**

```bash
cd /tmp/db_deep_research_agent
git add src/deep_research/prompts.py src/deep_research/nodes/clarifier.py src/deep_research/state.py
git commit -m "feat: clarifier node supports needs_clarification response"
```

---

## Task 5: Clarification — Backend Pipeline Pause/Resume

**Files:**
- Modify: `src/deep_research/api/jobs.py`
- Modify: `src/deep_research/api/app.py`

**Step 1: Add clarification support to JobManager**

In `src/deep_research/api/jobs.py`, add to the `JobStatus` dataclass:

```python
@dataclass
class JobStatus:
    job_id: str
    state: Literal["pending", "running", "completed", "cancelled", "failed"]
    query: str
    tools: list[str]
    output_mode: str = "chat"
    created_by: str = ""
    created_at: datetime | None = None
    current_node: str | None = None
    result: str | None = None
    error: str | None = None
    token_usage: dict[str, int] | None = None
```

Add these methods to `JobManager`:

```python
    def __init__(self) -> None:
        self._jobs: dict[str, JobStatus] = {}
        self._queues: dict[str, asyncio.Queue] = {}
        self._clarification_events: dict[str, asyncio.Event] = {}
        self._clarification_answers: dict[str, str] = {}

    def request_clarification(self, job_id: str) -> asyncio.Event:
        """Create an event that will be set when the user provides clarification."""
        event = asyncio.Event()
        self._clarification_events[job_id] = event
        return event

    def submit_clarification(self, job_id: str, answer: str) -> bool:
        """Submit a clarification answer for a waiting job. Returns True if job was waiting."""
        event = self._clarification_events.get(job_id)
        if event is None:
            return False
        self._clarification_answers[job_id] = answer
        event.set()
        return True

    def get_clarification_answer(self, job_id: str) -> str | None:
        """Get the clarification answer and clean up."""
        answer = self._clarification_answers.pop(job_id, None)
        self._clarification_events.pop(job_id, None)
        return answer
```

**Step 2: Add /clarify endpoint and modify _run_graph**

In `src/deep_research/api/app.py`, add the clarify endpoint inside `create_app`:

```python
    class ClarifyRequest(BaseModel):
        answer: str = Field(..., min_length=1)

    @app.post("/api/research/{job_id}/clarify")
    async def submit_clarification(job_id: str, req: ClarifyRequest):
        try:
            job_manager.get_status(job_id)
        except KeyError:
            raise HTTPException(status_code=404, detail="Job not found")
        if not job_manager.submit_clarification(job_id, req.answer):
            raise HTTPException(status_code=400, detail="Job is not waiting for clarification")
        return {"status": "ok", "job_id": job_id}
```

**Step 3: Modify _run_graph to handle clarification pause/resume**

In `_run_graph`, after the graph streaming loop completes, we need to intercept the clarifier output. The cleanest approach: instead of modifying graph execution mid-stream, check the accumulated state after the clarifier node yields.

Replace `_run_graph` with:

```python
async def _run_graph(
    graph: Any,
    job_manager: JobManager,
    job_id: str,
    initial_state: dict,
    session_manager: SessionManager | None = None,
    session_id: str | None = None,
) -> None:
    """Execute the research graph in the background, updating job state."""
    from deep_research.tracing import trace_research

    query = initial_state.get("user_query", "")
    tools = initial_state.get("selected_tools", [])
    output_mode = initial_state.get("output_mode", "chat")

    async with trace_research(job_id, query, tools, output_mode, mode="research") as trace_ctx:
        try:
            job_manager.update_state(job_id, "running")

            initial_state["_job_manager"] = job_manager
            initial_state["_job_id"] = job_id

            if hasattr(graph, "astream"):
                accumulated_state: dict = {}
                async for event in graph.astream(
                    initial_state,
                    stream_mode="updates",
                    config={"recursion_limit": 50},
                ):
                    for node_name in event:
                        logger.info(f"Job {job_id}: completed node '{node_name}'")
                        job_manager.update_state(
                            job_id, "running", current_node=node_name
                        )
                        job_manager.push_event(job_id, {"type": "node_started", "node": node_name})
                        node_output = event[node_name]
                        if isinstance(node_output, dict):
                            accumulated_state.update(node_output)

                        # Check for clarification after clarifier node
                        if node_name == "clarifier" and isinstance(node_output, dict):
                            if node_output.get("needs_clarification"):
                                question = node_output.get("clarification_question", "")
                                options = node_output.get("clarification_options", [])
                                best_guess = node_output.get("clarified_query", query)

                                # Push clarification event to UI
                                job_manager.push_event(job_id, {
                                    "type": "clarification_needed",
                                    "question": question,
                                    "options": options,
                                })

                                # Wait for user response (30s timeout)
                                wait_event = job_manager.request_clarification(job_id)
                                try:
                                    await asyncio.wait_for(wait_event.wait(), timeout=30.0)
                                    answer = job_manager.get_clarification_answer(job_id)
                                    if answer:
                                        # User answered — update the clarified query
                                        accumulated_state["clarified_query"] = answer
                                        accumulated_state["needs_clarification"] = False
                                        job_manager.push_event(job_id, {
                                            "type": "clarification_resolved",
                                            "answer": answer,
                                        })
                                except asyncio.TimeoutError:
                                    # Timeout — proceed with best guess
                                    job_manager.get_clarification_answer(job_id)  # cleanup
                                    accumulated_state["clarified_query"] = best_guess
                                    accumulated_state["needs_clarification"] = False
                                    job_manager.push_event(job_id, {
                                        "type": "clarification_timeout",
                                        "best_guess": best_guess,
                                    })

                final_output = accumulated_state.get("final_output", "")
                if not final_output:
                    final_output = "Research completed but produced no output."
                final_state = accumulated_state
            else:
                result = await graph.ainvoke(initial_state)
                final_output = result.get("final_output", "")
                if not final_output:
                    final_output = "Research completed but produced no output."
                final_state = result

            # Extract token usage if available
            token_usage = final_state.get("_token_usage")

            job_manager.update_state(job_id, "completed", result=final_output)
            completed_event: dict[str, Any] = {"type": "completed", "result": final_output}
            if token_usage:
                completed_event["token_usage"] = token_usage
                # Also store on job status
                status = job_manager.get_status(job_id)
                status.token_usage = token_usage
            job_manager.push_event(job_id, completed_event)
            trace_ctx["output"] = final_output
            trace_ctx["status"] = "completed"

            if session_manager and session_id:
                session_manager.add_turn(session_id, "assistant", final_output)
                evidence = final_state.get("evidence", [])
                if evidence:
                    session_manager.add_evidence(session_id, evidence)
                tool_calls = final_state.get("tool_call_log", [])
                if tool_calls:
                    session_manager.add_tool_calls(session_id, tool_calls)

        except Exception as exc:
            logger.exception(f"Graph execution failed for job {job_id}")
            job_manager.update_state(
                job_id, "failed", error=f"Research execution failed: {exc}"
            )
            job_manager.push_event(job_id, {"type": "failed", "error": f"Research execution failed: {exc}"})
            trace_ctx["status"] = "failed"
            trace_ctx["output"] = str(exc)
```

**Important note:** The clarification pause happens *within* the graph's astream loop. When the clarifier yields, we intercept it and wait. The graph continues streaming subsequent nodes after we unblock. This works because LangGraph's `astream(stream_mode="updates")` yields node-by-node — the graph doesn't advance to the planner until we consume the clarifier's yield. However, we need to be aware that the graph has already computed the clarifier output and the planner will use the state *as it was*. To make the user's answer take effect, we need to update the state that the planner sees.

**Limitation:** LangGraph's compiled graph doesn't allow modifying state mid-stream in `astream`. The `accumulated_state` we build is our local copy — the graph's internal state is separate. This means the clarification answer won't actually reach the planner through `astream`.

**Alternative approach:** Instead of trying to pause mid-graph, run the clarifier *before* the graph. This is simpler and more reliable:

Replace the entire `_run_graph` clarification block approach with a pre-graph clarification step. Add this logic in the `submit_research` endpoint:

Actually, the cleanest approach is: run clarifier as a standalone call before invoking the graph, and only start the graph if no clarification is needed (or after the user answers).

Let me revise. In `submit_research` endpoint in `app.py`, for the research path:

```python
        else:
            # Deep research path: full graph pipeline
            graph = getattr(request.app.state, "graph", None)
            model = getattr(request.app.state, "model", None)
            if graph is None:
                job_manager.update_state(
                    job_id, "failed", error="Research graph not initialized"
                )
                return {"job_id": job_id, "status": "failed", "session_id": session.session_id}

            config = getattr(request.app.state, "config", None)
            budget = None
            if config:
                from deep_research.models import Budget
                budget = Budget(
                    max_iterations=config.max_iterations,
                    max_tool_calls=config.max_tool_calls,
                    time_cap_seconds=config.time_cap_seconds,
                )

            initial_state = create_initial_state(
                user_query=req.query,
                selected_tools=req.tools,
                output_mode=req.output_mode,
                job_id=job_id,
                trace_id=f"trace-{uuid.uuid4().hex[:12]}",
                budget=budget,
                conversation_history=prior_history,
                prior_evidence=prior_evidence,
            )

            asyncio.create_task(
                _run_research_with_clarification(
                    model=model,
                    graph=graph,
                    job_manager=job_manager,
                    job_id=job_id,
                    initial_state=initial_state,
                    session_manager=session_manager,
                    session_id=session.session_id,
                )
            )
```

And add `_run_research_with_clarification`:

```python
async def _run_research_with_clarification(
    model: Any,
    graph: Any,
    job_manager: JobManager,
    job_id: str,
    initial_state: dict,
    session_manager: SessionManager,
    session_id: str,
) -> None:
    """Run clarifier first, handle clarification if needed, then run graph."""
    from deep_research.nodes.clarifier import clarifier_node

    job_manager.update_state(job_id, "running", current_node="clarifier")
    job_manager.push_event(job_id, {"type": "node_started", "node": "clarifier"})

    # Run clarifier standalone
    clarifier_result = await clarifier_node(initial_state, model=model)

    if clarifier_result.get("needs_clarification"):
        question = clarifier_result.get("clarification_question", "")
        options = clarifier_result.get("clarification_options", [])
        best_guess = clarifier_result.get("clarified_query", initial_state["user_query"])

        # Push event to UI
        job_manager.push_event(job_id, {
            "type": "clarification_needed",
            "question": question,
            "options": options,
        })

        # Wait for user answer
        wait_event = job_manager.request_clarification(job_id)
        try:
            await asyncio.wait_for(wait_event.wait(), timeout=30.0)
            answer = job_manager.get_clarification_answer(job_id)
            if answer:
                initial_state["clarified_query"] = answer
                job_manager.push_event(job_id, {
                    "type": "clarification_resolved",
                    "answer": answer,
                })
        except asyncio.TimeoutError:
            job_manager.get_clarification_answer(job_id)  # cleanup
            initial_state["clarified_query"] = best_guess
            job_manager.push_event(job_id, {
                "type": "clarification_timeout",
                "best_guess": best_guess,
            })
    else:
        initial_state["clarified_query"] = clarifier_result.get(
            "clarified_query", initial_state["user_query"]
        )

    # Mark clarification as resolved so graph's clarifier is a no-op
    initial_state["needs_clarification"] = False

    # Now run the full graph (clarifier will see clarified_query is already set)
    await _run_graph(
        graph, job_manager, job_id, initial_state,
        session_manager=session_manager,
        session_id=session_id,
    )
```

Update the clarifier node to short-circuit when clarified_query is already set:

In `clarifier_node`, add at the top of the function:

```python
    # If clarified_query is already set (from pre-graph clarification), skip
    if state.get("clarified_query"):
        return {"clarified_query": state["clarified_query"]}
```

**Step 4: Add test for clarification flow**

Add to `tests/test_api.py`:

```python
async def test_clarify_endpoint(client):
    """Test that /clarify endpoint returns 404 for unknown job."""
    response = await client.post(
        "/api/research/nonexistent/clarify",
        json={"answer": "test"},
    )
    assert response.status_code == 404


async def test_clarify_not_waiting(client):
    """Test that /clarify returns 400 if job isn't waiting for clarification."""
    # Submit a research job first
    res = await client.post("/api/research", json={
        "query": "test", "tools": [], "response_mode": "research"
    })
    job_id = res.json()["job_id"]
    response = await client.post(
        f"/api/research/{job_id}/clarify",
        json={"answer": "option A"},
    )
    assert response.status_code == 400
```

**Step 5: Run tests**

```bash
cd /tmp/db_deep_research_agent && PYTHONPATH=src pytest tests/test_api.py -v --timeout=30
```

**Step 6: Commit**

```bash
cd /tmp/db_deep_research_agent
git add src/deep_research/api/app.py src/deep_research/api/jobs.py src/deep_research/nodes/clarifier.py src/deep_research/state.py
git commit -m "feat: add clarification pause/resume in research pipeline"
```

---

## Task 6: Clarification — Frontend ClarificationPrompt Component

**Files:**
- Create: `ui/src/components/ClarificationPrompt.tsx`
- Modify: `ui/src/types.ts`
- Modify: `ui/src/api.ts`

**Step 1: Add ClarificationRequest type**

In `ui/src/types.ts`, add:

```typescript
export interface ClarificationRequest {
  jobId: string;
  question: string;
  options: string[];
}
```

**Step 2: Add submitClarification to api.ts**

In `ui/src/api.ts`, add:

```typescript
export async function submitClarification(
  jobId: string,
  answer: string
): Promise<void> {
  await client.post(`/api/research/${jobId}/clarify`, { answer });
}
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
git commit -m "feat(ui): add ClarificationPrompt component and API function"
```

---

## Task 7: Clarification — Wire into useResearch and ChatPanel

**Files:**
- Modify: `ui/src/hooks/useResearch.ts`
- Modify: `ui/src/components/ChatPanel.tsx`

**Step 1: Handle clarification events in useResearch**

In `ui/src/hooks/useResearch.ts`, add:

1. Import `submitClarification` from api and `ClarificationRequest` from types
2. Add `clarificationRequest` state
3. Handle `clarification_needed`, `clarification_resolved`, `clarification_timeout` events
4. Add `answerClarification` function

Replace the full hook with:

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
  const pendingTokenUsageRef = useRef<{ input: number; output: number } | null>(null);

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
                  content: `No response — proceeding with: "${event.best_guess}"`,
                  timestamp: new Date().toISOString(),
                  jobId,
                },
              ]);
            }
          } else if (event.type === "completed") {
            const pendingTableData = pendingTableDataRef.current;
            pendingTableDataRef.current = null;
            const tokenUsage = event.token_usage || null;
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
                    tokenUsage: tokenUsage ?? undefined,
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
                    tokenUsage: tokenUsage ?? undefined,
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
      await submitClarification(req.jobId, answer);
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

**Step 2: Update streamJob event type in api.ts**

In `ui/src/api.ts`, update the `onEvent` parameter type in `streamJob` to include new fields:

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
    question?: string;
    options?: string[];
    best_guess?: string;
    answer?: string;
    token_usage?: { input: number; output: number };
  }) => void,
```

**Step 3: Update ChatPanel to render ClarificationPrompt**

In `ui/src/components/ChatPanel.tsx`, add import and rendering:

Add to imports:
```typescript
import { ClarificationPrompt } from "./ClarificationPrompt";
import type { ClarificationRequest } from "../types";
```

Add props:
```typescript
  clarificationRequest: ClarificationRequest | null;
  onAnswerClarification: (answer: string) => void;
```

Add in JSX, after messages and before `{showTypingIndicator && ...}`:
```tsx
          {clarificationRequest && (
            <ClarificationPrompt
              request={clarificationRequest}
              onAnswer={onAnswerClarification}
            />
          )}
```

**Step 4: Update App.tsx to pass new props**

In `ui/src/App.tsx`, destructure `clarificationRequest` and `answerClarification` from `useResearch()`, and pass them to `ChatPanel`:

```tsx
  const { messages, currentJob, isLoading, error, clarificationRequest, send, cancel, answerClarification, rate } =
    useResearch();
```

And in the ChatPanel JSX:
```tsx
              clarificationRequest={clarificationRequest}
              onAnswerClarification={answerClarification}
```

**Step 5: Verify build**

```bash
cd /tmp/db_deep_research_agent/ui && npx tsc -b 2>&1
```

**Step 6: Commit**

```bash
cd /tmp/db_deep_research_agent
git add ui/src/hooks/useResearch.ts ui/src/api.ts ui/src/components/ChatPanel.tsx ui/src/App.tsx
git commit -m "feat(ui): wire clarification flow into useResearch and ChatPanel"
```

---

## Task 8: Token Usage — Backend Extraction

**Files:**
- Modify: `src/deep_research/api/app.py`
- Modify: `src/deep_research/state.py`

**Step 1: Add token usage helper and accumulation to _run_quick_reply**

In `src/deep_research/api/app.py`, add helper function at module level:

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

In `_run_quick_reply`, after the LLM call (the `response = await model.ainvoke(...)` line), add:

```python
            token_usage = _extract_token_usage(response)
```

And update the completed event to include it:

```python
            completed_event: dict[str, Any] = {"type": "completed", "result": reply}
            if token_usage and (token_usage["input"] > 0 or token_usage["output"] > 0):
                completed_event["token_usage"] = token_usage
            job_manager.push_event(job_id, completed_event)
```

(Remove the old `job_manager.push_event(job_id, {"type": "completed", "result": reply})` line.)

**Step 2: Add _token_usage to ResearchState**

In `src/deep_research/state.py`, add to `ResearchState`:

```python
    _token_usage: dict[str, int]
```

And in `create_initial_state`, add:

```python
        _token_usage={"input": 0, "output": 0},
```

**Step 3: Accumulate tokens in synthesizer (where streaming happens)**

In `src/deep_research/nodes/synthesizer.py`, after the streaming/invoke block, add token accumulation:

For the streaming path (after `final_output = "".join(chunks)`):
```python
        # Token usage not available from streaming — will be estimated
```

For the non-streaming path (after `final_output = response.content`):
```python
        # Accumulate token usage
        meta = getattr(response, "response_metadata", {}) or {}
        usage = meta.get("usage", {})
        prior = state.get("_token_usage", {"input": 0, "output": 0})
        token_usage = {
            "input": prior["input"] + (usage.get("input_tokens", 0) or usage.get("prompt_tokens", 0)),
            "output": prior["output"] + (usage.get("output_tokens", 0) or usage.get("completion_tokens", 0)),
        }
```

And include in the return:
```python
    return {"final_output": final_output, "_token_usage": token_usage}
```

For the streaming path, we can't easily get token counts from `astream`. Return the prior accumulation:

```python
    if job_manager and job_id and hasattr(model, "astream"):
        chunks = []
        async for chunk in model.astream(messages):
            token = chunk.content if hasattr(chunk, "content") else str(chunk)
            if token:
                chunks.append(token)
                job_manager.push_event(job_id, {"type": "token", "content": token})
        final_output = "".join(chunks)
        token_usage = state.get("_token_usage", {"input": 0, "output": 0})
    else:
        response = await model.ainvoke(messages)
        final_output = response.content
        meta = getattr(response, "response_metadata", {}) or {}
        usage = meta.get("usage", {})
        prior = state.get("_token_usage", {"input": 0, "output": 0})
        token_usage = {
            "input": prior["input"] + (usage.get("input_tokens", 0) or usage.get("prompt_tokens", 0)),
            "output": prior["output"] + (usage.get("output_tokens", 0) or usage.get("completion_tokens", 0)),
        }

    return {"final_output": final_output, "_token_usage": token_usage}
```

**Step 4: Verify tests pass**

```bash
cd /tmp/db_deep_research_agent && PYTHONPATH=src pytest tests/ -v --timeout=30 2>&1 | tail -20
```

**Step 5: Commit**

```bash
cd /tmp/db_deep_research_agent
git add src/deep_research/api/app.py src/deep_research/state.py src/deep_research/nodes/synthesizer.py
git commit -m "feat: extract and accumulate token usage from LLM responses"
```

---

## Task 9: Token Usage — Frontend Display

Token usage display in MessageBubble was already added in Task 3 (the `tokenUsage` label). The `useResearch` hook was updated in Task 7 to extract `token_usage` from the completed event. The type was added in Task 3.

This task verifies everything works end-to-end.

**Step 1: Verify the full build**

```bash
cd /tmp/db_deep_research_agent/ui && npm run build 2>&1 | tail -10
```

**Step 2: Run all backend tests**

```bash
cd /tmp/db_deep_research_agent && PYTHONPATH=src pytest tests/ -v --timeout=30 2>&1 | tail -30
```

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

```python
from databricks.sdk import WorkspaceClient
from databricks.sdk.service.workspace import ImportFormat
import os, base64

w = WorkspaceClient()
ws_base = "/Workspace/Users/brian.law@databricks.com/deep-research-agent"
local_base = "/tmp/db_deep_research_agent"

uploaded = 0
all_files = []

# UI dist
for root, dirs, files in os.walk(f"{local_base}/ui/dist"):
    for f in files:
        lp = os.path.join(root, f)
        all_files.append((lp, os.path.relpath(lp, local_base)))

# UI src
for root, dirs, files in os.walk(f"{local_base}/ui/src"):
    for f in files:
        lp = os.path.join(root, f)
        all_files.append((lp, os.path.relpath(lp, local_base)))

# Backend src
for root, dirs, files in os.walk(f"{local_base}/src"):
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
2. Test citations: Ask "What was CBA's closing price?" in Deep Research mode with Genie selected → should see styled citation badges
3. Test clarification: Ask something ambiguous like "Compare revenue" → should see clarification prompt (if clarifier detects ambiguity)
4. Test token count: Check that completed messages show a token count label
