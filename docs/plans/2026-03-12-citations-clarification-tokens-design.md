# Citations, Clarification & Token Display — Design Document

**Goal:** Add three features to the research agent: enhanced text citations styled as badges, interactive user clarification before research begins, and token usage display on completed messages.

**Architecture:** All three features extend the existing SSE event stream + React UI pattern. Citations are prompt-only + UI styling. Clarification adds a pause/resume mechanism to the graph pipeline. Token display adds metadata extraction from LLM responses.

**Tech Stack:** Python/FastAPI backend, React/TypeScript/Tailwind frontend, ChatDatabricks LLM, LangGraph pipeline.

---

## Feature 1: Enhanced Text Citations

### Current State
- Synthesizer prompt says `[Source: <title>]` but output is unstyled plain text
- Evidence model has `source_id`, `title`, `uri`, `tool_that_produced_it`
- The synthesizer receives evidence as `- [source_id] snippet` — no tool name

### Design
**Backend:** Update synthesizer prompts to produce richer citations. Change the evidence context format from `- [source_id] snippet` to `- [Source: Tool Display Name — title] snippet` so the LLM naturally cites with tool context.

**Frontend:** In MessageBubble's markdown rendering, use a custom ReactMarkdown component that detects `[Source: ...]` patterns via regex and renders them as inline styled pills/badges (small rounded elements with `bg-warm-sage/15 text-warm-sage` styling).

### Files
- `src/deep_research/prompts.py` — Update SYNTHESIZER_CHAT_SYSTEM and SYNTHESIZER_REPORT_SYSTEM
- `src/deep_research/nodes/synthesizer.py` — Include tool_name in evidence context
- `ui/src/components/MessageBubble.tsx` — Add citation badge rendering

---

## Feature 2: Interactive Clarification

### Current State
- Clarifier node auto-rewrites queries, never asks the user
- No mechanism for pipeline pause/resume
- SSE stream only has: node_started, token, table_data, completed, failed

### Design

**New SSE event:** `clarification_needed`
```json
{"type": "clarification_needed", "question": "...", "options": ["A", "B", "C"], "best_guess": "..."}
```

**New endpoint:** `POST /api/research/{job_id}/clarify`
```json
{"answer": "user's choice or free text"}
```

**Backend flow:**
1. Clarifier node returns either `{"clarified_query": "..."}` (unambiguous) or `{"needs_clarification": true, "question": "...", "options": [...], "best_guess": "..."}`
2. `_run_graph` checks clarifier output. If `needs_clarification`:
   - Push `clarification_needed` event to SSE
   - Create an `asyncio.Event` keyed by job_id, wait on it (with 30s timeout)
   - On timeout, use `best_guess` and continue
3. `/clarify` endpoint sets the answer on the event, unblocking the pipeline
4. Pipeline resumes from planner with the user-provided clarified query

**Frontend flow:**
1. `useResearch` handles `clarification_needed` event → sets a `clarificationRequest` state
2. New `ClarificationPrompt` component renders in the chat area with the question + clickable option buttons + free-text input
3. User clicks an option or types → calls `submitClarification(jobId, answer)` → clears the prompt
4. Pipeline resumes, progress continues as normal

**Clarifier prompt change:** Output schema becomes:
```json
// Unambiguous:
{"clarified_query": "..."}
// Ambiguous:
{"needs_clarification": true, "question": "...", "options": ["...", "..."], "best_guess": "..."}
```

**Timeout behavior:** If user doesn't answer in 30 seconds, the pipeline proceeds with `best_guess`. A message is shown: "No response — proceeding with: [best_guess]"

### Files
- `src/deep_research/prompts.py` — Rewrite CLARIFIER_SYSTEM for dual-mode output
- `src/deep_research/nodes/clarifier.py` — Return needs_clarification flag
- `src/deep_research/api/app.py` — Add /clarify endpoint, modify _run_graph for pause/resume
- `src/deep_research/api/jobs.py` — Add clarification event/answer storage on JobManager
- `ui/src/api.ts` — Add submitClarification function
- `ui/src/types.ts` — Add ClarificationRequest type
- `ui/src/hooks/useResearch.ts` — Handle clarification_needed event
- `ui/src/components/ClarificationPrompt.tsx` — New component
- `ui/src/components/ChatPanel.tsx` — Render ClarificationPrompt

---

## Feature 3: Token Usage Display

### Current State
- No token counting anywhere
- ChatDatabricks returns usage in `response.response_metadata["usage"]` (input_tokens, output_tokens)
- JobStatus has no token_usage field

### Design

**Backend:**
- Add `token_usage: dict` field to JobStatus (default empty dict)
- In each graph node, after `model.ainvoke()` or `model.astream()`, extract token usage from response metadata and accumulate into a running total
- Pass accumulated totals through the graph state as `_token_usage: {input: int, output: int}`
- When job completes, include `token_usage` in the completed event
- For quick reply: extract from the single response

**Token extraction helper:**
```python
def _extract_token_usage(response) -> dict[str, int]:
    meta = getattr(response, "response_metadata", {}) or {}
    usage = meta.get("usage", {})
    return {
        "input": usage.get("input_tokens", 0) or usage.get("prompt_tokens", 0),
        "output": usage.get("output_tokens", 0) or usage.get("completion_tokens", 0),
    }
```

**Frontend:**
- Add `tokenUsage?: {input: number, output: number}` to Message type
- Extract from completed event in useResearch
- Show below assistant messages as a small muted label: "1,550 tokens"
- Only on completed (non-streaming) assistant messages

### Files
- `src/deep_research/api/jobs.py` — Add token_usage to JobStatus
- `src/deep_research/api/app.py` — Include token_usage in completed event, add helper
- `src/deep_research/nodes/synthesizer.py` — Extract and accumulate tokens
- `src/deep_research/state.py` — Add _token_usage to ResearchState
- `ui/src/types.ts` — Add tokenUsage to Message
- `ui/src/hooks/useResearch.ts` — Extract token_usage from completed event
- `ui/src/components/MessageBubble.tsx` — Render token count label
