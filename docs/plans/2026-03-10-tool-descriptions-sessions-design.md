# Phase 5: Rich Tool Descriptions + Session State

## Problem

1. **Planner sees only tool names** — `Available tools: genie_aviation, vector_search_anz` with no descriptions, data schemas, or capabilities. Tool assignment relies on naming conventions.
2. **No session state** — every request starts from scratch. Follow-up questions re-research everything. No conversational memory.

## Design

### Feature 1: Rich Tool Descriptions

**Change**: Pass full tool catalog (name, display_name, description, type, capability) to the planner prompt instead of just names.

**Planner prompt before**:
```
Available tools: genie_aviation, vector_search_anz, knowledge_assistant
```

**Planner prompt after**:
```
Available tools:
- genie_aviation [data_query, read]: "Airport & Airline Passenger Analytics (Genie)" — Query airport, airline, flight, and passenger data via Genie
- vector_search_anz [semantic_search, read]: "ANZ 2024 Annual Report (Vector Search)" — Search the ANZ 2024 annual report via vector search index
- knowledge_assistant [knowledge_retrieval, read]: "Knowledge Assistant" — Knowledge assistant for answering questions from curated knowledge base
```

**Files changed**:
- `prompts.py` — update `PLANNER_SYSTEM` format to accept `{tool_catalog}` instead of `{tools}`
- `graph.py` — thread `mcp_manager` into `planner_node` via `partial()`
- `nodes/planner.py` — build tool catalog string from `mcp_manager.get_available_servers()`, pass to prompt. Fall back to just names if mcp_manager is None (tests).
- `nodes/query_adapter.py` — also receives tool descriptions for better reformulation context

### Feature 2: Minimal Session State

**SessionManager** — in-memory dict keyed by session_id with TTL expiry.

```python
@dataclass
class Session:
    session_id: str
    created_at: datetime
    last_active: datetime
    conversation_history: list[dict]  # [{role, content}, ...]
    accumulated_evidence: list[Evidence]
    accumulated_tool_calls: list[ToolCall]
    ttl_minutes: int = 30
```

**API changes**:
- `POST /api/research` — accepts optional `session_id` in request body
- Response includes `session_id` (auto-generated if not provided)
- `GET /api/sessions/{session_id}` — get session info (conversation history, evidence count)
- `DELETE /api/sessions/{session_id}` — end a session

**Graph state changes** — add to `ResearchState`:
- `conversation_history: list[dict]` — prior turns
- `prior_evidence: list[Evidence]` — evidence from previous turns (read-only context)

**Node behavior with session**:
- **Clarifier** — receives conversation history, contextualizes the new query against prior turns
- **Planner** — receives prior evidence summary, generates sub-questions only for NEW gaps
- **Researcher** — only executes new sub-questions (prior evidence already available)
- **Evaluator** — considers both prior + new evidence for sufficiency
- **Synthesizer** — synthesizes from all evidence (prior + new), aware of conversation context

**Drill-down flow**: User sends "Tell me more about X" with same session_id. Clarifier interprets it as a focused query. Planner sees prior evidence covers the broad topic, creates sub-questions only for the drill-down area.

**Files changed**:
- New: `session.py` — SessionManager class + Session dataclass
- `state.py` — add `conversation_history` and `prior_evidence` fields
- `api/app.py` — wire session_id into request/response, load/save session state
- `nodes/clarifier.py` — include conversation history in prompt
- `nodes/planner.py` — include prior evidence summary in prompt
- `nodes/evaluator.py` — count prior + new evidence
- `nodes/synthesizer.py` — include conversation context
- `prompts.py` — update CLARIFIER_SYSTEM, PLANNER_SYSTEM for conversation context

## Implementation Order

1. P5-1: Rich tool descriptions (planner + query_adapter)
2. P5-2: Session dataclass + SessionManager
3. P5-3: Wire sessions into API layer
4. P5-4: Update clarifier + planner for conversation context
5. P5-5: Update evaluator + synthesizer for accumulated evidence
6. P5-6: Tests for both features
7. P5-7: Deploy and verify

## Out of Scope (future)

- Persistent session storage (Delta table)
- LangGraph checkpointing
- Session sharing between users
- Branching conversations (fork a session)
