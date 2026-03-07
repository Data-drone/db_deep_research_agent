# Deep Research Bot — Design Document

**Date:** 2026-03-07
**Status:** Approved (v2 — incorporating GPT-5-4 architecture review)

## Overview

An enterprise deep research agent deployed as a Databricks App. It connects to internal data sources via MCP (Model Context Protocol), performs multi-step research with iterative refinement, and produces either conversational answers or structured reports with citations. Users select which tools the agent uses per query via a React UI.

## Architecture

```
┌─────────────────────────────────┐
│        React Frontend           │
│   (Chat + Report Viewer +       │
│    Tool Selector Sidebar)       │
├─────────────────────────────────┤
│        FastAPI Backend          │
│   (WebSocket/SSE for streaming, │
│    REST for config/tools/       │
│    feedback, job management)    │
├─────────────────────────────────┤
│       LangGraph Agent           │
│  (Clarifier → Planner →        │
│   Authorizer → Researcher →    │
│   Normalizer → Evaluator →     │
│   Compressor → Synthesizer →   │
│   Verifier)                     │
├─────────────────────────────────┤
│      MCP Client Layer           │
│  (Databricks managed +          │
│   custom MCP servers)           │
├─────────────────────────────────┤
│     Databricks Platform         │
│  Genie Agents │ Vector Search   │
│  FMAPIs       │ MLflow          │
└─────────────────────────────────┘
```

### Layers

- **React Frontend** — chat interface with streaming, report rendering panel, sidebar with tool checkboxes for selecting MCP servers per query, research progress indicators.
- **FastAPI Backend** — serves the React app, handles WebSocket streaming for real-time agent output, REST endpoints for tool configuration, feedback submission, and job management (cancel, resume, status).
- **LangGraph Agent** — multi-node research graph orchestrated by LangGraph. Uses Databricks Foundation Model APIs for inference. Includes clarification, authorization, evidence normalization, and citation verification stages.
- **MCP Client Layer** — connects to Databricks Managed MCP servers (Genie, Vector Search) and custom MCP servers. Connections established at startup via service principal.
- **Databricks Platform** — Genie for structured data queries, Vector Search for unstructured/document retrieval, FMAPIs for LLM inference, MLflow for tracing and feedback.

## Agent Flow

```
[User Query + Selected Tools]
        │
        ▼
  ┌────────────┐
  │ Clarifier   │  Detects ambiguity. Asks user to
  └─────┬──────┘  clarify if needed, or passes through.
        │
        ▼
  ┌────────────┐
  │ Planner     │  Breaks query into sub-questions,
  └─────┬──────┘  decides which tools to use,
        │         estimates budget/depth.
        ▼
  ┌────────────┐
  │ Authorizer  │  Validates tool access against
  └─────┬──────┘  policy. Filters unauthorized tools.
        │
        ▼
  ┌────────────┐
  │ Researcher  │  Executes tool calls via MCP
  └─────┬──────┘  (Genie queries, Vector Search, etc.)
        │
        ▼
  ┌────────────┐
  │ Normalizer  │  Converts raw tool output into
  └─────┬──────┘  structured Evidence objects.
        │
        ▼
  ┌────────────┐
  │ Evaluator   │  Checks sufficiency against
  └─────┬──────┘  formal stop conditions.
        │
   ┌────┴────┐
   │         │
 enough   gaps found
   │         │
   │         ▼
   │    loops back to Planner
   │    (subject to budget/iteration caps)
   │
   ▼
 ┌──────────────┐
 │ Compressor    │  Deduplicates, preserves
 └──────┬───────┘  uncertainties and contradictions.
        │
        ▼
 ┌──────────────┐
 │ Synthesizer   │  Two modes:
 └──────┬───────┘  chat answer or structured report.
        │
        ▼
 ┌──────────────┐
 │ Verifier      │  Checks claims have citations,
 └──────────────┘  softens unsupported claims,
                   flags contradictions.
        │
        ▼
   [Response to User]
```

### State Schema

The LangGraph state is a typed, versioned model. This ensures consistency across nodes and enables resumability.

```python
class ResearchState(TypedDict):
    # Input
    user_query: str
    selected_tools: list[str]
    output_mode: Literal["chat", "report"]

    # Clarification
    clarified_query: str | None
    clarification_needed: bool

    # Planning
    research_plan: list[SubQuestion]
    tool_assignments: dict[str, list[str]]  # sub-question -> tools
    budget: Budget  # max iterations, max tool calls, time cap

    # Research
    evidence: list[Evidence]  # structured evidence objects
    tool_call_log: list[ToolCall]
    iteration_count: int

    # Evaluation
    sufficiency_score: float
    missing_facets: list[str]
    stop_reason: str | None

    # Output
    compressed_findings: CompressedFindings
    final_output: str
    citations: list[Citation]
    verification_result: VerificationResult

    # Tracing
    trace_id: str
    job_id: str
```

### Evidence Model

Every piece of retrieved information is normalized into a structured Evidence object:

```python
class Evidence(TypedDict):
    evidence_id: str
    source_id: str              # MCP server + tool identifier
    source_type: str            # "genie_query", "vector_search", etc.
    title: str                  # human-readable source name
    uri: str | None             # table, document, or resource identifier
    snippet: str                # extracted claim or data
    confidence: float           # 0.0-1.0
    freshness: str              # timestamp of source data
    tool_that_produced_it: str
    iteration: int              # which research loop produced this
    metadata: dict              # additional context
```

### Nodes

- **Clarifier** — analyzes the query for ambiguity. If the query is underspecified (e.g. missing time range, unclear scope, undefined terms), prompts the user for clarification before proceeding. Passes through clear queries directly.
- **Planner** — decomposes the clarified query into sub-questions, assigns tools to each, and sets a budget (max iterations, max tool calls, time cap). Can suggest tools automatically based on query type; user selections constrain rather than dictate.
- **Authorizer** — validates that the planned tool usage complies with the access policy. Filters out tools the user is not entitled to use. Logs effective identity for each tool call.
- **Researcher** — calls MCP tools, collects raw results, logs each tool call.
- **Normalizer** — converts raw tool outputs into structured Evidence objects with source attribution, confidence, and freshness metadata. Treats all tool output as untrusted; strips suspicious content.
- **Evaluator** — assesses research sufficiency against formal stop conditions (see below). Outputs a structured decision with sufficiency score, missing facets, and stop/continue recommendation.
- **Compressor** — deduplicates overlapping evidence. Preserves key findings, open questions, uncertainties, and contradictory evidence. Uses structured memory rather than pure prose summarization. Keeps a raw evidence index available for final synthesis.
- **Synthesizer** — produces either a conversational answer or a structured report with sections, citations, and sources. Reports include assumptions, limitations, and "what would change this conclusion" sections.
- **Verifier** — final pass to check every major claim is supported by evidence, soften or remove unsupported claims, verify citations resolve correctly, and acknowledge contradictions.

All nodes are traced via MLflow autologging.

### Evaluator Stop Conditions

The Evaluator uses formal criteria to decide whether to continue the research loop:

- **Max iterations** — hard cap (configurable, default 5)
- **Max tool calls** — total tool invocation budget
- **Time cap** — wall-clock time limit per research job
- **Diminishing returns** — sufficiency score improvement below threshold between iterations
- **Evidence coverage** — all planned sub-questions have at least one evidence item
- **Unresolved critical questions** — if zero, research is sufficient

The Evaluator outputs a structured decision:

```python
class EvaluatorDecision(TypedDict):
    sufficiency_score: float        # 0.0-1.0
    missing_facets: list[str]       # what's still unknown
    recommended_actions: list[str]  # specific follow-up tool calls
    decision: Literal["continue", "stop"]
    reason: str
```

## MCP Configuration

Tools are configured via a YAML file:

```yaml
managed_servers:
  genie_sales:
    url: "{host}/api/2.0/mcp/genie/{space_id}"
    display_name: "Sales Data (Genie)"
    enabled: true
    risk_tier: "safe"           # safe | restricted | privileged
    capability: "read"          # read | read_write
    description: "Query sales data via Genie"
  vector_search_kb:
    url: "{host}/api/2.0/mcp/vector-search/{index}"
    display_name: "Knowledge Base"
    enabled: true
    risk_tier: "safe"
    capability: "read"
    description: "Search internal knowledge base documents"

custom_servers:
  # Future custom MCP servers hosted on Databricks Apps
```

- **Managed MCP servers** — Databricks-hosted (Genie, Vector Search). Auth handled automatically via workspace credentials.
- **Custom MCP servers** — anything built and hosted on Databricks Apps.
- Each server has a `display_name`, `enabled` flag, `risk_tier`, and `capability` classification.
- At startup, the app connects to all enabled servers, discovers their tools, and populates the sidebar.

### Tool Governance

- **Risk tiers** — `safe` (read-only analytics), `restricted` (sensitive data), `privileged` (write operations). Default research workflows use `safe` tools only.
- **Capability flags** — `read` vs `read_write`. Research mode restricts to read-only tools.
- **Access policy** — tools can be filtered by user/group/domain. Enforced by the Authorizer node before any tool execution.
- **Input validation** — tool inputs are validated against schemas before MCP calls.
- **Output sanitization** — all tool outputs are treated as untrusted. Suspicious content (prompt injection patterns) is stripped or flagged.
- **Circuit breakers** — per-tool timeout and failure limits. Slow/failing tools are gracefully degraded.
- **Tool call logging** — every tool invocation is logged with effective identity, inputs, outputs, and latency.

### UI Tool Selection

The system suggests relevant tools automatically based on the query. Users can override:

```
Available Tools
★ Sales Data (Genie)        [auto-suggested]
★ Knowledge Base (Vector Search)  [auto-suggested]
☐ HR Genie (Genie)
☐ Finance Reports (Genie)
```

- Stars indicate auto-suggested tools based on query analysis
- Users can add or remove tools from the selection
- Policy layer enforces the final allowed set regardless of user selection

Research progress is visible in the UI:
- Clarifying...
- Planning research strategy...
- Searching Sales Data...
- Analyzing results (iteration 2/5)...
- Compressing findings...
- Writing report...
- Verifying citations...

## Feedback & Observability

### User Feedback (v1)

Thumbs up / thumbs down on every response.

```
React UI (thumbs up/down)
  → POST /api/feedback { query_id, rating, comment? }
    → FastAPI backend
      → mlflow.log_feedback(
            trace_id=trace_id,
            name="user_rating",
            value="thumbs_up" | "thumbs_down",
            source=AssessmentSource(
                source_type=AssessmentSourceType.HUMAN,
                source_id=user_id
            ),
            rationale=comment
        )
```

### MLflow Tracing

- All agent executions produce full traces across all nodes.
- Traces include: node inputs/outputs, tool calls, LLM prompts/responses, evaluator decisions.
- User ratings attach to traces via `trace_id`.
- Traces and feedback are viewable in MLflow UI.
- **Redaction rules** — sensitive values (PII, credentials) are hashed or redacted before logging. Trace schemas are defined per node to control what gets logged.

### Future Feedback Enhancements

MLflow `log_feedback` supports richer feedback types for later versions:
- Numeric scores (1-5 stars, 0.0-1.0 relevance)
- Multi-dimensional ratings (dict with accuracy, helpfulness, etc.)
- Per-step feedback via `span_id` (e.g. rate individual tool results)
- LLM judge automated scoring via `source_type=LLM_JUDGE`
- Feedback override workflow via `mlflow.override_feedback()`
- Structured feedback taxonomy (wrong facts, missing sources, too shallow, wrong tool, too slow)

### Databricks Eval Integration

Feedback data is compatible with Mosaic AI Agent Evaluation:
- Compare model versions by rating distribution
- Build eval datasets from highly-rated responses
- Debug low-rated responses by inspecting the full trace
- Build golden datasets for: factuality, citation correctness, tool selection, completeness, latency
- Log evaluator decisions and stop reasons for analysis
- Compare answer quality by graph version, prompt version, and toolset

## Serving Architecture

Hybrid approach:
- **FastAPI handles real-time UX** — WebSocket streaming, progress updates as the agent works through its research loop.
- **MLflow logs everything** — traces, tool calls, final outputs, all via the SDK from within the FastAPI backend.
- **User ratings go to MLflow** — React frontend → FastAPI → `mlflow.log_feedback()`.

This gives full Databricks eval integration without the UX constraints of pure Model Serving (timeout limits, clunky streaming for long-running research).

### Session & Job Management

Research queries can take 30-60+ seconds. The serving layer handles this:

- **Job IDs** — each research query gets a unique job ID. Clients can poll status or receive updates via WebSocket.
- **Cancellation** — users can cancel in-progress research. The agent graph checks for cancellation between nodes.
- **Resume on disconnect** — if the browser disconnects, the research job continues. On reconnect, the client receives the completed or in-progress result.
- **Concurrency limits** — configurable per-user and global limits on concurrent research jobs.
- **State persistence** — LangGraph state is checkpointed to persistent storage (not in-memory only) to survive process restarts and enable resumability.
- **Heartbeat protocol** — WebSocket connection sends periodic heartbeats. Stale connections are cleaned up.

### Databricks App Considerations

- **Cold starts** — prewarm MCP connections and cache tool manifests on startup.
- **Compute sizing** — configure app compute based on expected concurrency.
- **WebSocket support** — verify Databricks App networking supports long-lived WebSocket connections; fall back to SSE polling if needed.
- **Graceful degradation** — if an MCP server is unavailable, the agent proceeds with remaining tools and notes the unavailability in its output.

## Authorization & Security

### Auth Model

- MCP servers authenticate via workspace service principal at startup.
- **Risk**: service principal may have broader access than the user.
- **Mitigation**: the Authorizer node enforces per-user/group tool allowlists before execution. Tools are tagged with data domains and access tiers.

### Prompt Injection Defense

Tool outputs are treated as untrusted external content:
- Strict delimiter boundaries between instructions and tool results in prompts
- Instruction hierarchy: system prompt > tool output
- Content scanning for injection patterns before re-entering prompts
- Evaluator checks for anomalous agent behavior post-tool-call

### Data Access

- Read-only tools for research workflows by default
- Write-capable tools require explicit `privileged` tier authorization
- Sensitive data tools require `restricted` tier and user/group allowlist
- Effective identity (user + service principal) logged for every tool call

## Key Libraries

| Library | Purpose |
|---------|---------|
| `langgraph` | Agent orchestration |
| `databricks-langchain` | Databricks LLM + embeddings integration |
| `mcp` | MCP client SDK |
| `mlflow` | Tracing + feedback logging |
| `fastapi` | Backend API with WebSocket streaming |
| React + Vite | Frontend |

## Deployment

Deployed as a Databricks App:
- FastAPI backend serves the React frontend and agent API
- MCP server connections authenticated via workspace service principal
- MLflow tracking configured to the workspace MLflow instance
- LangGraph state checkpointed to persistent storage

## Pre-Production Checklist

Questions to resolve before production deployment:

1. What is the final typed state schema version?
2. Where is graph/session state persisted (Delta table, Redis, filesystem)?
3. How is user-level authorization enforced when tools use service principals?
4. What are the specific evaluator loop stop thresholds?
5. What gets logged to MLflow, and what gets redacted?
6. What are the latency and cost budgets per request?
7. How are tool failures surfaced to users?
8. What is the fallback path when MCP servers are unavailable?
9. How is retrieval quality evaluated (recall, citation precision)?
10. What is the concurrency model per Databricks App instance?

## References

- [LangChain Open Deep Research](https://github.com/langchain-ai/open_deep_research) — LangGraph-based deep research with MCP support
- [Databricks MCP Genie Agent Example](https://github.com/IanGagnonDB/databricks-agent-mcp-genie) — Managed MCP server patterns for Genie
- [Company Research Agent](https://github.com/guy-hartstein/company-research-agent) — React + FastAPI architecture with specialized research nodes
- [MLflow Feedback API](https://mlflow.org/docs/latest/api_reference/python_api/mlflow.html#mlflow.log_feedback) — `log_feedback()` and `override_feedback()` APIs
- Architecture review by GPT-5-4 (2026-03-07) — identified gaps in state management, authorization, evidence modeling, loop control, and operational resilience
