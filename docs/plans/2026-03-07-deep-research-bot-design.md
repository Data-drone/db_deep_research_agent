# Deep Research Bot — Design Document

**Date:** 2026-03-07
**Status:** Approved

## Overview

An enterprise deep research agent deployed as a Databricks App. It connects to internal data sources via MCP (Model Context Protocol), performs multi-step research, and produces either conversational answers or structured reports. Users select which tools the agent uses per query via a React UI.

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
│    feedback)                    │
├─────────────────────────────────┤
│       LangGraph Agent           │
│  (Planner → Researcher →       │
│   Evaluator → Compressor →     │
│   Synthesizer)                  │
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

- **React Frontend** — chat interface with streaming, report rendering panel, sidebar with tool checkboxes for selecting MCP servers per query.
- **FastAPI Backend** — serves the React app, handles WebSocket streaming for real-time agent output, REST endpoints for tool configuration and feedback submission.
- **LangGraph Agent** — multi-node research graph orchestrated by LangGraph. Uses Databricks Foundation Model APIs for inference.
- **MCP Client Layer** — connects to Databricks Managed MCP servers (Genie, Vector Search) and custom MCP servers. Connections established at startup via service principal.
- **Databricks Platform** — Genie for structured data queries, Vector Search for unstructured/document retrieval, FMAPIs for LLM inference, MLflow for tracing and feedback.

## Agent Flow

```
[User Query + Selected Tools]
        │
        ▼
    ┌────────┐
    │ Planner │  Breaks query into sub-questions,
    └────┬───┘  decides which tools to use
         │
         ▼
    ┌────────────┐
    │ Researcher  │  Executes tool calls via MCP
    └────┬───────┘  (Genie queries, Vector Search, etc.)
         │
         ▼
    ┌────────────┐
    │ Evaluator   │  Checks: do we have enough info?
    └────┬───────┘  Are there gaps?
         │
    ┌────┴────┐
    │         │
  enough   gaps found
    │         │
    │         ▼
    │    loops back to Planner
    │    with follow-up questions
    │    (max iteration cap)
    │
    ▼
  ┌──────────────┐
  │ Compressor    │  Deduplicates and scores
  └──────┬───────┘  overlapping results for relevance
         │
         ▼
  ┌──────────────┐
  │ Synthesizer   │  Two modes:
  └──────────────┘
     │          │
     ▼          ▼
   Chat       Report
   answer     (structured sections,
              citations, sources)
```

### State

The LangGraph state carries:
- User query
- Research plan (sub-questions)
- Selected tools
- Collected evidence (per tool, per iteration)
- Iteration count
- Output mode (chat vs report)

### Nodes

- **Planner** — uses the LLM to decompose the query and pick relevant tools from the user's selection.
- **Researcher** — calls MCP tools, collects results, adds to evidence state.
- **Evaluator** — decides if another research loop is needed. Has a max iteration cap to prevent runaway loops.
- **Compressor** — deduplicates overlapping results from multiple tools and scores for relevance before synthesis.
- **Synthesizer** — produces either a conversational answer or a structured report with sections, citations, and sources.

All nodes are traced via MLflow autologging.

## MCP Configuration

Tools are configured via a YAML file:

```yaml
managed_servers:
  genie_sales:
    url: "{host}/api/2.0/mcp/genie/{space_id}"
    display_name: "Sales Data (Genie)"
    enabled: true
  vector_search_kb:
    url: "{host}/api/2.0/mcp/vector-search/{index}"
    display_name: "Knowledge Base"
    enabled: true

custom_servers:
  # Future custom MCP servers hosted on Databricks Apps
```

- **Managed MCP servers** — Databricks-hosted (Genie, Vector Search). Auth handled automatically via workspace credentials.
- **Custom MCP servers** — anything built and hosted on Databricks Apps.
- Each server has a `display_name` (shown in the UI) and an `enabled` flag.
- At startup, the app connects to all enabled servers, discovers their tools, and populates the sidebar.

### UI Tool Selection

Users see a sidebar with checkboxes:

```
Available Tools
☑ Sales Data (Genie)
☑ Knowledge Base (Vector Search)
☐ HR Genie (Genie)
```

They toggle which tools the agent uses per query. Connections are pre-established at startup — users only select from available tools.

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

- All agent executions produce full traces (Planner → Researcher → Evaluator → Compressor → Synthesizer).
- User ratings attach to traces via `trace_id`.
- Traces and feedback are viewable in MLflow UI.

### Future Feedback Enhancements

MLflow `log_feedback` supports richer feedback types for later versions:
- Numeric scores (1-5 stars, 0.0-1.0 relevance)
- Multi-dimensional ratings (dict with accuracy, helpfulness, etc.)
- Per-step feedback via `span_id`
- LLM judge automated scoring via `source_type=LLM_JUDGE`
- Feedback override workflow via `mlflow.override_feedback()`

### Databricks Eval Integration

Feedback data is compatible with Mosaic AI Agent Evaluation:
- Compare model versions by rating distribution
- Build eval datasets from highly-rated responses
- Debug low-rated responses by inspecting the full trace

## Serving Architecture

Hybrid approach:
- **FastAPI handles real-time UX** — WebSocket streaming, progress updates as the agent works through its research loop.
- **MLflow logs everything** — traces, tool calls, final outputs, all via the SDK from within the FastAPI backend.
- **User ratings go to MLflow** — React frontend → FastAPI → `mlflow.log_feedback()`.

This gives full Databricks eval integration without the UX constraints of pure Model Serving (timeout limits, clunky streaming for long-running research).

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

## References

- [LangChain Open Deep Research](https://github.com/langchain-ai/open_deep_research) — LangGraph-based deep research with MCP support
- [Databricks MCP Genie Agent Example](https://github.com/IanGagnonDB/databricks-agent-mcp-genie) — Managed MCP server patterns for Genie
- [Company Research Agent](https://github.com/guy-hartstein/company-research-agent) — React + FastAPI architecture with specialized research nodes
- [MLflow Feedback API](https://mlflow.org/docs/latest/api_reference/python_api/mlflow.html#mlflow.log_feedback) — `log_feedback()` and `override_feedback()` APIs
