# db_deep_research_agent

AI-powered deep research agent built on Databricks. Connects to enterprise data sources via MCP (Model Context Protocol), performs multi-step research, and produces comprehensive reports.

## Stack

- **Platform:** Databricks (deployed as a Databricks App)
- **Language:** Python (backend), TypeScript/React (frontend)
- **Agent Framework:** LangGraph
- **Data Sources:** Databricks Genie Agents, Vector Search (via MCP)
- **LLM:** Databricks Foundation Model APIs
- **Observability:** MLflow tracing + feedback

## Docs

- [Design Document](docs/plans/2026-03-07-deep-research-bot-design.md) — architecture, agent flow, MCP configuration, feedback system
- [Implementation Plan](docs/plans/2026-03-07-deep-research-bot-implementation.md) — phased build plan with testing strategy

## Setup

```bash
# Clone
git clone https://github.com/Data-drone/db_deep_research_agent.git
cd db_deep_research_agent

# Install dependencies
pip install -r requirements.txt
```

## License

MIT
