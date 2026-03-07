# Deep Research Bot Implementation Plan

> **For Claude:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.

**Goal:** Build an enterprise deep research agent deployed as a Databricks App with MCP tool integration, LangGraph orchestration, and MLflow observability.

**Architecture:** React frontend communicates over WebSocket/REST with a FastAPI backend. The backend runs a LangGraph agent graph (Clarifier → Planner → Authorizer → Researcher → Normalizer → Evaluator → Compressor → Synthesizer → Verifier). The agent calls external tools via MCP client connections to Databricks Genie and Vector Search. All traces and feedback are logged to MLflow.

**Tech Stack:** Python 3.11+, LangGraph, FastAPI, databricks-langchain, mcp, mlflow, React 18, Vite, TypeScript

**Design Doc:** `docs/plans/2026-03-07-deep-research-bot-design.md`

---

## Phase 1: Project Scaffolding

### Task 1.1: Python project structure with pyproject.toml

**Files:**
- Create: `pyproject.toml`
- Create: `src/deep_research/__init__.py`
- Create: `src/deep_research/config.py`
- Create: `tests/__init__.py`
- Create: `tests/conftest.py`
- Create: `.gitignore`
- Create: `mcp_config.yaml`

**Step 1: Create .gitignore**

```gitignore
__pycache__/
*.py[cod]
*.egg-info/
dist/
build/
.eggs/
*.egg
.env
.venv/
venv/
node_modules/
ui/dist/
.pytest_cache/
.mypy_cache/
.ruff_cache/
htmlcov/
.coverage
```

**Step 2: Create pyproject.toml**

```toml
[project]
name = "db-deep-research-agent"
version = "0.1.0"
description = "AI-powered deep research agent built on Databricks"
requires-python = ">=3.11"
license = {text = "MIT"}

dependencies = [
    "langgraph>=0.2.0",
    "langchain-core>=0.3.0",
    "databricks-langchain>=0.1.0",
    "mcp>=1.0.0",
    "mlflow>=2.15.0",
    "fastapi>=0.115.0",
    "uvicorn[standard]>=0.30.0",
    "websockets>=12.0",
    "pyyaml>=6.0",
    "pydantic>=2.0",
    "httpx>=0.27.0",
]

[project.optional-dependencies]
dev = [
    "pytest>=8.0",
    "pytest-asyncio>=0.24.0",
    "pytest-cov>=5.0",
    "ruff>=0.5.0",
    "mypy>=1.10",
    "httpx>=0.27.0",
]

[build-system]
requires = ["setuptools>=68.0"]
build-backend = "setuptools.backends._legacy:_Backend"

[tool.setuptools.packages.find]
where = ["src"]

[tool.pytest.ini_options]
testpaths = ["tests"]
asyncio_mode = "auto"

[tool.ruff]
target-version = "py311"
line-length = 100

[tool.ruff.lint]
select = ["E", "F", "I", "W"]
```

**Step 3: Create src/deep_research/__init__.py**

```python
"""Deep Research Agent — enterprise research bot on Databricks."""

__version__ = "0.1.0"
```

**Step 4: Create initial config loader**

```python
# src/deep_research/config.py
"""Configuration loading from YAML and environment."""

from __future__ import annotations

import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import Literal

import yaml


@dataclass(frozen=True)
class MCPServerConfig:
    url: str
    display_name: str
    enabled: bool = True
    risk_tier: Literal["safe", "restricted", "privileged"] = "safe"
    capability: Literal["read", "read_write"] = "read"
    description: str = ""


@dataclass(frozen=True)
class AppConfig:
    databricks_host: str
    databricks_token: str
    llm_endpoint: str
    max_iterations: int = 5
    max_tool_calls: int = 20
    time_cap_seconds: int = 120
    managed_servers: dict[str, MCPServerConfig] = field(default_factory=dict)
    custom_servers: dict[str, MCPServerConfig] = field(default_factory=dict)


def load_mcp_config(path: Path) -> dict[str, MCPServerConfig]:
    """Load MCP server configuration from YAML file."""
    with open(path) as f:
        raw = yaml.safe_load(f)

    servers: dict[str, MCPServerConfig] = {}
    for section in ("managed_servers", "custom_servers"):
        for name, cfg in (raw.get(section) or {}).items():
            if isinstance(cfg, dict):
                servers[name] = MCPServerConfig(**cfg)
    return servers


def load_app_config(mcp_config_path: Path | None = None) -> AppConfig:
    """Load full application config from environment + YAML."""
    mcp_path = mcp_config_path or Path("mcp_config.yaml")
    servers = load_mcp_config(mcp_path) if mcp_path.exists() else {}

    managed = {k: v for k, v in servers.items() if k.startswith("genie") or k.startswith("vector")}
    custom = {k: v for k, v in servers.items() if k not in managed}

    return AppConfig(
        databricks_host=os.environ.get("DATABRICKS_HOST", ""),
        databricks_token=os.environ.get("DATABRICKS_TOKEN", ""),
        llm_endpoint=os.environ.get("LLM_ENDPOINT_NAME", "databricks-meta-llama-3-1-70b-instruct"),
        max_iterations=int(os.environ.get("MAX_ITERATIONS", "5")),
        max_tool_calls=int(os.environ.get("MAX_TOOL_CALLS", "20")),
        time_cap_seconds=int(os.environ.get("TIME_CAP_SECONDS", "120")),
        managed_servers=managed,
        custom_servers=custom,
    )
```

**Step 5: Create mcp_config.yaml**

```yaml
managed_servers:
  genie_sales:
    url: "{host}/api/2.0/mcp/genie/{space_id}"
    display_name: "Sales Data (Genie)"
    enabled: true
    risk_tier: "safe"
    capability: "read"
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

**Step 6: Create tests/conftest.py**

```python
"""Shared test fixtures."""

from pathlib import Path

import pytest

from deep_research.config import AppConfig, MCPServerConfig


@pytest.fixture
def sample_mcp_server():
    return MCPServerConfig(
        url="https://test.databricks.net/api/2.0/mcp/genie/test-space",
        display_name="Test Genie",
        enabled=True,
        risk_tier="safe",
        capability="read",
        description="Test Genie space",
    )


@pytest.fixture
def sample_app_config(sample_mcp_server):
    return AppConfig(
        databricks_host="https://test.databricks.net",
        databricks_token="dapi_test_token",
        llm_endpoint="databricks-meta-llama-3-1-70b-instruct",
        managed_servers={"genie_test": sample_mcp_server},
    )


@pytest.fixture
def fixtures_dir():
    return Path(__file__).parent / "fixtures"
```

**Step 7: Create tests/__init__.py**

```python
```

**Step 8: Run the tests to verify setup**

Run: `cd /workspace/group/repo && pip install -e ".[dev]" && pytest tests/ -v`
Expected: 0 tests collected, no errors

**Step 9: Commit**

```bash
git add -A
git commit -m "feat: project scaffolding with pyproject.toml and config"
```

---

### Task 1.2: Write and test config loading

**Files:**
- Create: `tests/test_config.py`
- Create: `tests/fixtures/test_mcp_config.yaml`

**Step 1: Create test fixture YAML**

```yaml
# tests/fixtures/test_mcp_config.yaml
managed_servers:
  genie_sales:
    url: "https://test.databricks.net/api/2.0/mcp/genie/space-123"
    display_name: "Sales Data (Genie)"
    enabled: true
    risk_tier: "safe"
    capability: "read"
    description: "Query sales data via Genie"
  vector_search_kb:
    url: "https://test.databricks.net/api/2.0/mcp/vector-search/index-456"
    display_name: "Knowledge Base"
    enabled: true
    risk_tier: "safe"
    capability: "read"
    description: "Search internal knowledge base"

custom_servers: {}
```

**Step 2: Write the failing tests**

```python
# tests/test_config.py
"""Tests for configuration loading."""

from pathlib import Path

from deep_research.config import MCPServerConfig, load_mcp_config


def test_load_mcp_config_from_yaml(fixtures_dir):
    servers = load_mcp_config(fixtures_dir / "test_mcp_config.yaml")
    assert "genie_sales" in servers
    assert "vector_search_kb" in servers


def test_mcp_server_config_fields(fixtures_dir):
    servers = load_mcp_config(fixtures_dir / "test_mcp_config.yaml")
    genie = servers["genie_sales"]
    assert genie.display_name == "Sales Data (Genie)"
    assert genie.risk_tier == "safe"
    assert genie.capability == "read"
    assert genie.enabled is True


def test_mcp_server_config_defaults():
    server = MCPServerConfig(url="https://example.com", display_name="Test")
    assert server.enabled is True
    assert server.risk_tier == "safe"
    assert server.capability == "read"
    assert server.description == ""
```

**Step 3: Run tests to verify they pass**

Run: `pytest tests/test_config.py -v`
Expected: 3 PASSED

**Step 4: Commit**

```bash
git add tests/
git commit -m "test: add config loading tests"
```

---

## Phase 2: Data Models and State

### Task 2.1: Core data models (Evidence, Citation, Budget, etc.)

**Files:**
- Create: `src/deep_research/models.py`
- Create: `tests/test_models.py`

**Step 1: Write failing tests for models**

```python
# tests/test_models.py
"""Tests for core data models."""

from deep_research.models import (
    Budget,
    Citation,
    CompressedFindings,
    Evidence,
    EvaluatorDecision,
    SubQuestion,
    ToolCall,
)


def test_evidence_creation():
    e = Evidence(
        evidence_id="ev-001",
        source_id="genie_sales",
        source_type="genie_query",
        title="Q3 Revenue",
        uri="catalog.schema.sales_table",
        snippet="Q3 revenue was $12M, up 15% YoY",
        confidence=0.85,
        freshness="2026-03-01T00:00:00Z",
        tool_that_produced_it="genie_sales",
        iteration=1,
        metadata={"query": "Q3 revenue"},
    )
    assert e.evidence_id == "ev-001"
    assert e.confidence == 0.85


def test_budget_defaults():
    b = Budget()
    assert b.max_iterations == 5
    assert b.max_tool_calls == 20
    assert b.time_cap_seconds == 120


def test_evaluator_decision():
    d = EvaluatorDecision(
        sufficiency_score=0.7,
        missing_facets=["competitor analysis"],
        recommended_actions=["query vector_search for competitors"],
        decision="continue",
        reason="Missing competitor data",
    )
    assert d.decision == "continue"
    assert len(d.missing_facets) == 1


def test_sub_question():
    sq = SubQuestion(
        question="What was Q3 revenue?",
        assigned_tools=["genie_sales"],
        answered=False,
    )
    assert sq.answered is False


def test_citation():
    c = Citation(
        claim="Revenue grew 15% YoY",
        evidence_ids=["ev-001"],
        confidence=0.85,
    )
    assert len(c.evidence_ids) == 1


def test_tool_call():
    tc = ToolCall(
        tool_name="genie_sales",
        server_name="genie_sales",
        input_data={"query": "Q3 revenue"},
        output_data={"result": "$12M"},
        latency_ms=1500,
        success=True,
    )
    assert tc.success is True
    assert tc.latency_ms == 1500


def test_compressed_findings():
    cf = CompressedFindings(
        key_findings=["Revenue grew 15%"],
        open_questions=["What about Q4?"],
        uncertainties=["Competitor data may be stale"],
        contradictions=[],
    )
    assert len(cf.key_findings) == 1
    assert len(cf.contradictions) == 0
```

**Step 2: Run tests to verify they fail**

Run: `pytest tests/test_models.py -v`
Expected: FAIL (cannot import)

**Step 3: Implement models**

```python
# src/deep_research/models.py
"""Core data models for the deep research agent."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Literal


@dataclass
class Evidence:
    evidence_id: str
    source_id: str
    source_type: str
    title: str
    uri: str | None
    snippet: str
    confidence: float
    freshness: str
    tool_that_produced_it: str
    iteration: int
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass
class SubQuestion:
    question: str
    assigned_tools: list[str]
    answered: bool = False


@dataclass
class Budget:
    max_iterations: int = 5
    max_tool_calls: int = 20
    time_cap_seconds: int = 120


@dataclass
class ToolCall:
    tool_name: str
    server_name: str
    input_data: dict[str, Any]
    output_data: dict[str, Any]
    latency_ms: int
    success: bool
    error: str | None = None


@dataclass
class EvaluatorDecision:
    sufficiency_score: float
    missing_facets: list[str]
    recommended_actions: list[str]
    decision: Literal["continue", "stop"]
    reason: str


@dataclass
class Citation:
    claim: str
    evidence_ids: list[str]
    confidence: float


@dataclass
class CompressedFindings:
    key_findings: list[str]
    open_questions: list[str]
    uncertainties: list[str]
    contradictions: list[str]


@dataclass
class VerificationResult:
    all_claims_supported: bool
    unsupported_claims: list[str]
    weakened_claims: list[str]
    contradictions_noted: list[str]
```

**Step 4: Run tests to verify they pass**

Run: `pytest tests/test_models.py -v`
Expected: All PASSED

**Step 5: Commit**

```bash
git add src/deep_research/models.py tests/test_models.py
git commit -m "feat: add core data models (Evidence, Budget, Citation, etc.)"
```

---

### Task 2.2: LangGraph state schema

**Files:**
- Create: `src/deep_research/state.py`
- Create: `tests/test_state.py`

**Step 1: Write failing tests**

```python
# tests/test_state.py
"""Tests for LangGraph state schema."""

from deep_research.state import ResearchState, create_initial_state


def test_create_initial_state():
    state = create_initial_state(
        user_query="What was Q3 revenue?",
        selected_tools=["genie_sales"],
        output_mode="chat",
    )
    assert state["user_query"] == "What was Q3 revenue?"
    assert state["selected_tools"] == ["genie_sales"]
    assert state["output_mode"] == "chat"
    assert state["iteration_count"] == 0
    assert state["evidence"] == []
    assert state["clarification_needed"] is False


def test_initial_state_report_mode():
    state = create_initial_state(
        user_query="Deep analysis of churn",
        selected_tools=["genie_sales", "vector_search_kb"],
        output_mode="report",
    )
    assert state["output_mode"] == "report"
    assert len(state["selected_tools"]) == 2
```

**Step 2: Run tests to verify they fail**

Run: `pytest tests/test_state.py -v`
Expected: FAIL

**Step 3: Implement state**

```python
# src/deep_research/state.py
"""LangGraph state schema for the research agent."""

from __future__ import annotations

from typing import Annotated, Any, Literal, TypedDict

from langgraph.graph import add_messages

from deep_research.models import (
    Budget,
    Citation,
    CompressedFindings,
    Evidence,
    EvaluatorDecision,
    SubQuestion,
    ToolCall,
    VerificationResult,
)


class ResearchState(TypedDict, total=False):
    """Typed state for the research agent graph."""

    # Input
    user_query: str
    selected_tools: list[str]
    output_mode: Literal["chat", "report"]

    # Clarification
    clarified_query: str | None
    clarification_needed: bool

    # Planning
    research_plan: list[SubQuestion]
    tool_assignments: dict[str, list[str]]
    budget: Budget

    # Research
    evidence: list[Evidence]
    tool_call_log: list[ToolCall]
    iteration_count: int

    # Evaluation
    sufficiency_score: float
    missing_facets: list[str]
    stop_reason: str | None
    evaluator_decision: EvaluatorDecision | None

    # Output
    compressed_findings: CompressedFindings | None
    final_output: str
    citations: list[Citation]
    verification_result: VerificationResult | None

    # Tracing
    trace_id: str
    job_id: str

    # Job control
    cancelled: bool


def create_initial_state(
    user_query: str,
    selected_tools: list[str],
    output_mode: Literal["chat", "report"] = "chat",
    job_id: str = "",
    trace_id: str = "",
) -> ResearchState:
    """Create a fresh state for a new research query."""
    return ResearchState(
        user_query=user_query,
        selected_tools=selected_tools,
        output_mode=output_mode,
        clarified_query=None,
        clarification_needed=False,
        research_plan=[],
        tool_assignments={},
        budget=Budget(),
        evidence=[],
        tool_call_log=[],
        iteration_count=0,
        sufficiency_score=0.0,
        missing_facets=[],
        stop_reason=None,
        evaluator_decision=None,
        compressed_findings=None,
        final_output="",
        citations=[],
        verification_result=None,
        trace_id=trace_id,
        job_id=job_id,
        cancelled=False,
    )
```

**Step 4: Run tests to verify they pass**

Run: `pytest tests/test_state.py -v`
Expected: All PASSED

**Step 5: Commit**

```bash
git add src/deep_research/state.py tests/test_state.py
git commit -m "feat: add LangGraph ResearchState schema"
```

---

## Phase 3: MCP Client Layer

### Task 3.1: MCP client manager

**Files:**
- Create: `src/deep_research/mcp_client.py`
- Create: `tests/test_mcp_client.py`

**Step 1: Write failing tests**

```python
# tests/test_mcp_client.py
"""Tests for MCP client manager."""

import pytest

from deep_research.config import MCPServerConfig
from deep_research.mcp_client import MCPClientManager


@pytest.fixture
def mock_server_configs():
    return {
        "genie_sales": MCPServerConfig(
            url="https://test.databricks.net/api/2.0/mcp/genie/space-123",
            display_name="Sales Data (Genie)",
            enabled=True,
            risk_tier="safe",
            capability="read",
            description="Sales queries",
        ),
        "disabled_server": MCPServerConfig(
            url="https://test.databricks.net/api/2.0/mcp/genie/space-999",
            display_name="Disabled",
            enabled=False,
            risk_tier="safe",
            capability="read",
        ),
    }


def test_manager_filters_disabled_servers(mock_server_configs):
    manager = MCPClientManager(mock_server_configs, token="test")
    available = manager.get_available_servers()
    assert "genie_sales" in available
    assert "disabled_server" not in available


def test_manager_get_server_info(mock_server_configs):
    manager = MCPClientManager(mock_server_configs, token="test")
    info = manager.get_server_info("genie_sales")
    assert info.display_name == "Sales Data (Genie)"
    assert info.risk_tier == "safe"


def test_manager_get_servers_by_risk_tier(mock_server_configs):
    manager = MCPClientManager(mock_server_configs, token="test")
    safe = manager.get_servers_by_risk_tier("safe")
    assert "genie_sales" in safe
```

**Step 2: Run tests to verify they fail**

Run: `pytest tests/test_mcp_client.py -v`
Expected: FAIL

**Step 3: Implement MCP client manager**

```python
# src/deep_research/mcp_client.py
"""MCP client manager — connects to and manages MCP servers."""

from __future__ import annotations

import logging
from typing import Any

from deep_research.config import MCPServerConfig

logger = logging.getLogger(__name__)


class MCPClientManager:
    """Manages connections to MCP servers and tool discovery."""

    def __init__(
        self,
        server_configs: dict[str, MCPServerConfig],
        token: str,
    ) -> None:
        self._configs = server_configs
        self._token = token
        self._connections: dict[str, Any] = {}

    def get_available_servers(self) -> dict[str, MCPServerConfig]:
        """Return only enabled server configurations."""
        return {
            name: cfg
            for name, cfg in self._configs.items()
            if cfg.enabled
        }

    def get_server_info(self, server_name: str) -> MCPServerConfig:
        """Get configuration for a specific server."""
        if server_name not in self._configs:
            raise KeyError(f"Unknown MCP server: {server_name}")
        return self._configs[server_name]

    def get_servers_by_risk_tier(
        self, risk_tier: str
    ) -> dict[str, MCPServerConfig]:
        """Return enabled servers matching a risk tier."""
        return {
            name: cfg
            for name, cfg in self.get_available_servers().items()
            if cfg.risk_tier == risk_tier
        }

    async def connect_all(self) -> None:
        """Establish connections to all enabled MCP servers."""
        for name, cfg in self.get_available_servers().items():
            try:
                await self._connect_server(name, cfg)
                logger.info(f"Connected to MCP server: {name} ({cfg.display_name})")
            except Exception:
                logger.exception(f"Failed to connect to MCP server: {name}")

    async def _connect_server(self, name: str, cfg: MCPServerConfig) -> None:
        """Connect to a single MCP server. Placeholder for real MCP SDK integration."""
        # TODO: Replace with actual MCP client SDK connection
        # from mcp import ClientSession
        # session = await ClientSession.connect(cfg.url, token=self._token)
        # self._connections[name] = session
        self._connections[name] = {"url": cfg.url, "status": "connected"}

    async def call_tool(
        self, server_name: str, tool_name: str, arguments: dict[str, Any]
    ) -> dict[str, Any]:
        """Call a tool on a specific MCP server."""
        if server_name not in self._connections:
            raise RuntimeError(f"Not connected to server: {server_name}")

        # TODO: Replace with actual MCP tool call
        # session = self._connections[server_name]
        # result = await session.call_tool(tool_name, arguments)
        # return result
        return {"status": "placeholder", "server": server_name, "tool": tool_name}

    async def disconnect_all(self) -> None:
        """Close all MCP server connections."""
        for name in list(self._connections.keys()):
            try:
                # TODO: session.close()
                del self._connections[name]
                logger.info(f"Disconnected from MCP server: {name}")
            except Exception:
                logger.exception(f"Error disconnecting from: {name}")
```

**Step 4: Run tests to verify they pass**

Run: `pytest tests/test_mcp_client.py -v`
Expected: All PASSED

**Step 5: Commit**

```bash
git add src/deep_research/mcp_client.py tests/test_mcp_client.py
git commit -m "feat: add MCP client manager with server discovery"
```

---

## Phase 4: Agent Graph Nodes

### Task 4.1: Prompts module

**Files:**
- Create: `src/deep_research/prompts.py`

**Step 1: Create prompts file**

```python
# src/deep_research/prompts.py
"""Prompt templates for each agent node."""

CLARIFIER_SYSTEM = """You are a research query clarifier. Analyze the user's query and determine if it is clear enough to research.

If the query is ambiguous or underspecified (missing time range, unclear scope, undefined terms, multiple interpretations), output:
{{"clarification_needed": true, "question": "<your clarifying question>"}}

If the query is clear enough to proceed, output:
{{"clarification_needed": false, "clarified_query": "<the query, possibly lightly rephrased for precision>"}}

Output valid JSON only."""

PLANNER_SYSTEM = """You are a research planner. Given a query and a list of available tools, break the query into sub-questions and assign tools to each.

Available tools: {tools}

Output valid JSON:
{{
  "sub_questions": [
    {{"question": "...", "assigned_tools": ["tool_name"]}}
  ],
  "estimated_iterations": <int>
}}

Be specific. Each sub-question should be answerable by one or two tool calls. Do not create unnecessary sub-questions."""

EVALUATOR_SYSTEM = """You are a research evaluator. Given a research plan and the evidence collected so far, decide if the research is sufficient.

Research plan sub-questions: {sub_questions}
Evidence collected: {evidence_count} items
Current iteration: {iteration} / {max_iterations}

For each sub-question, check if there is at least one evidence item that addresses it.

Output valid JSON:
{{
  "sufficiency_score": <float 0.0-1.0>,
  "missing_facets": ["<what is still unknown>"],
  "recommended_actions": ["<specific follow-up tool calls>"],
  "decision": "continue" | "stop",
  "reason": "<why>"
}}

Stop if: sufficiency_score >= 0.8, or all sub-questions are answered, or we are at max iterations."""

COMPRESSOR_SYSTEM = """You are a research compressor. Given a list of evidence items, produce a structured summary.

Preserve:
- Key findings (the most important facts)
- Open questions (things we still don't know)
- Uncertainties (things we're not sure about)
- Contradictions (conflicting evidence)

Output valid JSON:
{{
  "key_findings": ["..."],
  "open_questions": ["..."],
  "uncertainties": ["..."],
  "contradictions": ["..."]
}}"""

SYNTHESIZER_CHAT_SYSTEM = """You are a research synthesizer. Given compressed findings and evidence, produce a clear, concise answer to the user's query.

- Lead with the direct answer
- Support with evidence
- Note uncertainties and limitations
- Cite sources using [Source: <title>] format"""

SYNTHESIZER_REPORT_SYSTEM = """You are a research report writer. Given compressed findings and evidence, produce a structured report.

Report structure:
1. Executive Summary (2-3 sentences)
2. Key Findings (bulleted, with citations)
3. Detailed Analysis (organized by sub-topic)
4. Limitations & Uncertainties
5. What Would Change This Conclusion
6. Sources

Use [Source: <title>] format for citations. Be thorough but concise."""

VERIFIER_SYSTEM = """You are a citation verifier. Given a draft response and a list of evidence items, check that:

1. Every major claim is supported by at least one evidence item
2. No claims are fabricated or unsupported
3. Contradictions are acknowledged

Output valid JSON:
{{
  "all_claims_supported": <bool>,
  "unsupported_claims": ["<claim text>"],
  "weakened_claims": ["<claim that needs softening>"],
  "contradictions_noted": ["<contradiction>"]
}}

If all claims are supported, return all_claims_supported: true with empty lists."""
```

**Step 2: Commit**

```bash
git add src/deep_research/prompts.py
git commit -m "feat: add prompt templates for all agent nodes"
```

---

### Task 4.2: Individual agent nodes

**Files:**
- Create: `src/deep_research/nodes/__init__.py`
- Create: `src/deep_research/nodes/clarifier.py`
- Create: `src/deep_research/nodes/planner.py`
- Create: `src/deep_research/nodes/authorizer.py`
- Create: `src/deep_research/nodes/researcher.py`
- Create: `src/deep_research/nodes/normalizer.py`
- Create: `src/deep_research/nodes/evaluator.py`
- Create: `src/deep_research/nodes/compressor.py`
- Create: `src/deep_research/nodes/synthesizer.py`
- Create: `src/deep_research/nodes/verifier.py`
- Create: `tests/test_nodes/`

**Note:** Each node is a function that takes `ResearchState` and returns a partial state update. The LLM calls will use `databricks-langchain` ChatDatabricks. For testability, each node accepts a `model` parameter.

This is a large task. Implement each node as a separate step, with tests. Since the nodes depend on LLM calls, tests should mock the LLM. Use `unittest.mock.AsyncMock` for the chat model.

**Step 1: Create nodes/__init__.py**

```python
# src/deep_research/nodes/__init__.py
"""Agent graph nodes."""

from deep_research.nodes.clarifier import clarifier_node
from deep_research.nodes.planner import planner_node
from deep_research.nodes.authorizer import authorizer_node
from deep_research.nodes.researcher import researcher_node
from deep_research.nodes.normalizer import normalizer_node
from deep_research.nodes.evaluator import evaluator_node
from deep_research.nodes.compressor import compressor_node
from deep_research.nodes.synthesizer import synthesizer_node
from deep_research.nodes.verifier import verifier_node

__all__ = [
    "clarifier_node",
    "planner_node",
    "authorizer_node",
    "researcher_node",
    "normalizer_node",
    "evaluator_node",
    "compressor_node",
    "synthesizer_node",
    "verifier_node",
]
```

**Step 2-10:** Implement each node following TDD (write test → verify fail → implement → verify pass → commit). Each node follows this pattern:

```python
# Example: src/deep_research/nodes/clarifier.py
"""Clarifier node — detects ambiguity in user queries."""

from __future__ import annotations

import json
import logging
from typing import Any

from deep_research.prompts import CLARIFIER_SYSTEM
from deep_research.state import ResearchState

logger = logging.getLogger(__name__)


async def clarifier_node(state: ResearchState, *, model: Any) -> dict:
    """Analyze query for ambiguity. Returns state update."""
    response = await model.ainvoke([
        {"role": "system", "content": CLARIFIER_SYSTEM},
        {"role": "user", "content": state["user_query"]},
    ])

    try:
        result = json.loads(response.content)
    except (json.JSONDecodeError, AttributeError):
        # If parsing fails, pass through the original query
        return {
            "clarification_needed": False,
            "clarified_query": state["user_query"],
        }

    return {
        "clarification_needed": result.get("clarification_needed", False),
        "clarified_query": result.get("clarified_query", state["user_query"]),
    }
```

Each remaining node follows the same pattern. Implement one at a time: clarifier → planner → authorizer → researcher → normalizer → evaluator → compressor → synthesizer → verifier.

**Step 11: Commit after all nodes are implemented**

```bash
git add src/deep_research/nodes/ tests/test_nodes/
git commit -m "feat: implement all agent graph nodes with tests"
```

---

### Task 4.3: Wire up the LangGraph graph

**Files:**
- Create: `src/deep_research/graph.py`
- Create: `tests/test_graph.py`

**Step 1: Write failing test**

```python
# tests/test_graph.py
"""Tests for the LangGraph research graph."""

from deep_research.graph import build_research_graph


def test_graph_builds_without_error():
    graph = build_research_graph(model=None, mcp_manager=None)
    assert graph is not None


def test_graph_has_expected_nodes():
    graph = build_research_graph(model=None, mcp_manager=None)
    node_names = set(graph.nodes.keys())
    expected = {
        "clarifier", "planner", "authorizer", "researcher",
        "normalizer", "evaluator", "compressor", "synthesizer", "verifier",
    }
    assert expected.issubset(node_names)
```

**Step 2: Run tests to verify they fail**

Run: `pytest tests/test_graph.py -v`
Expected: FAIL

**Step 3: Implement graph builder**

```python
# src/deep_research/graph.py
"""LangGraph research agent graph builder."""

from __future__ import annotations

from typing import Any

from langgraph.graph import END, StateGraph

from deep_research.mcp_client import MCPClientManager
from deep_research.nodes import (
    authorizer_node,
    clarifier_node,
    compressor_node,
    evaluator_node,
    normalizer_node,
    planner_node,
    researcher_node,
    synthesizer_node,
    verifier_node,
)
from deep_research.state import ResearchState


def _should_continue(state: ResearchState) -> str:
    """Routing function after evaluator: continue research or compress."""
    decision = state.get("evaluator_decision")
    if decision and decision.decision == "continue":
        budget = state.get("budget")
        iteration = state.get("iteration_count", 0)
        if budget and iteration >= budget.max_iterations:
            return "compressor"
        return "planner"
    return "compressor"


def build_research_graph(
    model: Any,
    mcp_manager: MCPClientManager | None,
) -> StateGraph:
    """Build the full research agent graph."""
    graph = StateGraph(ResearchState)

    # Add nodes — each wraps the node function with injected dependencies
    graph.add_node("clarifier", lambda s: clarifier_node(s, model=model))
    graph.add_node("planner", lambda s: planner_node(s, model=model))
    graph.add_node("authorizer", lambda s: authorizer_node(s, model=model))
    graph.add_node("researcher", lambda s: researcher_node(s, model=model, mcp_manager=mcp_manager))
    graph.add_node("normalizer", lambda s: normalizer_node(s, model=model))
    graph.add_node("evaluator", lambda s: evaluator_node(s, model=model))
    graph.add_node("compressor", lambda s: compressor_node(s, model=model))
    graph.add_node("synthesizer", lambda s: synthesizer_node(s, model=model))
    graph.add_node("verifier", lambda s: verifier_node(s, model=model))

    # Define edges
    graph.set_entry_point("clarifier")
    graph.add_edge("clarifier", "planner")
    graph.add_edge("planner", "authorizer")
    graph.add_edge("authorizer", "researcher")
    graph.add_edge("researcher", "normalizer")
    graph.add_edge("normalizer", "evaluator")
    graph.add_conditional_edges("evaluator", _should_continue, {
        "planner": "planner",
        "compressor": "compressor",
    })
    graph.add_edge("compressor", "synthesizer")
    graph.add_edge("synthesizer", "verifier")
    graph.add_edge("verifier", END)

    return graph.compile()
```

**Step 4: Run tests to verify they pass**

Run: `pytest tests/test_graph.py -v`
Expected: All PASSED

**Step 5: Commit**

```bash
git add src/deep_research/graph.py tests/test_graph.py
git commit -m "feat: wire up LangGraph research agent graph"
```

---

## Phase 5: FastAPI Backend

### Task 5.1: Core FastAPI app with health check

**Files:**
- Create: `src/deep_research/api/__init__.py`
- Create: `src/deep_research/api/app.py`
- Create: `tests/test_api.py`

**Step 1: Write failing test**

```python
# tests/test_api.py
"""Tests for FastAPI backend."""

import pytest
from httpx import ASGITransport, AsyncClient

from deep_research.api.app import create_app


@pytest.fixture
def app():
    return create_app()


@pytest.fixture
async def client(app):
    transport = ASGITransport(app=app)
    async with AsyncClient(transport=transport, base_url="http://test") as c:
        yield c


async def test_health_check(client):
    response = await client.get("/health")
    assert response.status_code == 200
    assert response.json()["status"] == "ok"


async def test_list_tools(client):
    response = await client.get("/api/tools")
    assert response.status_code == 200
    assert isinstance(response.json(), list)
```

**Step 2: Run tests to verify they fail**

Run: `pytest tests/test_api.py -v`
Expected: FAIL

**Step 3: Implement FastAPI app**

```python
# src/deep_research/api/__init__.py
```

```python
# src/deep_research/api/app.py
"""FastAPI application factory."""

from __future__ import annotations

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware


def create_app() -> FastAPI:
    app = FastAPI(title="Deep Research Agent", version="0.1.0")

    app.add_middleware(
        CORSMiddleware,
        allow_origins=["*"],
        allow_methods=["*"],
        allow_headers=["*"],
    )

    @app.get("/health")
    async def health():
        return {"status": "ok"}

    @app.get("/api/tools")
    async def list_tools():
        # TODO: Return actual MCP server tool list
        return []

    return app
```

**Step 4: Run tests to verify they pass**

Run: `pytest tests/test_api.py -v`
Expected: All PASSED

**Step 5: Commit**

```bash
git add src/deep_research/api/ tests/test_api.py
git commit -m "feat: add FastAPI app with health check and tools endpoint"
```

---

### Task 5.2: Research endpoint and job management

**Files:**
- Modify: `src/deep_research/api/app.py`
- Create: `src/deep_research/api/jobs.py`
- Create: `tests/test_jobs.py`

This task adds:
- `POST /api/research` — submit a research query, get back a job_id
- `GET /api/research/{job_id}` — poll for status/result
- `DELETE /api/research/{job_id}` — cancel a job
- In-memory job store (upgrade to persistent later)

**Step 1: Write failing tests for job manager**

```python
# tests/test_jobs.py
"""Tests for job management."""

import pytest

from deep_research.api.jobs import JobManager, JobStatus


@pytest.fixture
def job_manager():
    return JobManager()


def test_create_job(job_manager):
    job_id = job_manager.create_job(query="What is revenue?", tools=["genie"])
    assert job_id is not None
    status = job_manager.get_status(job_id)
    assert status.state == "pending"


def test_cancel_job(job_manager):
    job_id = job_manager.create_job(query="test", tools=[])
    job_manager.cancel_job(job_id)
    status = job_manager.get_status(job_id)
    assert status.state == "cancelled"


def test_unknown_job_raises(job_manager):
    with pytest.raises(KeyError):
        job_manager.get_status("nonexistent")
```

**Step 2: Implement job manager, wire into app, run tests, commit.**

```bash
git add src/deep_research/api/jobs.py tests/test_jobs.py
git commit -m "feat: add job manager for research queries"
```

---

### Task 5.3: Feedback endpoint

**Files:**
- Modify: `src/deep_research/api/app.py`
- Create: `tests/test_feedback.py`

Adds `POST /api/feedback` endpoint that accepts `{query_id, rating, comment}` and logs to MLflow.

**Step 1: Write tests → Step 2: Implement → Step 3: Run → Step 4: Commit**

```bash
git commit -m "feat: add feedback endpoint with MLflow logging"
```

---

## Phase 6: React Frontend

### Task 6.1: Vite + React project setup

**Files:**
- Create: `ui/` directory with Vite + React + TypeScript scaffolding

**Step 1: Scaffold React project**

Run:
```bash
cd /workspace/group/repo
npm create vite@latest ui -- --template react-ts
cd ui && npm install
```

**Step 2: Install dependencies**

Run:
```bash
cd ui
npm install axios
npm install -D @types/node
```

**Step 3: Verify it builds**

Run: `cd ui && npm run build`
Expected: Build succeeds

**Step 4: Commit**

```bash
git add ui/
git commit -m "feat: scaffold React + Vite + TypeScript frontend"
```

---

### Task 6.2: Chat interface component

**Files:**
- Create: `ui/src/components/ChatPanel.tsx`
- Create: `ui/src/components/MessageBubble.tsx`
- Create: `ui/src/hooks/useResearch.ts`

Implements:
- Chat message list with user/assistant bubbles
- Input box with send button
- WebSocket connection for streaming responses
- Thumbs up/down on assistant messages

**Step 1: Build components → Step 2: Wire to API → Step 3: Verify → Step 4: Commit**

```bash
git commit -m "feat: add chat interface with streaming support"
```

---

### Task 6.3: Tool selector sidebar

**Files:**
- Create: `ui/src/components/ToolSelector.tsx`

Implements:
- Fetches available tools from `GET /api/tools`
- Checkboxes with display names
- Auto-suggested tools marked with star
- Selected tools passed to research endpoint

```bash
git commit -m "feat: add tool selector sidebar"
```

---

### Task 6.4: Report viewer panel

**Files:**
- Create: `ui/src/components/ReportPanel.tsx`

Implements:
- Renders structured reports from markdown
- Collapsible sections
- Citation highlighting

```bash
git commit -m "feat: add report viewer panel"
```

---

### Task 6.5: Progress indicators

**Files:**
- Modify: `ui/src/components/ChatPanel.tsx`

Implements:
- Shows current agent stage (Clarifying, Planning, Researching, etc.)
- Iteration counter (e.g. "Analyzing results (2/5)")
- Cancel button

```bash
git commit -m "feat: add research progress indicators"
```

---

## Phase 7: Observability

### Task 7.1: MLflow tracing integration

**Files:**
- Create: `src/deep_research/tracing.py`
- Create: `tests/test_tracing.py`

Implements:
- MLflow autologging for LangGraph
- Custom span creation per node
- Trace ID propagation through state

```bash
git commit -m "feat: add MLflow tracing for agent graph"
```

---

### Task 7.2: Feedback logging to MLflow

**Files:**
- Modify: `src/deep_research/api/app.py` (feedback endpoint)
- Create: `src/deep_research/feedback.py`

Implements:
- `log_user_feedback(trace_id, rating, comment, user_id)`
- Uses `mlflow.log_feedback()` with `AssessmentSourceType.HUMAN`

```bash
git commit -m "feat: add user feedback logging to MLflow"
```

---

## Phase 8: Integration and Deployment

### Task 8.1: End-to-end integration

**Files:**
- Create: `src/deep_research/main.py`
- Modify: `src/deep_research/api/app.py`

Implements:
- App startup: load config, connect MCP servers, build graph
- Wire graph execution to research endpoint
- Serve React build from FastAPI

```bash
git commit -m "feat: wire up end-to-end integration"
```

---

### Task 8.2: Databricks App configuration

**Files:**
- Create: `app.yaml` (Databricks App config)
- Create: `Dockerfile` (optional, for containerized deployment)
- Modify: `README.md` (deployment instructions)

**Step 1: Create app.yaml**

```yaml
# app.yaml — Databricks App configuration
command:
  - "uvicorn"
  - "deep_research.api.app:create_app"
  - "--host"
  - "0.0.0.0"
  - "--port"
  - "8000"
  - "--factory"
env:
  - name: DATABRICKS_HOST
    valueFrom: secret
  - name: DATABRICKS_TOKEN
    valueFrom: secret
  - name: LLM_ENDPOINT_NAME
    value: "databricks-meta-llama-3-1-70b-instruct"
```

**Step 2: Commit**

```bash
git add app.yaml
git commit -m "feat: add Databricks App deployment config"
```

---

### Task 8.3: Final README update

**Files:**
- Modify: `README.md`

Update with:
- Full setup instructions (backend + frontend)
- Environment variable reference
- MCP configuration guide
- Deployment instructions
- Development workflow

```bash
git commit -m "docs: update README with full setup and deployment guide"
```

---

## Phase 9: Integration & End-to-End Testing

Unit tests on individual functions don't tell you if the system actually works for a user. This phase focuses on testing the real user experience.

### Task 9.1: Mock MCP server for testing

**Files:**
- Create: `tests/mock_mcp_server.py`
- Create: `tests/fixtures/mock_genie_responses.json`
- Create: `tests/fixtures/mock_vector_search_responses.json`

Build a fake MCP server that returns canned responses. This lets us test the full pipeline without needing a live Databricks workspace.

**Step 1: Create mock MCP server**

```python
# tests/mock_mcp_server.py
"""Mock MCP server for integration testing.

Returns canned responses for Genie and Vector Search queries.
Implements the same interface as a real MCP server so the agent
graph can run end-to-end without Databricks access.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

FIXTURES = Path(__file__).parent / "fixtures"


class MockMCPServer:
    """Simulates an MCP server with pre-loaded responses."""

    def __init__(self, server_name: str, responses_file: str) -> None:
        self.server_name = server_name
        self._responses = self._load_responses(responses_file)

    def _load_responses(self, filename: str) -> dict[str, Any]:
        path = FIXTURES / filename
        if path.exists():
            with open(path) as f:
                return json.load(f)
        return {}

    async def call_tool(self, tool_name: str, arguments: dict) -> dict[str, Any]:
        """Return canned response matching the query, or a default."""
        query = arguments.get("query", "")
        for pattern, response in self._responses.items():
            if pattern.lower() in query.lower():
                return response
        return {
            "result": f"No mock data for query: {query}",
            "source": self.server_name,
        }


class MockMCPClientManager:
    """Drop-in replacement for MCPClientManager in tests."""

    def __init__(self) -> None:
        self._servers = {
            "genie_sales": MockMCPServer("genie_sales", "mock_genie_responses.json"),
            "vector_search_kb": MockMCPServer("vector_search_kb", "mock_vector_search_responses.json"),
        }

    def get_available_servers(self):
        from deep_research.config import MCPServerConfig
        return {
            name: MCPServerConfig(
                url=f"mock://{name}",
                display_name=name.replace("_", " ").title(),
                enabled=True,
                risk_tier="safe",
                capability="read",
            )
            for name in self._servers
        }

    async def call_tool(self, server_name: str, tool_name: str, arguments: dict) -> dict:
        server = self._servers.get(server_name)
        if not server:
            raise RuntimeError(f"No mock server: {server_name}")
        return await server.call_tool(tool_name, arguments)
```

**Step 2: Create mock response fixtures**

```json
// tests/fixtures/mock_genie_responses.json
{
    "revenue": {
        "result": "Q3 2025 revenue was $12.4M, representing 15% YoY growth. Product revenue $8.1M, services revenue $4.3M.",
        "source": "genie_sales",
        "table": "catalog.sales.quarterly_revenue"
    },
    "churn": {
        "result": "Customer churn rate for Q3 was 4.2%, down from 5.1% in Q2. Enterprise churn was 1.8%, SMB churn was 7.6%.",
        "source": "genie_sales",
        "table": "catalog.sales.customer_metrics"
    },
    "top customers": {
        "result": "Top 5 customers by ARR: Acme Corp ($1.2M), Beta Inc ($890K), Gamma Ltd ($720K), Delta Co ($650K), Epsilon ($540K).",
        "source": "genie_sales",
        "table": "catalog.sales.customer_arr"
    }
}
```

```json
// tests/fixtures/mock_vector_search_responses.json
{
    "revenue growth": {
        "result": "According to the Q3 board deck, revenue growth was driven primarily by expansion in existing accounts (62%) and new logo acquisition (38%). The sales team exceeded quota by 12%.",
        "source": "vector_search_kb",
        "document": "Q3-2025-board-deck.pdf"
    },
    "churn analysis": {
        "result": "The churn analysis report from September indicates that primary churn drivers were: price sensitivity (35%), feature gaps (28%), poor onboarding (22%), and competitive displacement (15%).",
        "source": "vector_search_kb",
        "document": "churn-analysis-sept-2025.pdf"
    }
}
```

**Step 3: Commit**

```bash
git add tests/mock_mcp_server.py tests/fixtures/
git commit -m "test: add mock MCP server and response fixtures"
```

---

### Task 9.2: Full pipeline integration test

**Files:**
- Create: `tests/test_integration.py`

Tests the complete agent graph end-to-end: user submits a query → graph runs through all nodes → returns a synthesized answer with citations.

**Step 1: Write integration test**

```python
# tests/test_integration.py
"""End-to-end integration tests for the research pipeline.

These tests run the full agent graph with mock MCP servers and a
mock LLM to verify the pipeline works as a user would experience it.
"""

import pytest
from unittest.mock import AsyncMock, MagicMock

from deep_research.graph import build_research_graph
from deep_research.state import create_initial_state
from tests.mock_mcp_server import MockMCPClientManager


def make_mock_model(responses: list[str]):
    """Create a mock LLM that returns canned responses in sequence."""
    model = AsyncMock()
    call_count = 0

    async def fake_invoke(messages):
        nonlocal call_count
        idx = min(call_count, len(responses) - 1)
        call_count += 1
        result = MagicMock()
        result.content = responses[idx]
        return result

    model.ainvoke = fake_invoke
    return model


# Canned LLM responses for each node in sequence:
# clarifier → planner → authorizer (passthrough) → researcher →
# normalizer → evaluator → compressor → synthesizer → verifier
SIMPLE_QUERY_LLM_RESPONSES = [
    # Clarifier: query is clear
    '{"clarification_needed": false, "clarified_query": "What was Q3 revenue?"}',
    # Planner: one sub-question
    '{"sub_questions": [{"question": "What was Q3 revenue?", "assigned_tools": ["genie_sales"]}], "estimated_iterations": 1}',
    # Evaluator: sufficient
    '{"sufficiency_score": 0.9, "missing_facets": [], "recommended_actions": [], "decision": "stop", "reason": "Query answered"}',
    # Compressor
    '{"key_findings": ["Q3 revenue was $12.4M, up 15% YoY"], "open_questions": [], "uncertainties": [], "contradictions": []}',
    # Synthesizer
    'Q3 2025 revenue was $12.4M, representing 15% year-over-year growth. [Source: Sales Data (Genie)]',
    # Verifier
    '{"all_claims_supported": true, "unsupported_claims": [], "weakened_claims": [], "contradictions_noted": []}',
]


@pytest.mark.asyncio
async def test_simple_query_end_to_end():
    """User asks a simple question, gets a direct answer."""
    mock_model = make_mock_model(SIMPLE_QUERY_LLM_RESPONSES)
    mock_mcp = MockMCPClientManager()

    graph = build_research_graph(model=mock_model, mcp_manager=mock_mcp)

    initial_state = create_initial_state(
        user_query="What was Q3 revenue?",
        selected_tools=["genie_sales"],
        output_mode="chat",
    )

    result = await graph.ainvoke(initial_state)

    # User should get an answer
    assert result["final_output"] != ""
    assert "revenue" in result["final_output"].lower() or "12" in result["final_output"]
    # Should have evidence
    assert len(result.get("evidence", [])) > 0 or result["final_output"] != ""
    # Verification should pass
    vr = result.get("verification_result")
    if vr:
        assert vr.all_claims_supported is True


MULTI_TOOL_LLM_RESPONSES = [
    # Clarifier
    '{"clarification_needed": false, "clarified_query": "Analyze customer churn and its drivers"}',
    # Planner: two sub-questions, two tools
    '{"sub_questions": [{"question": "What is the current churn rate?", "assigned_tools": ["genie_sales"]}, {"question": "What are the churn drivers?", "assigned_tools": ["vector_search_kb"]}], "estimated_iterations": 1}',
    # Evaluator
    '{"sufficiency_score": 0.85, "missing_facets": [], "recommended_actions": [], "decision": "stop", "reason": "Both questions answered"}',
    # Compressor
    '{"key_findings": ["Churn rate is 4.2%", "Main drivers: price sensitivity, feature gaps"], "open_questions": [], "uncertainties": ["SMB churn much higher than enterprise"], "contradictions": []}',
    # Synthesizer
    '## Churn Analysis\\n\\nCustomer churn rate is 4.2% in Q3. Primary drivers are price sensitivity (35%) and feature gaps (28%). [Source: Sales Data] [Source: Knowledge Base]\\n\\n### Limitations\\nSMB churn (7.6%) is significantly higher than enterprise (1.8%).',
    # Verifier
    '{"all_claims_supported": true, "unsupported_claims": [], "weakened_claims": [], "contradictions_noted": []}',
]


@pytest.mark.asyncio
async def test_multi_tool_query():
    """User asks a complex question requiring multiple tools."""
    mock_model = make_mock_model(MULTI_TOOL_LLM_RESPONSES)
    mock_mcp = MockMCPClientManager()

    graph = build_research_graph(model=mock_model, mcp_manager=mock_mcp)

    initial_state = create_initial_state(
        user_query="Analyze customer churn and its drivers",
        selected_tools=["genie_sales", "vector_search_kb"],
        output_mode="report",
    )

    result = await graph.ainvoke(initial_state)

    assert result["final_output"] != ""
    assert result["output_mode"] == "report"


@pytest.mark.asyncio
async def test_iterative_research_loop():
    """Agent loops back when evaluator says evidence is insufficient."""
    responses = [
        # Clarifier
        '{"clarification_needed": false, "clarified_query": "Compare revenue to competitors"}',
        # Planner (iteration 1)
        '{"sub_questions": [{"question": "What is our revenue?", "assigned_tools": ["genie_sales"]}, {"question": "What are competitor revenues?", "assigned_tools": ["vector_search_kb"]}], "estimated_iterations": 2}',
        # Evaluator (iteration 1): needs more
        '{"sufficiency_score": 0.4, "missing_facets": ["competitor revenue data"], "recommended_actions": ["search for competitor analysis"], "decision": "continue", "reason": "Missing competitor data"}',
        # Planner (iteration 2)
        '{"sub_questions": [{"question": "Competitor revenue comparison", "assigned_tools": ["vector_search_kb"]}], "estimated_iterations": 1}',
        # Evaluator (iteration 2): sufficient
        '{"sufficiency_score": 0.8, "missing_facets": [], "recommended_actions": [], "decision": "stop", "reason": "Sufficient data"}',
        # Compressor
        '{"key_findings": ["Our revenue is $12.4M", "Competitor A is ~$15M based on estimates"], "open_questions": ["Exact competitor figures unavailable"], "uncertainties": ["Competitor data is estimated"], "contradictions": []}',
        # Synthesizer
        'Our Q3 revenue was $12.4M. Based on available data, our primary competitor is estimated at ~$15M. [Source: Sales Data] [Source: Knowledge Base]\\n\\nNote: Competitor figures are estimates.',
        # Verifier
        '{"all_claims_supported": true, "unsupported_claims": [], "weakened_claims": ["competitor revenue is estimated"], "contradictions_noted": []}',
    ]

    mock_model = make_mock_model(responses)
    mock_mcp = MockMCPClientManager()

    graph = build_research_graph(model=mock_model, mcp_manager=mock_mcp)

    initial_state = create_initial_state(
        user_query="Compare revenue to competitors",
        selected_tools=["genie_sales", "vector_search_kb"],
        output_mode="chat",
    )

    result = await graph.ainvoke(initial_state)

    assert result["final_output"] != ""
    # Should have done more than 1 iteration
    assert result.get("iteration_count", 0) >= 1
```

**Step 2: Run integration tests**

Run: `pytest tests/test_integration.py -v`
Expected: All PASSED (after agent graph is fully wired)

**Step 3: Commit**

```bash
git add tests/test_integration.py
git commit -m "test: add end-to-end integration tests for research pipeline"
```

---

### Task 9.3: API integration test (user's HTTP perspective)

**Files:**
- Create: `tests/test_api_integration.py`

Tests the full flow from the user's perspective: HTTP request → backend → agent → response.

**Step 1: Write API integration test**

```python
# tests/test_api_integration.py
"""API-level integration tests — tests from the user's HTTP perspective.

These simulate what happens when a user submits a query through the UI.
"""

import pytest
from httpx import ASGITransport, AsyncClient

from deep_research.api.app import create_app


@pytest.fixture
def app():
    """Create app with mock dependencies."""
    return create_app(use_mocks=True)  # create_app accepts flag for test mode


@pytest.fixture
async def client(app):
    transport = ASGITransport(app=app)
    async with AsyncClient(transport=transport, base_url="http://test") as c:
        yield c


class TestUserResearchFlow:
    """Tests the full user journey: submit query → poll → get result."""

    async def test_submit_research_query(self, client):
        """User submits a research query and gets a job ID back."""
        response = await client.post("/api/research", json={
            "query": "What was Q3 revenue?",
            "tools": ["genie_sales"],
            "output_mode": "chat",
        })
        assert response.status_code == 200
        data = response.json()
        assert "job_id" in data
        assert data["status"] == "pending" or data["status"] == "running"

    async def test_poll_for_result(self, client):
        """User polls until research is complete."""
        # Submit
        submit = await client.post("/api/research", json={
            "query": "What was Q3 revenue?",
            "tools": ["genie_sales"],
            "output_mode": "chat",
        })
        job_id = submit.json()["job_id"]

        # Poll (in real use this would loop with delays)
        result = await client.get(f"/api/research/{job_id}")
        assert result.status_code == 200
        data = result.json()
        assert data["status"] in ("pending", "running", "completed")

    async def test_cancel_research(self, client):
        """User cancels an in-progress research query."""
        submit = await client.post("/api/research", json={
            "query": "Deep analysis of everything",
            "tools": ["genie_sales", "vector_search_kb"],
            "output_mode": "report",
        })
        job_id = submit.json()["job_id"]

        cancel = await client.delete(f"/api/research/{job_id}")
        assert cancel.status_code == 200

        status = await client.get(f"/api/research/{job_id}")
        assert status.json()["status"] == "cancelled"

    async def test_submit_feedback(self, client):
        """User submits thumbs up/down on a response."""
        response = await client.post("/api/feedback", json={
            "query_id": "test-query-123",
            "rating": "thumbs_up",
            "comment": "Great answer!",
        })
        assert response.status_code == 200

    async def test_list_available_tools(self, client):
        """User loads the tool selector sidebar."""
        response = await client.get("/api/tools")
        assert response.status_code == 200
        tools = response.json()
        assert isinstance(tools, list)
        # In mock mode, should have test tools available
        if len(tools) > 0:
            assert "display_name" in tools[0]
            assert "risk_tier" in tools[0]


class TestEdgeCases:
    """Tests error handling and edge cases a user might hit."""

    async def test_empty_query_rejected(self, client):
        """User submits an empty query."""
        response = await client.post("/api/research", json={
            "query": "",
            "tools": ["genie_sales"],
        })
        assert response.status_code == 422 or response.status_code == 400

    async def test_no_tools_selected(self, client):
        """User submits query with no tools selected."""
        response = await client.post("/api/research", json={
            "query": "What is revenue?",
            "tools": [],
        })
        # Should either reject or auto-suggest tools
        assert response.status_code in (200, 400, 422)

    async def test_unknown_job_id(self, client):
        """User polls for a nonexistent job."""
        response = await client.get("/api/research/nonexistent-job-id")
        assert response.status_code == 404

    async def test_feedback_invalid_rating(self, client):
        """User submits invalid rating value."""
        response = await client.post("/api/feedback", json={
            "query_id": "test",
            "rating": "invalid_value",
        })
        assert response.status_code == 422 or response.status_code == 400
```

**Step 2: Run API integration tests**

Run: `pytest tests/test_api_integration.py -v`
Expected: All PASSED

**Step 3: Commit**

```bash
git add tests/test_api_integration.py
git commit -m "test: add API integration tests for user research flow"
```

---

### Task 9.4: Notebook-based manual test harness

**Files:**
- Create: `notebooks/test_harness.py`

A simple script (runnable as a Databricks notebook or locally) for manual testing against a real Databricks workspace. This is for when you want to test with real Genie spaces and Vector Search indexes.

**Step 1: Create test harness**

```python
# notebooks/test_harness.py
"""Manual test harness for testing against real Databricks workspace.

Usage:
    # Set environment variables first:
    # DATABRICKS_HOST, DATABRICKS_TOKEN, GENIE_SPACE_ID, etc.

    python notebooks/test_harness.py "What was Q3 revenue?"
    python notebooks/test_harness.py "Analyze customer churn" --mode report
    python notebooks/test_harness.py "Compare our metrics" --tools genie_sales,vector_search_kb
"""

import argparse
import asyncio
import json
import sys
import time

from deep_research.config import load_app_config
from deep_research.graph import build_research_graph
from deep_research.mcp_client import MCPClientManager
from deep_research.state import create_initial_state


async def run_research(query: str, tools: list[str], mode: str) -> None:
    """Run a research query and print results."""
    config = load_app_config()

    # Initialize MCP client
    all_servers = {**config.managed_servers, **config.custom_servers}
    mcp_manager = MCPClientManager(all_servers, token=config.databricks_token)
    await mcp_manager.connect_all()

    # Build graph
    # TODO: Initialize real ChatDatabricks model
    # from langchain_databricks import ChatDatabricks
    # model = ChatDatabricks(endpoint=config.llm_endpoint)
    model = None  # Replace with real model

    graph = build_research_graph(model=model, mcp_manager=mcp_manager)

    # Run
    state = create_initial_state(
        user_query=query,
        selected_tools=tools,
        output_mode=mode,
    )

    print(f"\n{'='*60}")
    print(f"Query: {query}")
    print(f"Tools: {tools}")
    print(f"Mode: {mode}")
    print(f"{'='*60}\n")

    start = time.time()
    result = await graph.ainvoke(state)
    elapsed = time.time() - start

    print(f"\n{'='*60}")
    print(f"RESULT ({elapsed:.1f}s)")
    print(f"{'='*60}\n")
    print(result.get("final_output", "No output"))

    print(f"\n--- Metadata ---")
    print(f"Iterations: {result.get('iteration_count', 0)}")
    print(f"Evidence items: {len(result.get('evidence', []))}")
    print(f"Citations: {len(result.get('citations', []))}")

    vr = result.get("verification_result")
    if vr:
        print(f"All claims supported: {vr.all_claims_supported}")
        if vr.unsupported_claims:
            print(f"Unsupported claims: {vr.unsupported_claims}")

    await mcp_manager.disconnect_all()


def main():
    parser = argparse.ArgumentParser(description="Test the research agent")
    parser.add_argument("query", help="Research query")
    parser.add_argument("--tools", default="genie_sales", help="Comma-separated tool names")
    parser.add_argument("--mode", default="chat", choices=["chat", "report"])
    args = parser.parse_args()

    tools = [t.strip() for t in args.tools.split(",")]
    asyncio.run(run_research(args.query, tools, args.mode))


if __name__ == "__main__":
    main()
```

**Step 2: Commit**

```bash
git add notebooks/
git commit -m "test: add manual test harness for real workspace testing"
```

---

## Testing Strategy Summary

| Level | What it tests | When to run | Dependencies |
|-------|--------------|-------------|--------------|
| Unit tests (`tests/test_*.py`) | Individual functions, models, config | Every commit | None |
| Integration tests (`tests/test_integration.py`) | Full agent graph pipeline | Every commit | Mock MCP + Mock LLM |
| API integration (`tests/test_api_integration.py`) | User HTTP journey | Every commit | Mock dependencies |
| Manual harness (`notebooks/test_harness.py`) | Real Databricks workspace | Before deploy | Live Databricks workspace |

Run all automated tests: `pytest tests/ -v`
Run only integration: `pytest tests/test_integration.py tests/test_api_integration.py -v`

---

## Summary

| Phase | Tasks | What it delivers |
|-------|-------|-----------------|
| 1. Scaffolding | 1.1-1.2 | Project structure, config loading |
| 2. Data Models | 2.1-2.2 | Evidence, State, Budget, Citation models |
| 3. MCP Client | 3.1 | MCP server manager and tool discovery |
| 4. Agent Graph | 4.1-4.3 | All 9 nodes + LangGraph wiring |
| 5. Backend | 5.1-5.3 | FastAPI with research, jobs, feedback |
| 6. Frontend | 6.1-6.5 | React chat, tools, reports, progress |
| 7. Observability | 7.1-7.2 | MLflow tracing + feedback |
| 8. Integration | 8.1-8.3 | End-to-end wiring + deployment |
| 9. Testing | 9.1-9.4 | Mock MCP, integration tests, API tests, manual harness |

Each phase builds on the previous. Phases 1-4 are the core backend. Phase 5 exposes it. Phase 6 puts a UI on it. Phases 7-8 make it production-ready. Phase 9 ensures it actually works from a user's perspective.
