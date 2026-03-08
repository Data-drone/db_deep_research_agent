# Phase 0: Eval Baseline + Dual-Model Config — Implementation Plan

> **For Claude:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.

**Goal:** Establish a measurable baseline with MLflow Evaluate before changing the research loop, and wire up dual-model config (Sonnet 4.6 worker + GPT-5.4 critic).

**Architecture:** Add `critic_llm_endpoint` to AppConfig, create a second ChatDatabricks instance, thread it through `build_research_graph()` into critic-role nodes (evaluator, verifier — currently both use the worker model). Build an eval harness that runs sample queries and scores results with `mlflow.evaluate()`.

**Tech Stack:** Python 3.11, LangGraph, databricks-langchain (ChatDatabricks), MLflow, pytest, pytest-asyncio

**Working directory:** `/workspace/group/repo/.worktrees/implementation`

---

### Task 1: Add `critic_llm_endpoint` to AppConfig

**Files:**
- Modify: `src/deep_research/config.py:26-35` (AppConfig dataclass)
- Modify: `src/deep_research/config.py:102-111` (load_app_config return)
- Test: `tests/test_config.py`

**Step 1: Write the failing test**

Add to `tests/test_config.py`:

```python
def test_load_app_config_has_critic_endpoint(tmp_path, monkeypatch, fixtures_dir):
    monkeypatch.setenv("DATABRICKS_HOST", "https://test.databricks.net")
    monkeypatch.setenv("DATABRICKS_TOKEN", "dapi_test")
    monkeypatch.setenv("CRITIC_LLM_ENDPOINT", "databricks-gpt-5-4")
    config = load_app_config(mcp_config_path=fixtures_dir / "test_mcp_config.yaml")
    assert config.critic_llm_endpoint == "databricks-gpt-5-4"


def test_critic_endpoint_defaults_to_worker(tmp_path, monkeypatch, fixtures_dir):
    monkeypatch.setenv("DATABRICKS_HOST", "https://test.databricks.net")
    monkeypatch.setenv("DATABRICKS_TOKEN", "dapi_test")
    monkeypatch.delenv("CRITIC_LLM_ENDPOINT", raising=False)
    config = load_app_config(mcp_config_path=fixtures_dir / "test_mcp_config.yaml")
    # Falls back to worker endpoint when not set
    assert config.critic_llm_endpoint == config.llm_endpoint
```

**Step 2: Run test to verify it fails**

Run: `PYTHONPATH=src python -m pytest tests/test_config.py::test_load_app_config_has_critic_endpoint tests/test_config.py::test_critic_endpoint_defaults_to_worker -v`
Expected: FAIL — `AttributeError: ... has no attribute 'critic_llm_endpoint'`

**Step 3: Write minimal implementation**

In `src/deep_research/config.py`, add field to `AppConfig` (after line 29):

```python
critic_llm_endpoint: str = ""
```

In `load_app_config()`, add to the return block (around line 105):

```python
llm_ep = os.environ.get("LLM_ENDPOINT_NAME", "databricks-claude-sonnet-4-6")
critic_ep = os.environ.get("CRITIC_LLM_ENDPOINT", "") or llm_ep
```

And pass both into the AppConfig constructor:
```python
llm_endpoint=llm_ep,
critic_llm_endpoint=critic_ep,
```

**Step 4: Run test to verify it passes**

Run: `PYTHONPATH=src python -m pytest tests/test_config.py -v`
Expected: All config tests PASS

**Step 5: Commit**

```bash
git add src/deep_research/config.py tests/test_config.py
git commit -m "feat(config): add critic_llm_endpoint for dual-model support"
```

---

### Task 2: Update conftest fixture for dual-model

**Files:**
- Modify: `tests/conftest.py:26-32` (sample_app_config fixture)

**Step 1: Update the fixture**

In `tests/conftest.py`, update `sample_app_config` to include the new field:

```python
@pytest.fixture
def sample_app_config(sample_mcp_server):
    return AppConfig(
        databricks_host="https://test.databricks.net",
        databricks_token="dapi_test_token",
        llm_endpoint="databricks-claude-sonnet-4-6",
        critic_llm_endpoint="databricks-gpt-5-4",
        managed_servers=(sample_mcp_server,),
    )
```

**Step 2: Run full test suite to verify nothing breaks**

Run: `PYTHONPATH=src python -m pytest tests/ -v`
Expected: All tests PASS

**Step 3: Commit**

```bash
git add tests/conftest.py
git commit -m "test: update conftest fixture with dual-model endpoints"
```

---

### Task 3: Add `_create_critic_model` to main.py and thread into graph

**Files:**
- Modify: `src/deep_research/main.py:31-50` (add _create_critic_model)
- Modify: `src/deep_research/main.py:90-95` (create critic + pass to graph)
- Modify: `src/deep_research/graph.py:37-70` (accept critic_model param)
- Test: `tests/test_graph.py`

**Step 1: Write the failing test**

Add to `tests/test_graph.py`:

```python
def test_graph_builds_with_critic_model():
    graph = build_research_graph(model=None, mcp_manager=None, critic_model=None)
    assert graph is not None


def test_graph_has_expected_nodes_with_critic():
    graph = build_research_graph(model=None, mcp_manager=None, critic_model=None)
    node_names = set(graph.get_graph().nodes.keys())
    expected = {
        "clarifier", "planner", "authorizer", "researcher",
        "normalizer", "evaluator", "compressor", "synthesizer", "verifier",
    }
    assert expected.issubset(node_names)
```

**Step 2: Run test to verify it fails**

Run: `PYTHONPATH=src python -m pytest tests/test_graph.py::test_graph_builds_with_critic_model -v`
Expected: FAIL — `TypeError: build_research_graph() got an unexpected keyword argument 'critic_model'`

**Step 3: Write minimal implementation**

In `src/deep_research/graph.py`, update signature and node wiring:

```python
def build_research_graph(
    model: Any,
    mcp_manager: MCPClientManager | None,
    critic_model: Any | None = None,
) -> Any:
    """Build and compile the full research agent graph."""
    # Use critic_model for judgment nodes, fall back to worker model
    _critic = critic_model or model

    graph = StateGraph(ResearchState)

    # Worker nodes
    graph.add_node("clarifier", partial(clarifier_node, model=model))
    graph.add_node("planner", partial(planner_node, model=model))
    graph.add_node("authorizer", partial(authorizer_node, model=model))
    graph.add_node("researcher", partial(researcher_node, model=model, mcp_manager=mcp_manager))
    graph.add_node("normalizer", partial(normalizer_node, model=model))
    graph.add_node("compressor", partial(compressor_node, model=model))
    graph.add_node("synthesizer", partial(synthesizer_node, model=model))

    # Critic nodes (GPT-5.4 when available)
    graph.add_node("evaluator", partial(evaluator_node, model=_critic))
    graph.add_node("verifier", partial(verifier_node, model=_critic))

    # Edges unchanged
    ...
```

In `src/deep_research/main.py`, add `_create_critic_model`:

```python
def _create_critic_model(config):
    """Create the critic LLM model for evaluation/verification.

    Falls back to worker model if critic endpoint matches worker.
    """
    from deep_research.config import ConfigError

    if config.critic_llm_endpoint == config.llm_endpoint:
        return None  # Will use worker model as fallback

    try:
        from databricks_langchain import ChatDatabricks
        model = ChatDatabricks(endpoint=config.critic_llm_endpoint)
        logger.info(f"Critic model initialized: {config.critic_llm_endpoint}")
        return model
    except ImportError:
        logger.warning("databricks-langchain not installed — critic model disabled")
        return None
    except Exception as e:
        logger.warning(f"Failed to initialize critic model: {e} — falling back to worker")
        return None
```

Update `create_production_app()` around line 90-95:

```python
model = _create_model(config)
critic_model = _create_critic_model(config)
logger.info("Model(s) created successfully")

graph = build_research_graph(model=model, mcp_manager=mcp_manager, critic_model=critic_model)
```

**Step 4: Run tests to verify they pass**

Run: `PYTHONPATH=src python -m pytest tests/test_graph.py tests/test_config.py -v`
Expected: All PASS

**Step 5: Commit**

```bash
git add src/deep_research/graph.py src/deep_research/main.py tests/test_graph.py
git commit -m "feat: wire dual-model through graph — critic for evaluator+verifier"
```

---

### Task 4: Add CRITIC_LLM_ENDPOINT to app.yaml

**Files:**
- Modify: `app.yaml`

**Step 1: Add the env var**

Add after the `LLM_ENDPOINT_NAME` line:

```yaml
  - name: CRITIC_LLM_ENDPOINT
    value: "databricks-gpt-5-4"
```

**Step 2: Commit**

```bash
git add app.yaml
git commit -m "config: add CRITIC_LLM_ENDPOINT=databricks-gpt-5-4 to app.yaml"
```

---

### Task 5: Create eval dataset

**Files:**
- Create: `eval/eval_dataset.json`

**Step 1: Create the eval directory and dataset**

Create `eval/eval_dataset.json` with 12 sample queries covering all 4 tools:

```json
[
  {
    "id": "vs-01",
    "query": "What was ANZ's total revenue in 2024?",
    "expected_tools": ["vector_search_anz"],
    "difficulty": "simple",
    "expected_facets": ["2024 revenue figure"]
  },
  {
    "id": "vs-02",
    "query": "What were the key risk factors discussed in the ANZ 2024 annual report?",
    "expected_tools": ["vector_search_anz"],
    "difficulty": "medium",
    "expected_facets": ["risk factors", "specific risks named"]
  },
  {
    "id": "vs-03",
    "query": "How did ANZ's net interest margin change year over year?",
    "expected_tools": ["vector_search_anz"],
    "difficulty": "medium",
    "expected_facets": ["NIM figures", "YoY change", "trend direction"]
  },
  {
    "id": "ga-01",
    "query": "Which airport had the most passengers in the dataset?",
    "expected_tools": ["genie_aviation"],
    "difficulty": "simple",
    "expected_facets": ["airport name", "passenger count"]
  },
  {
    "id": "ga-02",
    "query": "What are the top 5 airlines by passenger volume and how do they compare?",
    "expected_tools": ["genie_aviation"],
    "difficulty": "medium",
    "expected_facets": ["top 5 airlines", "passenger volumes", "comparison"]
  },
  {
    "id": "gs-01",
    "query": "What is the total sales pipeline value?",
    "expected_tools": ["genie_sales_pipeline"],
    "difficulty": "simple",
    "expected_facets": ["total pipeline value"]
  },
  {
    "id": "gs-02",
    "query": "Which sales stages have the most deals and what is the conversion rate?",
    "expected_tools": ["genie_sales_pipeline"],
    "difficulty": "medium",
    "expected_facets": ["deals per stage", "conversion rates"]
  },
  {
    "id": "ka-01",
    "query": "What topics can the knowledge assistant help with?",
    "expected_tools": ["knowledge_assistant"],
    "difficulty": "simple",
    "expected_facets": ["capability description"]
  },
  {
    "id": "ka-02",
    "query": "Explain the key regulatory requirements for Australian banking.",
    "expected_tools": ["knowledge_assistant"],
    "difficulty": "medium",
    "expected_facets": ["regulatory bodies", "key requirements"]
  },
  {
    "id": "multi-01",
    "query": "How does ANZ's financial performance compare to trends in the aviation industry?",
    "expected_tools": ["vector_search_anz", "genie_aviation"],
    "difficulty": "hard",
    "expected_facets": ["ANZ financials", "aviation trends", "comparison"]
  },
  {
    "id": "multi-02",
    "query": "What insights from the ANZ report could inform our sales pipeline strategy?",
    "expected_tools": ["vector_search_anz", "genie_sales_pipeline"],
    "difficulty": "hard",
    "expected_facets": ["ANZ insights", "pipeline relevance", "strategic recommendations"]
  },
  {
    "id": "multi-03",
    "query": "Summarize ANZ's 2024 outlook and supplement with any relevant knowledge base context.",
    "expected_tools": ["vector_search_anz", "knowledge_assistant"],
    "difficulty": "hard",
    "expected_facets": ["2024 outlook", "supplementary context", "synthesis"]
  }
]
```

**Step 2: Commit**

```bash
git add eval/eval_dataset.json
git commit -m "data: add eval dataset with 12 queries across all MCP tools"
```

---

### Task 6: Create eval harness

**Files:**
- Create: `eval/eval_harness.py`
- Test: Run it locally against the live workspace

**Step 1: Write the eval harness**

Create `eval/eval_harness.py`:

```python
"""Eval harness — runs queries through the research graph and scores with MLflow Evaluate."""

from __future__ import annotations

import argparse
import asyncio
import json
import logging
import os
import time
from pathlib import Path

import mlflow
import pandas as pd

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def load_eval_dataset(path: str = "eval/eval_dataset.json") -> list[dict]:
    with open(path) as f:
        return json.load(f)


async def run_single_query(graph, state_factory, query_item: dict) -> dict:
    """Run one eval query through the graph, return results + timing."""
    from deep_research.state import create_initial_state
    from deep_research.models import Budget

    query = query_item["query"]
    tools = query_item["expected_tools"]

    state = create_initial_state(
        user_query=query,
        selected_tools=tools,
        output_mode="report",
        budget=Budget(max_iterations=3, max_tool_calls=15, time_cap_seconds=120),
    )

    start = time.monotonic()
    try:
        result = await graph.ainvoke(state)
        elapsed = time.monotonic() - start

        return {
            "id": query_item["id"],
            "query": query,
            "expected_tools": json.dumps(tools),
            "difficulty": query_item["difficulty"],
            "expected_facets": json.dumps(query_item.get("expected_facets", [])),
            "answer": result.get("final_output", ""),
            "evidence_count": len(result.get("evidence", [])),
            "tool_calls_used": result.get("tool_calls_used", 0),
            "iterations": result.get("iteration_count", 0),
            "sufficiency_score": result.get("sufficiency_score", 0.0),
            "latency_seconds": round(elapsed, 2),
            "error": "",
        }
    except Exception as e:
        elapsed = time.monotonic() - start
        logger.error(f"Query failed: {query_item['id']} — {e}")
        return {
            "id": query_item["id"],
            "query": query,
            "expected_tools": json.dumps(tools),
            "difficulty": query_item["difficulty"],
            "expected_facets": json.dumps(query_item.get("expected_facets", [])),
            "answer": "",
            "evidence_count": 0,
            "tool_calls_used": 0,
            "iterations": 0,
            "sufficiency_score": 0.0,
            "latency_seconds": round(elapsed, 2),
            "error": str(e),
        }


async def run_all_queries(graph, dataset: list[dict]) -> list[dict]:
    """Run all eval queries sequentially (to avoid MCP connection contention)."""
    results = []
    for i, item in enumerate(dataset):
        logger.info(f"Running query {i+1}/{len(dataset)}: {item['id']}")
        result = await run_single_query(graph, None, item)
        results.append(result)
        logger.info(
            f"  → {result['id']}: {result['latency_seconds']}s, "
            f"evidence={result['evidence_count']}, error={result['error'] or 'none'}"
        )
    return results


def compute_metrics(results_df: pd.DataFrame) -> dict:
    """Compute aggregate metrics from eval results."""
    successful = results_df[results_df["error"] == ""]
    return {
        "total_queries": len(results_df),
        "successful_queries": len(successful),
        "error_rate": round(1 - len(successful) / max(len(results_df), 1), 3),
        "latency_p50": round(successful["latency_seconds"].median(), 2) if len(successful) else 0,
        "latency_p95": round(successful["latency_seconds"].quantile(0.95), 2) if len(successful) else 0,
        "avg_evidence_count": round(successful["evidence_count"].mean(), 1) if len(successful) else 0,
        "avg_tool_calls": round(successful["tool_calls_used"].mean(), 1) if len(successful) else 0,
        "avg_sufficiency": round(successful["sufficiency_score"].mean(), 3) if len(successful) else 0,
        "tool_efficiency": round(
            successful["evidence_count"].sum() / max(successful["tool_calls_used"].sum(), 1), 3
        ) if len(successful) else 0,
    }


def run_mlflow_evaluate(results_df: pd.DataFrame, phase_tag: str, experiment_name: str):
    """Run MLflow evaluate with LLM-as-judge metrics."""
    mlflow.set_experiment(experiment_name)

    with mlflow.start_run(run_name=f"eval-{phase_tag}") as run:
        # Log the phase tag
        mlflow.set_tag("phase", phase_tag)
        mlflow.set_tag("eval_type", "research_quality")

        # Log aggregate metrics
        metrics = compute_metrics(results_df)
        mlflow.log_metrics(metrics)

        # Log the raw results as artifact
        results_path = f"/tmp/eval_results_{phase_tag}.json"
        results_df.to_json(results_path, orient="records", indent=2)
        mlflow.log_artifact(results_path)

        # Use mlflow.evaluate with the results
        eval_df = results_df[results_df["error"] == ""][["query", "answer", "expected_facets"]].copy()
        eval_df = eval_df.rename(columns={"query": "inputs", "answer": "predictions"})

        if len(eval_df) > 0:
            try:
                eval_result = mlflow.evaluate(
                    data=eval_df,
                    predictions="predictions",
                    model_type="question-answering",
                    extra_metrics=[],
                    evaluator_config={
                        "col_mapping": {
                            "inputs": "inputs",
                            "predictions": "predictions",
                        }
                    },
                )
                logger.info(f"MLflow evaluate metrics: {eval_result.metrics}")
            except Exception as e:
                logger.warning(f"MLflow evaluate failed (non-fatal): {e}")

        logger.info(f"Eval run logged: {run.info.run_id}")
        logger.info(f"Metrics: {metrics}")
        return run.info.run_id


async def main():
    parser = argparse.ArgumentParser(description="Deep Research Eval Harness")
    parser.add_argument("--phase", default="baseline", help="Phase tag (e.g., baseline, phase1)")
    parser.add_argument("--dataset", default="eval/eval_dataset.json", help="Path to eval dataset")
    parser.add_argument(
        "--experiment",
        default=os.environ.get(
            "MLFLOW_EXPERIMENT_NAME", "/Users/brian.law@databricks.com/deep-research-bot"
        ),
        help="MLflow experiment name",
    )
    parser.add_argument("--max-queries", type=int, default=0, help="Max queries to run (0=all)")
    args = parser.parse_args()

    # Setup
    os.environ.setdefault("MLFLOW_TRACKING_URI", "databricks")

    from deep_research.config import load_app_config
    from deep_research.graph import build_research_graph
    from deep_research.mcp_client import MCPClientManager

    config = load_app_config()

    # Create models
    from databricks_langchain import ChatDatabricks

    model = ChatDatabricks(endpoint=config.llm_endpoint)
    critic_model = None
    if config.critic_llm_endpoint != config.llm_endpoint:
        critic_model = ChatDatabricks(endpoint=config.critic_llm_endpoint)

    # Build graph
    all_servers = {}
    for cfg in [*config.managed_servers, *config.custom_servers]:
        all_servers[cfg.name] = cfg
    mcp_manager = MCPClientManager(all_servers, token=config.databricks_token)
    await mcp_manager.connect_all()

    graph = build_research_graph(model=model, mcp_manager=mcp_manager, critic_model=critic_model)

    # Load dataset
    dataset = load_eval_dataset(args.dataset)
    if args.max_queries > 0:
        dataset = dataset[:args.max_queries]

    logger.info(f"Running {len(dataset)} queries with phase={args.phase}")

    # Run queries
    results = await run_all_queries(graph, dataset)
    results_df = pd.DataFrame(results)

    # Log to MLflow
    run_id = run_mlflow_evaluate(results_df, args.phase, args.experiment)

    # Print summary
    metrics = compute_metrics(results_df)
    print(f"\n{'='*60}")
    print(f"Eval complete — phase={args.phase}, run_id={run_id}")
    print(f"{'='*60}")
    for k, v in metrics.items():
        print(f"  {k}: {v}")

    await mcp_manager.disconnect_all()


if __name__ == "__main__":
    asyncio.run(main())
```

**Step 2: Verify it loads without syntax errors**

Run: `PYTHONPATH=src python -c "import eval.eval_harness; print('OK')"` (may fail on import — that's fine, just check syntax)

Or simpler: `python -m py_compile eval/eval_harness.py`

**Step 3: Commit**

```bash
git add eval/eval_harness.py
git commit -m "feat: add eval harness with MLflow Evaluate integration"
```

---

### Task 7: Run baseline evaluation

**Files:**
- None (execution only)

**Step 1: Run the eval harness against 3 queries as a smoke test**

```bash
PYTHONPATH=src \
DATABRICKS_HOST=https://adb-984752964297111.11.azuredatabricks.net \
DATABRICKS_TOKEN=$DATABRICKS_TOKEN \
MLFLOW_TRACKING_URI=databricks \
python eval/eval_harness.py --phase baseline --max-queries 3
```

Expected: 3 queries run, metrics logged to MLflow, no crashes.

**Step 2: If smoke test passes, run full eval**

```bash
PYTHONPATH=src \
DATABRICKS_HOST=https://adb-984752964297111.11.azuredatabricks.net \
DATABRICKS_TOKEN=$DATABRICKS_TOKEN \
MLFLOW_TRACKING_URI=databricks \
python eval/eval_harness.py --phase baseline
```

Expected: All 12 queries run. Some may error (that's data — we want the baseline).

**Step 3: Verify in MLflow**

Check that the run appears in the experiment with `phase=baseline` tag and logged metrics.

---

### Task 8: Run full test suite and deploy

**Step 1: Run all tests**

```bash
PYTHONPATH=src python -m pytest tests/ -v
```

Expected: All tests PASS (including new ones from Tasks 1-3).

**Step 2: Deploy to Databricks**

Upload changed files and trigger deployment. Verify the app starts with both models configured.

**Step 3: Final commit**

```bash
git add -A
git commit -m "feat: Phase 0 complete — dual-model config + eval baseline"
git push
```

---

## Summary of Changes

| File | Action | Description |
|------|--------|-------------|
| `src/deep_research/config.py` | Modify | Add `critic_llm_endpoint` field + env var loading |
| `src/deep_research/main.py` | Modify | Add `_create_critic_model()`, pass to graph |
| `src/deep_research/graph.py` | Modify | Accept `critic_model`, wire to evaluator + verifier |
| `app.yaml` | Modify | Add `CRITIC_LLM_ENDPOINT` env var |
| `tests/conftest.py` | Modify | Update fixture with dual-model fields |
| `tests/test_config.py` | Modify | Add critic endpoint tests |
| `tests/test_graph.py` | Modify | Add critic_model graph tests |
| `eval/eval_dataset.json` | Create | 12 sample queries across 4 tools |
| `eval/eval_harness.py` | Create | MLflow Evaluate runner script |
