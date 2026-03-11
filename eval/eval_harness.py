"""Eval harness — runs queries through the research graph and scores with MLflow Evaluate."""

from __future__ import annotations

import argparse
import asyncio
import json
import logging
import os
import time

import mlflow
import pandas as pd

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def load_eval_dataset(path: str = "eval/eval_dataset.json") -> list[dict]:
    with open(path) as f:
        return json.load(f)


async def run_single_query(graph, query_item: dict) -> dict:
    """Run one eval query through the graph, return results + timing."""
    from deep_research.models import Budget
    from deep_research.state import create_initial_state

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
        result = await run_single_query(graph, item)
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
        "latency_p50": round(successful["latency_seconds"].median(), 2)
        if len(successful)
        else 0,
        "latency_p95": round(successful["latency_seconds"].quantile(0.95), 2)
        if len(successful)
        else 0,
        "avg_evidence_count": round(successful["evidence_count"].mean(), 1)
        if len(successful)
        else 0,
        "avg_tool_calls": round(successful["tool_calls_used"].mean(), 1)
        if len(successful)
        else 0,
        "avg_sufficiency": round(successful["sufficiency_score"].mean(), 3)
        if len(successful)
        else 0,
        "tool_efficiency": round(
            successful["evidence_count"].sum()
            / max(successful["tool_calls_used"].sum(), 1),
            3,
        )
        if len(successful)
        else 0,
    }


def run_mlflow_evaluate(
    results_df: pd.DataFrame, phase_tag: str, experiment_name: str
):
    """Run MLflow evaluate with LLM-as-judge metrics."""
    mlflow.set_experiment(experiment_name)

    with mlflow.start_run(run_name=f"eval-{phase_tag}") as run:
        mlflow.set_tag("phase", phase_tag)
        mlflow.set_tag("eval_type", "research_quality")

        metrics = compute_metrics(results_df)
        mlflow.log_metrics(metrics)

        results_path = f"/tmp/eval_results_{phase_tag}.json"
        results_df.to_json(results_path, orient="records", indent=2)
        mlflow.log_artifact(results_path)

        eval_df = results_df[results_df["error"] == ""][
            ["query", "answer", "expected_facets"]
        ].copy()
        eval_df = eval_df.rename(
            columns={"query": "inputs", "answer": "predictions"}
        )

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
    parser.add_argument(
        "--phase", default="baseline", help="Phase tag (e.g., baseline, phase1)"
    )
    parser.add_argument(
        "--dataset", default="eval/eval_dataset.json", help="Path to eval dataset"
    )
    parser.add_argument(
        "--experiment",
        default=os.environ.get(
            "MLFLOW_EXPERIMENT_NAME",
            "/Users/brian.law@databricks.com/deep-research-bot",
        ),
        help="MLflow experiment name",
    )
    parser.add_argument(
        "--max-queries", type=int, default=0, help="Max queries to run (0=all)"
    )
    args = parser.parse_args()

    os.environ.setdefault("MLFLOW_TRACKING_URI", "databricks")

    from deep_research.config import load_app_config
    from deep_research.graph import build_research_graph
    from deep_research.mcp_client import MCPClientManager

    config = load_app_config()

    from databricks_langchain import ChatDatabricks

    model = ChatDatabricks(endpoint=config.llm_endpoint)
    critic_model = None
    if config.critic_llm_endpoint != config.llm_endpoint:
        critic_model = ChatDatabricks(endpoint=config.critic_llm_endpoint)

    all_servers = {}
    for cfg in [*config.managed_servers, *config.custom_servers]:
        all_servers[cfg.name] = cfg
    mcp_manager = MCPClientManager(all_servers, token=config.databricks_token)
    await mcp_manager.connect_all()

    graph = build_research_graph(
        model=model, mcp_manager=mcp_manager, critic_model=critic_model
    )

    dataset = load_eval_dataset(args.dataset)
    if args.max_queries > 0:
        dataset = dataset[: args.max_queries]

    logger.info(f"Running {len(dataset)} queries with phase={args.phase}")

    results = await run_all_queries(graph, dataset)
    results_df = pd.DataFrame(results)

    run_id = run_mlflow_evaluate(results_df, args.phase, args.experiment)

    metrics = compute_metrics(results_df)
    print(f"\n{'='*60}")
    print(f"Eval complete — phase={args.phase}, run_id={run_id}")
    print(f"{'='*60}")
    for k, v in metrics.items():
        print(f"  {k}: {v}")

    await mcp_manager.disconnect_all()


if __name__ == "__main__":
    asyncio.run(main())
