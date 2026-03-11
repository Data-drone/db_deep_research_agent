"""Manual test harness for testing against real Databricks workspace.

Usage:
    # Set environment variables first:
    # DATABRICKS_HOST, DATABRICKS_TOKEN, GENIE_SPACE_ID, etc.

    python notebooks/test_harness.py "What was Q3 revenue?"
    python notebooks/test_harness.py "Analyze customer churn" --mode report
    python notebooks/test_harness.py "Compare metrics" --tools genie_sales,vector_search_kb
"""

import argparse
import asyncio
import time

from deep_research.config import load_app_config
from deep_research.graph import build_research_graph
from deep_research.mcp_client import MCPClientManager
from deep_research.state import create_initial_state


async def run_research(query: str, tools: list[str], mode: str) -> None:
    """Run a research query and print results."""
    config = load_app_config()

    # Initialize MCP client
    all_servers = {}
    for cfg in config.managed_servers:
        all_servers[cfg.name] = cfg
    for cfg in config.custom_servers:
        all_servers[cfg.name] = cfg

    mcp_manager = MCPClientManager(all_servers, token=config.databricks_token)
    await mcp_manager.connect_all()

    # TODO: Initialize real ChatDatabricks model
    # from langchain_databricks import ChatDatabricks
    # model = ChatDatabricks(endpoint=config.llm_endpoint)
    model = None  # Replace with real model

    graph = build_research_graph(model=model, mcp_manager=mcp_manager)

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

    print("\n--- Metadata ---")
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
    parser.add_argument(
        "--tools", default="genie_aus_market", help="Comma-separated tool names"
    )
    parser.add_argument("--mode", default="chat", choices=["chat", "report"])
    args = parser.parse_args()

    tools = [t.strip() for t in args.tools.split(",")]
    asyncio.run(run_research(args.query, tools, args.mode))


if __name__ == "__main__":
    main()
