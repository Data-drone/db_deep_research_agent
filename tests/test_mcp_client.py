"""Tests for MCP client manager."""

from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from deep_research.config import MCPServerConfig
from deep_research.mcp_client import (
    MCPClientManager,
    _extract_text_from_call_result,
    _parse_genie_async_response,
)


@pytest.fixture
def mock_server_configs():
    return {
        "genie_sales": MCPServerConfig(
            name="genie_sales",
            url="https://test.databricks.net/api/2.0/mcp/genie/space-123",
            display_name="Sales Data (Genie)",
            server_kind="managed",
            enabled=True,
            risk_tier="safe",
            capability="read",
            managed_type="genie",
            description="Sales queries",
        ),
        "disabled_server": MCPServerConfig(
            name="disabled_server",
            url="https://test.databricks.net/api/2.0/mcp/genie/space-999",
            display_name="Disabled",
            server_kind="managed",
            enabled=False,
            risk_tier="safe",
            capability="read",
            managed_type="genie",
        ),
        "vector_kb": MCPServerConfig(
            name="vector_kb",
            url="https://test.databricks.net/api/2.0/mcp/vector-search/cat/schema/idx",
            display_name="Knowledge Base",
            server_kind="managed",
            enabled=True,
            risk_tier="safe",
            capability="read",
            managed_type="vector_search",
        ),
    }


# ── Sync filter/query tests (unchanged) ──


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


def test_manager_get_server_info_unknown(mock_server_configs):
    manager = MCPClientManager(mock_server_configs, token="test")
    with pytest.raises(KeyError, match="Unknown MCP server"):
        manager.get_server_info("nonexistent")


def test_manager_get_servers_by_risk_tier(mock_server_configs):
    manager = MCPClientManager(mock_server_configs, token="test")
    safe = manager.get_servers_by_risk_tier("safe")
    assert "genie_sales" in safe
    assert "vector_kb" in safe
    # disabled not included
    assert "disabled_server" not in safe


def test_manager_get_servers_by_capability(mock_server_configs):
    manager = MCPClientManager(mock_server_configs, token="test")
    read_only = manager.get_servers_by_capability("read")
    assert "genie_sales" in read_only


# ── Helper function tests ──


def test_extract_text_from_call_result_with_content():
    """CallToolResult with .content list of TextContent items."""
    item1 = SimpleNamespace(text="Hello ")
    item2 = SimpleNamespace(text="World")
    result = SimpleNamespace(content=[item1, item2])
    assert _extract_text_from_call_result(result) == "Hello \nWorld"


def test_extract_text_from_call_result_fallback():
    """Falls back to str() when no .content attribute."""
    assert _extract_text_from_call_result({"some": "dict"}) == "{'some': 'dict'}"


def test_parse_genie_async_response_filtering():
    text = (
        'The query is being processed. Status: FILTERING_CONTEXT. '
        'conversation_id: "conv-abc", message_id: "msg-def"'
    )
    result = _parse_genie_async_response(text)
    assert result is not None
    assert result["conversation_id"] == "conv-abc"
    assert result["message_id"] == "msg-def"


def test_parse_genie_async_response_json():
    import json
    text = json.dumps({
        "status": "FILTERING_CONTEXT",
        "conversation_id": "conv-123",
        "message_id": "msg-456",
    })
    # The text contains FILTERING_CONTEXT so should match
    result = _parse_genie_async_response(text)
    assert result is not None
    assert result["conversation_id"] == "conv-123"


def test_parse_genie_async_response_completed():
    """A completed response should return None (not a polling response)."""
    text = "The total pipeline value is $4.2M across 127 deals."
    result = _parse_genie_async_response(text)
    assert result is None


# ── Async connection/call tests ──


def _make_mock_tool(name="query_space_123"):
    return SimpleNamespace(name=name)


@pytest.mark.asyncio
async def test_connect_all_discovers_tools(mock_server_configs):
    """connect_all should create DatabricksMCPClient and call alist_tools."""
    manager = MCPClientManager(mock_server_configs, token="test")

    mock_client = MagicMock()
    mock_client.alist_tools = AsyncMock(return_value=[
        _make_mock_tool("query_space_123"),
        _make_mock_tool("poll_response_123"),
    ])

    mock_ws = MagicMock()

    with patch.object(manager, "_create_workspace_client", return_value=mock_ws), \
         patch("databricks_mcp.DatabricksMCPClient", return_value=mock_client):
        await manager.connect_all()

    # genie_sales and vector_kb should be connected (disabled_server excluded)
    assert "genie_sales" in manager._connections
    assert "vector_kb" in manager._connections
    assert "disabled_server" not in manager._connections

    conn = manager._connections["genie_sales"]
    assert conn["query_tool"] == "query_space_123"
    assert conn["poll_tool"] == "poll_response_123"


@pytest.mark.asyncio
async def test_call_tool_vector_search(mock_server_configs):
    """call_tool on a vector search server returns text result."""
    manager = MCPClientManager(mock_server_configs, token="test")

    mock_result = SimpleNamespace(
        content=[SimpleNamespace(text="ANZ reported revenue of $10B in FY2024.")]
    )
    mock_client = MagicMock()
    mock_client.acall_tool = AsyncMock(return_value=mock_result)

    manager._connections["vector_kb"] = {
        "client": mock_client,
        "tools": [_make_mock_tool("cat__schema__idx__mcp__v1")],
        "query_tool": "cat__schema__idx__mcp__v1",
        "poll_tool": None,
        "config": mock_server_configs["vector_kb"],
    }

    result = await manager.call_tool("vector_kb", "query", {"query": "ANZ revenue"})
    assert "ANZ reported revenue" in result["result"]
    assert result["source"] == "vector_kb"
    mock_client.acall_tool.assert_called_once_with(
        "cat__schema__idx__mcp__v1", {"query": "ANZ revenue"}
    )


@pytest.mark.asyncio
async def test_call_tool_genie_with_polling(mock_server_configs):
    """call_tool on Genie server polls until complete."""
    manager = MCPClientManager(mock_server_configs, token="test")

    # First call returns async/polling response
    async_result = SimpleNamespace(
        content=[SimpleNamespace(
            text='Status: FILTERING_CONTEXT. conversation_id: "c1", message_id: "m1"'
        )]
    )
    # Poll returns still processing, then completed
    poll_processing = SimpleNamespace(
        content=[SimpleNamespace(
            text='Status: EXECUTING_QUERY. conversation_id: "c1", message_id: "m1"'
        )]
    )
    poll_done = SimpleNamespace(
        content=[SimpleNamespace(text="Total pipeline: $5M")]
    )

    mock_client = MagicMock()
    mock_client.acall_tool = AsyncMock(
        side_effect=[async_result, poll_processing, poll_done]
    )

    manager._connections["genie_sales"] = {
        "client": mock_client,
        "tools": [
            _make_mock_tool("query_space_123"),
            _make_mock_tool("poll_response_123"),
        ],
        "query_tool": "query_space_123",
        "poll_tool": "poll_response_123",
        "config": mock_server_configs["genie_sales"],
    }

    # Patch sleep to avoid waiting in tests
    with patch("deep_research.mcp_client.asyncio.sleep", new_callable=AsyncMock):
        result = await manager.call_tool(
            "genie_sales", "query", {"query": "total pipeline"}
        )

    assert "Total pipeline: $5M" in result["result"]
    assert result["source"] == "genie_sales"
    # Should have called: 1 query + 2 polls = 3 total
    assert mock_client.acall_tool.call_count == 3


@pytest.mark.asyncio
async def test_call_tool_not_connected(mock_server_configs):
    """call_tool raises RuntimeError if server not connected."""
    manager = MCPClientManager(mock_server_configs, token="test")
    with pytest.raises(RuntimeError, match="Not connected"):
        await manager.call_tool("genie_sales", "query", {"query": "test"})


@pytest.mark.asyncio
async def test_connect_all_handles_failure(mock_server_configs):
    """connect_all logs error but continues if one server fails."""
    manager = MCPClientManager(mock_server_configs, token="test")

    call_count = 0

    def make_client(*args, **kwargs):
        nonlocal call_count
        call_count += 1
        client = MagicMock()
        if call_count == 1:
            client.alist_tools = AsyncMock(side_effect=Exception("connection refused"))
        else:
            client.alist_tools = AsyncMock(return_value=[_make_mock_tool("tool1")])
        return client

    mock_ws = MagicMock()

    with patch.object(manager, "_create_workspace_client", return_value=mock_ws), \
         patch("databricks_mcp.DatabricksMCPClient", side_effect=make_client):
        await manager.connect_all()

    # One should have failed, one should have succeeded
    # (genie_sales fails, vector_kb succeeds — or vice versa depending on dict ordering)
    assert len(manager._connections) == 1


# ── Knowledge Assistant wrapper tests ──


@pytest.fixture
def ka_server_config():
    return {
        "knowledge_assistant": MCPServerConfig(
            name="knowledge_assistant",
            url="https://test.databricks.net/serving-endpoints/ka-endpoint/invocations",
            display_name="Knowledge Assistant",
            server_kind="managed",
            enabled=True,
            risk_tier="safe",
            capability="read",
            managed_type="knowledge_assistant",
            description="Test KA",
        ),
    }


@pytest.mark.asyncio
async def test_call_tool_knowledge_assistant(ka_server_config):
    """call_tool on a knowledge_assistant uses httpx POST, not MCP client."""
    import json as _json
    manager = MCPClientManager(ka_server_config, token="test")

    # Manually set up connection (skip connect_all which would hit real endpoint)
    manager._connections["knowledge_assistant"] = {
        "client": None,
        "tools": [_make_mock_tool("knowledge_assistant_knowledge_assistant")],
        "query_tool": "knowledge_assistant_knowledge_assistant",
        "poll_tool": None,
        "config": ka_server_config["knowledge_assistant"],
        "auth_headers": {"Authorization": "Bearer fake-token"},
    }

    ka_response = {
        "output": [{
            "type": "message",
            "role": "assistant",
            "content": [{"type": "output_text", "text": "ANZ reported $10B revenue in 2024."}],
        }]
    }

    mock_resp = MagicMock()
    mock_resp.status_code = 200
    mock_resp.json.return_value = ka_response

    mock_client_instance = AsyncMock()
    mock_client_instance.post = AsyncMock(return_value=mock_resp)
    mock_client_instance.__aenter__ = AsyncMock(return_value=mock_client_instance)
    mock_client_instance.__aexit__ = AsyncMock(return_value=None)

    with patch("deep_research.mcp_client.httpx.AsyncClient", return_value=mock_client_instance):
        result = await manager.call_tool(
            "knowledge_assistant", "query", {"query": "What was ANZ revenue?"}
        )

    assert "ANZ reported $10B revenue" in result["result"]
    assert result["source"] == "knowledge_assistant"
    # Verify the POST was called with correct format
    mock_client_instance.post.assert_called_once()
    call_args = mock_client_instance.post.call_args
    assert call_args[1]["json"]["input"][0]["content"] == "What was ANZ revenue?"


@pytest.mark.asyncio
async def test_call_tool_knowledge_assistant_error(ka_server_config):
    """call_tool on KA raises RuntimeError on non-200 response."""
    manager = MCPClientManager(ka_server_config, token="test")

    manager._connections["knowledge_assistant"] = {
        "client": None,
        "tools": [_make_mock_tool("knowledge_assistant_knowledge_assistant")],
        "query_tool": "knowledge_assistant_knowledge_assistant",
        "poll_tool": None,
        "config": ka_server_config["knowledge_assistant"],
        "auth_headers": {},
    }

    mock_resp = MagicMock()
    mock_resp.status_code = 500
    mock_resp.text = "Internal Server Error"

    mock_client_instance = AsyncMock()
    mock_client_instance.post = AsyncMock(return_value=mock_resp)
    mock_client_instance.__aenter__ = AsyncMock(return_value=mock_client_instance)
    mock_client_instance.__aexit__ = AsyncMock(return_value=None)

    with patch("deep_research.mcp_client.httpx.AsyncClient", return_value=mock_client_instance):
        with pytest.raises(RuntimeError, match="KA endpoint returned 500"):
            await manager.call_tool(
                "knowledge_assistant", "query", {"query": "test"}
            )
