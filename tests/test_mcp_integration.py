import sys
from pathlib import Path

import pytest
from mcp import StdioServerParameters

from mini_llm.mcp_integration.adapter import mcp_tool_to_schema, register_mcp_tools
from mini_llm.mcp_integration.client import MCPClient
from mini_llm.tools.builtin.read_file import READ_FILE_TOOL
from mini_llm.tools.executor import ToolExecutor
from mini_llm.tools.registry import ToolRegistry


@pytest.mark.asyncio
async def test_mcp_tool_through_tool_executor():
    server_params = StdioServerParameters(
        command=sys.executable,
        args=["-m", "mini_llm.mcp_integration.server"],
    )

    async with MCPClient(server_params=server_params) as client:
        tools_result = await client.list_tools()

        tool_names = {tool.name for tool in tools_result.tools}
        assert "count_lines" in tool_names
        assert "slow_tool" in tool_names

        mcp_tool = tools_result.tools[0]

        schema = mcp_tool_to_schema(
            tool=mcp_tool,
            client=client,
            namespace="demo",
        )

        assert schema.name == "demo.count_lines"
        assert schema.parameters["properties"]["text"]["type"] == "string"

        registry = ToolRegistry()
        registry.register(schema)

        executor = ToolExecutor(registry)

        result = await executor.execute(
            "demo.count_lines",
            {
                "text": "hello\nworld\nmcp",
            },
        )

        assert result.success is True
        assert result.tool_name == "demo.count_lines"
        assert result.output == {"result": 3}
        assert result.error is None


@pytest.mark.asyncio
async def test_mcp_client_requires_connection():
    server_params = StdioServerParameters(
        command=sys.executable,
        args=["-m", "mini_llm.mcp_integration.server"],
    )

    client = MCPClient(server_params=server_params)

    with pytest.raises(
        RuntimeError,
        match="MCP client is not connected",
    ):
        await client.call_tool(
            "count_lines",
            {"text": "hello"},
        )


@pytest.mark.asyncio
async def test_mcp_tool_timeout_returns_failed_tool_result():
    server_params = StdioServerParameters(
        command=sys.executable,
        args=["-m", "mini_llm.mcp_integration.server"],
    )

    async with MCPClient(server_params=server_params) as client:
        tools_result = await client.list_tools()
        slow_tool = next(
            tool for tool in tools_result.tools if tool.name == "slow_tool"
        )

        schema = mcp_tool_to_schema(
            tool=slow_tool,
            client=client,
            namespace="demo",
            timeout=0.1,
        )

        registry = ToolRegistry()
        registry.register(schema)

        executor = ToolExecutor(registry)

        result = await executor.execute(
            "demo.slow_tool",
            {
                "seconds": 1.0,
            },
        )

        assert result.success is False
        assert result.output is None
        assert result.error is not None
        assert "timed out" in result.error


@pytest.mark.asyncio
async def test_mcp_tool_error_returns_failed_tool_result():
    server_params = StdioServerParameters(
        command=sys.executable,
        args=["-m", "mini_llm.mcp_integration.server"],
    )

    async with MCPClient(server_params=server_params) as client:
        tools_result = await client.list_tools()
        always_fail = next(
            tool for tool in tools_result.tools if tool.name == "always_fail"
        )

        schema = mcp_tool_to_schema(
            tool=always_fail,
            client=client,
            namespace="demo",
        )

        registry = ToolRegistry()
        registry.register(schema)

        executor = ToolExecutor(registry)

        result = await executor.execute(
            "demo.always_fail",
            {},
        )

        assert result.success is False
        assert result.output is None
        assert result.error is not None
        assert result.error == "Error executing tool always_fail"


@pytest.mark.asyncio
async def test_register_mcp_tools():
    server_params = StdioServerParameters(
        command=sys.executable,
        args=["-m", "mini_llm.mcp_integration.server"],
    )

    registry = ToolRegistry()

    async with MCPClient(server_params=server_params) as client:
        schemas = await register_mcp_tools(
            client=client,
            registry=registry,
            namespace="demo",
        )

        schema_names = {schema.name for schema in schemas}

        assert "demo.count_lines" in schema_names
        assert "demo.slow_tool" in schema_names
        assert "demo.always_fail" in schema_names

        assert registry.get("demo.count_lines").name == "demo.count_lines"
        assert registry.get("demo.slow_tool").name == "demo.slow_tool"
        assert registry.get("demo.always_fail").name == "demo.always_fail"


@pytest.mark.asyncio
async def test_native_and_mcp_tools_share_same_executor(tmp_path: Path):

    file_path = tmp_path / "test.txt"
    file_path.write_text("hello\nworld", encoding="utf-8")

    registry = ToolRegistry()

    # 按你 Week 2 的现有方式注册一个 Native Tool
    registry.register(READ_FILE_TOOL)

    server_params = StdioServerParameters(
        command=sys.executable,
        args=["-m", "mini_llm.mcp_integration.server"],
    )

    async with MCPClient(server_params) as client:
        await register_mcp_tools(
            client=client,
            registry=registry,
            namespace="demo",
        )

        executor = ToolExecutor(registry)

        native_result = await executor.execute(
            "read_file",
            {"path": str(file_path), "start_line": 2, "end_line": 2},
        )

        mcp_result = await executor.execute(
            "demo.count_lines",
            {
                "text": "hello\nworld",
            },
        )

        assert native_result.success is True
        assert native_result.output == "world"
        assert mcp_result.success is True
        assert mcp_result.output == {"result": 2}
