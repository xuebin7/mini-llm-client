import json
import sys
from pathlib import Path

import pytest
from mcp import StdioServerParameters

from mini_llm.agent.loop import AgentLoop
from mini_llm.mcp_integration.adapter import register_mcp_tools
from mini_llm.mcp_integration.client import MCPClient
from mini_llm.models import ResponseResult, ToolCall
from mini_llm.tools.builtin.read_file import READ_FILE_TOOL
from mini_llm.tools.builtin.search_code import SEARCH_CODE_TOOL
from mini_llm.tools.registry import ToolRegistry


class FakeLLMClient:
    def __init__(self, file_path: Path):
        self.file_path = file_path
        self.call_count = 0

    def responses(self, request):
        self.call_count += 1

        if self.call_count == 1:
            return ResponseResult(
                text=None,
                tool_calls=[
                    ToolCall(
                        call_id="call_123",
                        name="read_file",
                        arguments=json.dumps(
                            {
                                "path": str(self.file_path),
                            }
                        ),
                    )
                ],
                model="test-model",
                response_id="resp_1",
                output=[
                    {
                        "type": "function_call",
                        "call_id": "call_123",
                        "name": "read_file",
                        "arguments": json.dumps(
                            {
                                "path": str(self.file_path),
                            }
                        ),
                    }
                ],
            )

        return ResponseResult(
            text="The file says hello.",
            tool_calls=[],
            model="test-model",
            response_id="resp_2",
            output=[
                {
                    "type": "message",
                    "content": [
                        {
                            "type": "output_text",
                            "text": "The file says hello.",
                        }
                    ],
                }
            ],
        )


@pytest.mark.asyncio
async def test_agent_oog_executes_tool_returns_answer(tmp_path: Path):
    file_path = tmp_path / "hello.txt"
    file_path.write_text(
        "hello",
        encoding="utf-8",
    )

    registry = ToolRegistry()
    registry.register(READ_FILE_TOOL)

    client = FakeLLMClient(file_path)

    agent = AgentLoop(
        client=client,
        registry=registry,
        model="test-model",
    )

    result = await agent.run("Read the file.")

    assert result == "The file says hello."


class FakeRecoveryLLMClient:
    def __init__(self, search_path: Path):
        self.search_path = search_path
        self.call_count = 0
        self.requests = []

    def responses(self, request):
        self.call_count += 1
        self.requests.append(request)

        if self.call_count == 1:
            arguments = json.dumps(
                {
                    "path": str(self.search_path / "missing.py"),
                }
            )

            return ResponseResult(
                text=None,
                tool_calls=[
                    ToolCall(
                        call_id="call_1",
                        name="read_file",
                        arguments=arguments,
                    )
                ],
                model="test-model",
                response_id="resp_1",
                output=[
                    {
                        "type": "function_call",
                        "call_id": "call_1",
                        "name": "read_file",
                        "arguments": arguments,
                    }
                ],
            )

        if self.call_count == 2:
            arguments = json.dumps(
                {
                    "query": "ToolExecutor",
                    "path": str(self.search_path),
                }
            )

            return ResponseResult(
                text=None,
                tool_calls=[
                    ToolCall(
                        call_id="call_2",
                        name="search_code",
                        arguments=arguments,
                    )
                ],
                model="test-model",
                response_id="resp_2",
                output=[
                    {
                        "type": "function_call",
                        "call_id": "call_2",
                        "name": "search_code",
                        "arguments": arguments,
                    }
                ],
            )

        return ResponseResult(
            text="I found ToolExecutor in executor.py.",
            tool_calls=[],
            model="test-model",
            response_id="resp_3",
            output=[
                {
                    "type": "message",
                    "content": {
                        "type": "output_text",
                        "text": "I found ToolExecutor in executor.py.",
                    },
                }
            ],
        )


@pytest.mark.asyncio
async def test_agent_loop_recovers_from_tool_failure(tmp_path: Path):
    file_path = tmp_path / "executor.py"
    file_path.write_text(
        "class ToolExecutor\n    pass\n",
        encoding="utf-8",
    )

    registry = ToolRegistry()
    registry.register(READ_FILE_TOOL)
    registry.register(SEARCH_CODE_TOOL)

    client = FakeRecoveryLLMClient(tmp_path)

    agent = AgentLoop(
        client=client,
        registry=registry,
        model="test-model",
    )

    result = await agent.run("Find ToolExecutor.")

    assert result == "I found ToolExecutor in executor.py."
    assert client.call_count == 3

    second_request = client.requests[1]
    assert isinstance(second_request.input, list)

    tool_outputs = [
        item
        for item in second_request.input
        if item.get("type") == "function_call_output"
    ]

    assert len(tool_outputs) == 1

    failure = json.loads(tool_outputs[0]["output"])

    assert failure["success"] is False
    assert "File not found" in failure["error"]


class FakeUnknownToolClient:
    def __init__(self):
        self.call_count = 0
        self.requests = []

    def responses(self, request):
        self.call_count += 1
        self.requests.append(request)

        if self.call_count == 1:
            return ResponseResult(
                text=None,
                tool_calls=[
                    ToolCall(
                        call_id="call_1",
                        name="unknown_tool",
                        arguments="{}",
                    )
                ],
                model="test-model",
                response_id="resp_1",
                output=[
                    {
                        "type": "function_call",
                        "call_id": "call_1",
                        "name": "unknown_tool",
                        "arguments": "{}",
                    }
                ],
            )

        return ResponseResult(
            text="The requested tool is unavailable.",
            tool_calls=[],
            model="test-model",
            response_id="resp_2",
            output=[],
        )


@pytest.mark.asyncio
async def test_unknown_loop_handles_unknown_tool():
    registry = ToolRegistry()
    client = FakeUnknownToolClient()

    agent = AgentLoop(
        client=client,
        registry=registry,
        model="test-model",
    )

    result = await agent.run("Use unknown tool.")

    assert result == "The requested tool is unavailable."
    assert client.call_count == 2

    second_request = client.requests[1]

    tool_outputs = [
        item
        for item in second_request.input
        if item.get("type") == "function_call_output"
    ]

    assert len(tool_outputs) == 1

    failure = json.loads(tool_outputs[0]["output"])

    assert failure["success"] is False
    assert "Unknown tool" in failure["error"]


class FakeInfiniteToolClient:
    def __init__(self):
        self.call_count = 0

    def responses(self, request):
        self.call_count += 1

        return ResponseResult(
            text=None,
            tool_calls=[
                ToolCall(
                    call_id=f"call_{self.call_count}",
                    name="read_file",
                    arguments='{"path": "missing.txt"}',
                )
            ],
            model="test-model",
            response_id=f"resp_{self.call_count}",
            output=[
                {
                    "type": "function_call",
                    "call_id": f"call_{self.call_count}",
                    "name": "read_file",
                    "arguments": '{"path": "missing.txt"}',
                }
            ],
        )


@pytest.mark.asyncio
async def test_agent_loop_stops_at_max_steps():
    registry = ToolRegistry()
    registry.register(READ_FILE_TOOL)

    client = FakeInfiniteToolClient()

    agent = AgentLoop(
        client=client,
        registry=registry,
        model="test-model",
        max_steps=3,
    )

    with pytest.raises(
        RuntimeError,
        match="Agent exceeded max_steps=3",
    ):
        await agent.run("Keep reading the file.")

    assert client.call_count == 3


class FakeNativeMCPClient:
    def __init__(self, file_path: Path):
        self.file_path = file_path
        self.call_count = 0
        self.requests = []

    def responses(self, request):
        self.call_count += 1
        self.requests.append(request)

        if self.call_count == 1:
            arguments = json.dumps(
                {
                    "path": str(self.file_path),
                    "start_line": 1,
                    "end_line": 2,
                }
            )

            return ResponseResult(
                text=None,
                tool_calls=[
                    ToolCall(
                        call_id="call_native",
                        name="read_file",
                        arguments=arguments,
                    )
                ],
                model="test-model",
                response_id="resp_1",
                output=[
                    {
                        "type": "function_call",
                        "call_id": "call_native",
                        "name": "read_file",
                        "arguments": arguments,
                    }
                ],
            )

        if self.call_count == 2:
            arguments = json.dumps(
                {
                    "text": "hello\nworld",
                }
            )

            return ResponseResult(
                text=None,
                tool_calls=[
                    ToolCall(
                        call_id="call_mcp",
                        name="demo.count_lines",
                        arguments=arguments,
                    )
                ],
                model="test-model",
                response_id="resp_2",
                output=[
                    {
                        "type": "function_call",
                        "call_id": "call_mcp",
                        "name": "demo.count_lines",
                        "arguments": arguments,
                    }
                ],
            )

        return ResponseResult(
            text="The file has 2 lines.",
            tool_calls=[],
            model="test-model",
            response_id="resp_3",
            output=[
                {
                    "type": "message",
                    "content": [
                        {
                            "type": "output_text",
                            "text": "The file has 2 lines.",
                        }
                    ],
                }
            ],
        )


@pytest.mark.asyncio
async def test_agent_loop_can_use_native_and_mcp_tools(tmp_path: Path):
    file_path = tmp_path / "hello.txt"
    file_path.write_text(
        "hello\nworld",
        encoding="utf-8",
    )

    registry = ToolRegistry()
    registry.register(READ_FILE_TOOL)

    server_params = StdioServerParameters(
        command=sys.executable,
        args=["-m", "mini_llm.mcp_integration.server"],
    )

    async with MCPClient(server_params=server_params) as mcp_client:
        await register_mcp_tools(
            client=mcp_client,
            registry=registry,
            namespace="demo",
        )

        client = FakeNativeMCPClient(file_path)

        agent = AgentLoop(
            client=client,
            registry=registry,
            model="test-model",
        )

        result = await agent.run("Read the file and count its lines.")

        assert result == "The file has 2 lines."
        assert client.call_count == 3

        second_request = client.requests[1]

        native_outputs = [
            item
            for item in second_request.input
            if item.get("type") == "function_call_output"
        ]

        assert len(native_outputs) == 1

        third_request = client.requests[2]

        tool_outputs = [
            item
            for item in third_request.input
            if item.get("type") == "function_call_output"
        ]

        assert len(tool_outputs) == 2
