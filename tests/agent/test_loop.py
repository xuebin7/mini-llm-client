import json
import sys
from pathlib import Path
from typing import Any

import pytest
from mcp import StdioServerParameters

from mini_llm.agent.loop import AgentLoop
from mini_llm.harness.collector import CollectingEventSink
from mini_llm.harness.events import EventType
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


class FakeContextManager:
    def __init__(self):
        self.prepare_called = False
        self.truncate_called = False

    def prepare(
        self,
        items: list[dict[str, Any]],
    ) -> list[dict[str, Any]]:
        self.prepare_called = True
        return items

    def truncate_tool_result(self, text):
        self.truncate_called = True
        return text


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

    context_manager = FakeContextManager()

    agent = AgentLoop(
        client=client,
        registry=registry,
        context_manager=context_manager,
        model="test-model",
    )

    result = await agent.run("Read the file.")

    assert result == "The file says hello."
    assert context_manager.prepare_called
    assert context_manager.truncate_called


@pytest.mark.asyncio
async def test_agent_loop_emits_tool_events(tmp_path: Path):
    file_path = tmp_path / "hello.txt"
    file_path.write_text(
        "hello",
        encoding="utf-8",
    )

    registry = ToolRegistry()
    registry.register(READ_FILE_TOOL)

    client = FakeLLMClient(file_path)

    context_manager = FakeContextManager()

    sink = CollectingEventSink()

    agent = AgentLoop(
        client=client,
        registry=registry,
        context_manager=context_manager,
        model="test-model",
        event_sink=sink,
    )

    result = await agent.run("Read the file.")

    assert [event.type for event in sink.events] == [
        EventType.TURN_STARTED,
        EventType.TOOL_STARTED,
        EventType.TOOL_COMPLETED,
        EventType.AGENT_MESSAGE,
        EventType.TURN_COMPLETED,
    ]

    tool_started = sink.events[1]
    tool_completed = sink.events[2]
    agent_message = sink.events[-2]

    assert tool_started.data["tool_name"] == "read_file"
    assert tool_completed.data["tool_name"] == "read_file"
    assert tool_started.data["call_id"] == tool_completed.data["call_id"]
    assert agent_message.data["text"] == "The file says hello."


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

    context_manager = FakeContextManager()

    sink = CollectingEventSink()

    agent = AgentLoop(
        client=client,
        registry=registry,
        context_manager=context_manager,
        model="test-model",
        event_sink=sink,
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

    assert [event.type for event in sink.events] == [
        EventType.TURN_STARTED,
        EventType.TOOL_STARTED,
        EventType.TOOL_FAILED,
        EventType.TOOL_STARTED,
        EventType.TOOL_COMPLETED,
        EventType.AGENT_MESSAGE,
        EventType.TURN_COMPLETED,
    ]

    agent_message = sink.events[-2]

    assert agent_message.type == EventType.AGENT_MESSAGE
    assert agent_message.data["text"] == "I found ToolExecutor in executor.py."


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

    context_manager = FakeContextManager()

    agent = AgentLoop(
        client=client,
        registry=registry,
        context_manager=context_manager,
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

    context_manager = FakeContextManager()

    sink = CollectingEventSink()

    agent = AgentLoop(
        client=client,
        registry=registry,
        context_manager=context_manager,
        model="test-model",
        event_sink=sink,
        max_steps=3,
    )

    with pytest.raises(
        RuntimeError,
        match="Agent exceeded max_steps=3",
    ):
        await agent.run(
            user_input="Keep reading the file.",
            thread_id="thread_123",
            turn_id="turn_123",
        )

    assert client.call_count == 3

    assert sink.events[0].type == EventType.TURN_STARTED
    assert sink.events[-1].type == EventType.TURN_FAILED
    assert sink.events[-1].data["reason"] == "max_steps_exceeded"
    assert sink.events[-1].data["max_steps"] == agent.max_steps


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

        context_manager = FakeContextManager()

        agent = AgentLoop(
            client=client,
            registry=registry,
            context_manager=context_manager,
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


@pytest.mark.asyncio
async def test_agent_loop_emits_turn_events(tmp_path: Path):
    file_path = tmp_path / "hello.txt"
    file_path.write_text(
        "hello",
        encoding="utf-8",
    )

    registry = ToolRegistry()
    registry.register(READ_FILE_TOOL)

    client = FakeLLMClient(file_path)

    context_manager = FakeContextManager()

    sink = CollectingEventSink()

    agent = AgentLoop(
        client=client,
        registry=registry,
        context_manager=context_manager,
        model="test-model",
        event_sink=sink,
    )

    result = await agent.run(
        user_input="Read the file.",
        thread_id="thread_123",
        turn_id="turn_123",
    )

    assert result is not None
    assert sink.events[0].type == EventType.TURN_STARTED
    assert sink.events[-1].type == EventType.TURN_COMPLETED

    assert sink.events[0].thread_id == "thread_123"
    assert sink.events[0].turn_id == "turn_123"

    assert sink.events[-1].thread_id == "thread_123"
    assert sink.events[-1].turn_id == "turn_123"


@pytest.mark.asyncio
async def test_run_event_sink_overrides_default_sink(tmp_path: Path):
    default_sink = CollectingEventSink()
    run_sink = CollectingEventSink()

    file_path = tmp_path / "hello.txt"
    file_path.write_text(
        "hello",
        encoding="utf-8",
    )

    registry = ToolRegistry()
    registry.register(READ_FILE_TOOL)

    client = FakeLLMClient(file_path)

    context_manager = FakeContextManager()

    agent = AgentLoop(
        client=client,
        registry=registry,
        context_manager=context_manager,
        model="test-model",
        event_sink=default_sink,
    )

    result = await agent.run(
        user_input="Read the file.",
        thread_id="thread_123",
        turn_id="turn_123",
        event_sink=run_sink,
    )

    assert result is not None
    assert default_sink.events == []
    assert [event.type for event in run_sink.events] == [
        EventType.TURN_STARTED,
        EventType.TOOL_STARTED,
        EventType.TOOL_COMPLETED,
        EventType.AGENT_MESSAGE,
        EventType.TURN_COMPLETED,
    ]


@pytest.mark.asyncio
async def test_run_uses_default_sink_when_no_run_sink(tmp_path: Path):
    default_sink = CollectingEventSink()

    file_path = tmp_path / "hello.txt"
    file_path.write_text(
        "hello",
        encoding="utf-8",
    )

    registry = ToolRegistry()
    registry.register(READ_FILE_TOOL)

    client = FakeLLMClient(file_path)

    context_manager = FakeContextManager()

    agent = AgentLoop(
        client=client,
        registry=registry,
        context_manager=context_manager,
        model="test-model",
        event_sink=default_sink,
    )

    assert default_sink.events == []

    await agent.run(
        user_input="Read the file.",
        thread_id="thread_123",
        turn_id="turn_123",
    )

    assert default_sink.events[0].type == EventType.TURN_STARTED
    assert default_sink.events[-1].type == EventType.TURN_COMPLETED
