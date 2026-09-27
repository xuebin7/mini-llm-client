import json
from pathlib import Path

import pytest

from mini_llm.agent.runtime import execute_tool_call, tool_result_to_output
from mini_llm.models import ToolCall
from mini_llm.tools.builtin.read_file import READ_FILE_TOOL
from mini_llm.tools.executor import ToolExecutor, ToolResult
from mini_llm.tools.registry import ToolRegistry


@pytest.mark.asyncio
async def test_execute_tool_call(tmp_path: Path):
    file_path = tmp_path / "hello.txt"
    file_path.write_text(
        "hello\nworld\n",
        encoding="utf-8",
    )

    registry = ToolRegistry()
    registry.register(READ_FILE_TOOL)

    executor = ToolExecutor(registry=registry)

    tool_call = ToolCall(
        call_id="call_123",
        name="read_file",
        arguments=json.dumps(
            {
                "path": str(file_path),
                "start_line": 2,
                "end_line": 2,
            }
        ),
    )

    result = await execute_tool_call(
        tool_call=tool_call,
        executor=executor,
    )

    assert result.success is True
    assert result.error is None
    assert result.output == "world\n"


@pytest.mark.asyncio
async def test_execute_tool_call_invalid_json():
    registry = ToolRegistry()
    registry.register(READ_FILE_TOOL)

    executor = ToolExecutor(registry=registry)

    tool_call = ToolCall(
        call_id="call_123",
        name="read_file",
        arguments="not valid json",
    )

    result = await execute_tool_call(
        tool_call,
        executor,
    )

    assert result.success is False
    assert result.error is not None
    assert "Invalid tool arguments JSON" in result.error


@pytest.mark.asyncio
async def test_execute_tool_call_arguments_must_be_object():
    registry = ToolRegistry()
    registry.register(READ_FILE_TOOL)

    executor = ToolExecutor(registry=registry)

    tool_call = ToolCall(
        call_id="call_123",
        name="read_file",
        arguments='["hello", "world"]',
    )

    result = await execute_tool_call(
        tool_call,
        executor,
    )

    assert result.success is False
    assert result.error == "Tool arguments must be a JSON object"


def test_tool_result_to_output_success():
    tool_call = ToolCall(
        call_id="call_123",
        name="read_file",
        arguments='{"path": "hello.txt"}',
    )

    result = ToolResult(
        tool_name="read_file",
        success=True,
        output="hello world",
    )

    output = tool_result_to_output(
        tool_call=tool_call,
        result=result,
    )

    assert output["type"] == "function_call_output"
    assert output["call_id"] == "call_123"

    content = json.loads(output["output"])

    assert content["success"] is True
    assert content["output"] == "hello world"


def test_tool_result_to_output_failure():
    tool_call = ToolCall(
        call_id="call_123",
        name="read_file",
        arguments='{"path": "missing.txt"}',
    )

    result = ToolResult(
        tool_name="read_file",
        success=False,
        error="File not found: missing.txt",
    )

    output = tool_result_to_output(
        tool_call=tool_call,
        result=result,
    )

    content = json.loads(output["output"])

    assert output["type"] == "function_call_output"
    assert output["call_id"] == "call_123"
    assert content["success"] is False
    assert content["error"] == "File not found: missing.txt"
