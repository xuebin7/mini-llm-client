import json
from pathlib import Path

import pytest

from mini_llm.tools.builtin.read_file import create_read_file_tool
from mini_llm.tools.builtin.search_code import create_search_code_tool
from mini_llm.tools.executor import ToolExecutor
from mini_llm.tools.registry import ToolRegistry
from mini_llm.workspace import Workspace


def test_builtin_tools_are_json_serializable(tmp_path: Path):
    registry = ToolRegistry()
    registry.register(create_read_file_tool(Workspace(tmp_path)))
    registry.register(create_search_code_tool(Workspace(tmp_path)))

    tools = registry.to_openai_tools()

    assert json.loads(json.dumps(tools)) == tools


def test_builtin_tools_use_responses_api_format(tmp_path: Path):
    registry = ToolRegistry()
    registry.register(create_read_file_tool(Workspace(tmp_path)))
    registry.register(create_search_code_tool(Workspace(tmp_path)))

    tools = registry.to_openai_tools()

    assert len(tools) == 2

    for exported, tool in zip(
        tools,
        (
            create_read_file_tool(Workspace(tmp_path)),
            create_search_code_tool(Workspace(tmp_path)),
        ),
        strict=True,
    ):
        assert set(exported) == {
            "type",
            "name",
            "description",
            "parameters",
        }

        assert exported["type"] == "function"
        assert exported["name"] == tool.name
        assert exported["description"] == tool.description
        assert exported["parameters"] == tool.parameters
        assert exported["parameters"]["type"] == "object"


@pytest.mark.asyncio
async def test_execute_search_code(tmp_path: Path):
    file_path = tmp_path / "example.py"
    file_path.write_text("class ToolExecutor:\n    pass\n", encoding="utf-8")

    registry = ToolRegistry()
    registry.register(create_search_code_tool(Workspace(tmp_path)))

    executor = ToolExecutor(registry)

    result = await executor.execute(
        "search_code",
        {
            "query": "ToolExecutor",
            "path": str(tmp_path),
        },
    )

    assert result.success is True
    assert result.error is None
    assert len(result.output) == 1
    assert result.output[0]["path"] == str(file_path.relative_to(tmp_path))
    assert result.output[0]["line"] == 1


@pytest.mark.asyncio
async def test_execute_read_file(tmp_path: Path):
    file_path = tmp_path / "hello.txt"
    file_path.write_text("hello\nworld\n", encoding="utf-8")

    registry = ToolRegistry()
    registry.register(create_read_file_tool(Workspace(tmp_path)))

    executor = ToolExecutor(registry=registry)

    result = await executor.execute(
        "read_file",
        {"path": str(file_path), "start_line": 2, "end_line": 2},
    )

    assert result.success is True
    assert result.error is None
    assert result.output == "world\n"


@pytest.mark.asyncio
async def test_read_file_failure(tmp_path: Path):
    registry = ToolRegistry()
    registry.register(create_read_file_tool(Workspace(tmp_path)))

    executor = ToolExecutor(registry=registry)

    result = await executor.execute(
        "read_file", {"path": str(tmp_path / "missing.txt")}
    )

    assert result.success is False
    assert result.output is None
    assert result.error is not None
    assert "File not found" in result.error


@pytest.mark.asyncio
async def test_execute_unknown_tool():
    registry = ToolRegistry()
    executor = ToolExecutor(registry=registry)

    result = await executor.execute(
        "unknown tool",
        {},
    )

    assert result.success is False
    assert result.output is None
    assert result.error is not None
    assert "Unknown tool" in result.error
