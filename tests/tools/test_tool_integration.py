import json
from pathlib import Path

from mini_llm.tools.builtin.read_file import READ_FILE_TOOL
from mini_llm.tools.builtin.search_code import SEARCH_CODE_TOOL
from mini_llm.tools.executor import ToolExecutor
from mini_llm.tools.registry import ToolRegistry


def test_builtin_tools_are_json_serializable():
    registry = ToolRegistry()
    registry.register(READ_FILE_TOOL)
    registry.register(SEARCH_CODE_TOOL)

    tools = registry.openai_tools()

    assert json.loads(json.dumps(tools)) == tools


def test_builtin_tools_use_chat_completions_format():
    registry = ToolRegistry()
    registry.register(READ_FILE_TOOL)
    registry.register(SEARCH_CODE_TOOL)

    tools = registry.openai_tools()

    assert len(tools) == 2
    for exported, tool in zip(tools, (READ_FILE_TOOL, SEARCH_CODE_TOOL), strict=True):
        assert set(exported) == {"type", "function"}
        assert exported["type"] == "function"
        function = exported["function"]
        assert set(function) == {"name", "description", "parameters"}
        assert function["name"] == tool.name
        assert function["description"] == tool.description
        assert function["parameters"] == tool.parameters
        assert function["parameters"]["type"] == "object"


def test_execute_search_code(tmp_path: Path):
    file_path = tmp_path / "example.py"
    file_path.write_text("class ToolExecutor:\n    pass\n", encoding="utf-8")

    registry = ToolRegistry()
    registry.register(SEARCH_CODE_TOOL)

    executor = ToolExecutor(registry)

    result = executor.execute(
        "search_code",
        {
            "query": "ToolExecutor",
            "path": str(tmp_path),
        },
    )

    assert result.success is True
    assert result.error is None
    assert len(result.output) == 1
    assert result.output[0]["path"] == str(file_path)
    assert result.output[0]["line"] == 1


def test_execute_read_file(tmp_path: Path):
    file_path = tmp_path / "hello.txt"
    file_path.write_text("hello\nworld\n", encoding="utf-8")

    registry = ToolRegistry()
    registry.register(READ_FILE_TOOL)

    executor = ToolExecutor(registry=registry)

    result = executor.execute(
        "read_file", {"path": str(file_path), "start_line": 2, "end_line": 2}
    )

    assert result.success is True
    assert result.error is None
    assert result.output == "world\n"


def test_read_file_failure(tmp_path: Path):
    registry = ToolRegistry()
    registry.register(READ_FILE_TOOL)

    executor = ToolExecutor(registry=registry)

    result = executor.execute("read_file", {"path": str(tmp_path / "missing.txt")})

    assert result.success is False
    assert result.output is None
    assert result.error is not None
    assert "File not found" in result.error


def test_execute_unknown_tool():
    registry = ToolRegistry()
    executor = ToolExecutor(registry=registry)

    result = executor.execute(
        "unknown tool",
        {},
    )

    assert result.success is False
    assert result.output is None
    assert result.error is not None
    assert "Unknown tool" in result.error
