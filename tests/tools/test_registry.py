from pathlib import Path

from mini_llm.tools.defaults import create_default_registry
from mini_llm.workspace import Workspace


def test_default_registry_contains_builtin_tools(tmp_path: Path):
    registry = create_default_registry(Workspace(tmp_path))

    tools = registry.to_openai_tools()

    names = [tool["name"] for tool in tools]

    assert "read_file" in names
    assert "write_file" in names
    assert "search_code" in names
    assert "git_log" in names


def test_openai_tools_do_not_include_handler(tmp_path: Path):
    registry = create_default_registry(Workspace(tmp_path))

    tools = registry.to_openai_tools()

    for tool in tools:
        assert "handler" not in tool
        assert "name" in tool
        assert "description" in tool
        assert "parameters" in tool
        assert tool["type"] == "function"
