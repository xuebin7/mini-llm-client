from mini_llm.tools.defaults import create_default_registry


def test_default_registry_contains_builtin_tools():
    registry = create_default_registry()

    tools = registry.to_openai_tools()

    names = [tool["name"] for tool in tools]

    assert "read_file" in names
    assert "search_code" in names
    assert "git_log" in names


def test_openai_tools_do_not_include_handler():
    registry = create_default_registry()

    tools = registry.to_openai_tools()

    for tool in tools:
        assert "handler" not in tool
        assert "name" in tool
        assert "description" in tool
        assert "parameters" in tool
        assert tool["type"] == "function"
