from typing import Any

from mini_llm.tools.schema import ToolSchema


class ToolRegistry:
    def __init__(self) -> None:
        self._tools: dict[str, ToolSchema] = {}

    def register(self, tool: ToolSchema) -> None:
        if tool.name in self._tools:
            raise ValueError(f"Tool already registered: {tool.name}")

        self._tools[tool.name] = tool

    def get(self, name: str) -> ToolSchema:
        try:
            return self._tools[name]
        except KeyError:
            raise KeyError(f"Unknown tool: {name}") from None

    def list_tools(self) -> list[ToolSchema]:
        return list(self._tools.values())

    def to_openai_tools(self) -> list[dict[str, Any]]:
        return [tool.to_openai_tool() for tool in self._tools.values()]
