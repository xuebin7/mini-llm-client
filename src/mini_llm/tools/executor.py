import inspect
from dataclasses import dataclass
from typing import Any

from mini_llm.tools.registry import ToolRegistry


@dataclass
class ToolResult:
    tool_name: str
    success: bool
    output: Any = None
    error: str | None = None


class ToolExecutor:
    def __init__(self, registry: ToolRegistry) -> None:
        self.registry = registry

    async def execute(
        self,
        tool_name: str,
        arguments: dict[str, Any],
    ) -> ToolResult:

        try:
            tool = self.registry.get(tool_name)
        except KeyError as exc:
            return ToolResult(
                tool_name=tool_name,
                success=False,
                error=str(exc),
            )

        try:
            output = tool.handler(**arguments)

            if inspect.isawaitable(output):
                output = await output

            return ToolResult(
                tool_name=tool_name,
                success=True,
                output=output,
            )

        except Exception as exc:  # noqa: BLE001
            # Tool failures must not crash the agent runtime.
            return ToolResult(
                tool_name=tool_name,
                success=False,
                error=str(exc),
            )
