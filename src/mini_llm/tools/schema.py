from collections.abc import Callable
from dataclasses import dataclass
from typing import Any

ToolHandler = Callable[..., Any]


@dataclass(frozen=True)
class ToolSchema:
    name: str
    description: str
    parameters: dict[str, Any]
    handler: ToolHandler

    def to_openai_tool(self) -> dict[str, Any]:
        """Export a function tool for the Chat Completions API."""
        return {
            "type": "function",
            "function": {
                "name": self.name,
                "description": self.description,
                "parameters": self.parameters,
            },
        }
