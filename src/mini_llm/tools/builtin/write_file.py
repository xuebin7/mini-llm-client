from collections.abc import Callable
from pathlib import Path

from mini_llm.tools.schema import ToolSchema
from mini_llm.workspace import Workspace


def write_file(
    path: str | Path,
    content: str,
) -> None:
    file_path = Path(path)
    file_path.parent.mkdir(parents=True, exist_ok=True)
    file_path.write_text(content, encoding="utf-8")


def create_write_file_handler(
    workspace: Workspace,
) -> Callable[..., None]:
    def handler(
        path: str,
        content: str,
    ) -> None:
        resolved_path = workspace.resolve_path(path)

        write_file(
            resolved_path,
            content,
        )

    return handler


def create_write_file_tool(
    workspace: Workspace,
) -> ToolSchema:
    return ToolSchema(
        name="write_file",
        description="Write UTF-8 text content to a file.",
        parameters={
            "type": "object",
            "properties": {
                "path": {
                    "type": "string",
                    "description": "Path to the file to write.",
                },
                "content": {
                    "type": "string",
                    "description": "UTF-8 text content to write.",
                },
            },
            "required": ["path", "content"],
            "additionalProperties": False,
        },
        handler=create_write_file_handler(workspace),
    )
