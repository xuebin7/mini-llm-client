from collections.abc import Callable
from pathlib import Path

from mini_llm.tools.schema import ToolSchema
from mini_llm.workspace import Workspace


def apply_patch(
    path: str | Path,
    old_text: str,
    new_text: str,
) -> None:
    file_path = Path(path)

    if not file_path.exists():
        raise FileNotFoundError(f"File not found: {path}")

    if not file_path.is_file():
        raise ValueError(f"Path is not a file: {path}")

    content = file_path.read_text(encoding="utf-8")

    if not old_text:
        raise ValueError("old_text must not be empty")

    count = content.count(old_text)

    if count == 0:
        raise ValueError("old_text not found in file")

    if count > 1:
        raise ValueError("old_text matched multiple locations")

    updated = content.replace(
        old_text,
        new_text,
        1,
    )

    file_path.write_text(updated, encoding="utf-8")


def create_apply_patch_handler(
    workspace: Workspace,
) -> Callable[..., None]:

    def handler(
        path: str,
        old_text: str,
        new_text: str,
    ) -> None:
        resolved_path = workspace.resolve_path(path)

        apply_patch(
            resolved_path,
            old_text,
            new_text,
        )

    return handler


def create_apply_patch_tool(
    workspace: Workspace,
) -> ToolSchema:
    return ToolSchema(
        name="apply_patch",
        description="Replace exactly one occurrence of old_text with new_text in a UTF-8 text file.",
        parameters={
            "type": "object",
            "properties": {
                "path": {
                    "type": "string",
                    "description": "Path to the file to apply patch.",
                },
                "old_text": {
                    "type": "string",
                    "description": (
                        "Exact existing text to replace. "
                        "It must occur exactly once in the file."
                    ),
                },
                "new_text": {
                    "type": "string",
                    "description": "Replacement text.",
                },
            },
            "required": ["path", "old_text", "new_text"],
            "additionalProperties": False,
        },
        handler=create_apply_patch_handler(workspace),
    )
