from pathlib import Path

from mini_llm.tools.schema import ToolSchema


def search_code(
    query: str,
    path: str = ".",
) -> list[dict[str, str | int]]:
    """
    Search text files recursively for a string.

    Args:
        query: Text to search for.
        path: File or directory to search.

    Returns:
        A list of matches containing path, line number, and line content.

    Raises:
        FileNotFoundError: If the path does not exist.
        ValueError: If query is empty.
    """

    if not query:
        raise ValueError("query must not be empty")

    root = Path(path)

    if not root.exists():
        raise FileNotFoundError(f"Path not found: {path}")

    results: list[dict[str, str | int]] = []

    files = [root] if root.is_file() else root.rglob("*")

    for file_path in files:
        if not file_path.is_file():
            continue

        try:
            content = file_path.read_text(encoding="utf-8")
        except (UnicodeDecodeError, OSError):
            continue

        for line_number, line in enumerate(content.splitlines(), start=1):
            if query in line:
                results.append(
                    {"path": str(file_path), "line": line_number, "content": line}
                )

    return results


SEARCH_CODE_TOOL = ToolSchema(
    name="search_code",
    description="Search text files recursively for a string.",
    parameters={
        "type": "object",
        "properties": {
            "query": {
                "type": "string",
                "description": "Text to search for.",
            },
            "path": {
                "type": "string",
                "description": "File or directory to search.",
                "default": ".",
            },
        },
        "required": ["query"],
        "additionalProperties": False,
    },
    handler=search_code,
)
