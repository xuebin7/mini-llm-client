from pathlib import Path

from mini_llm.tools.schema import ToolSchema

DEFAULT_EXCLUDE_DIRS = [
    ".git",
    ".pytest_cache",
    ".venv",
    "__pycache__",
]


def search_code(
    query: str,
    path: str = ".",
    max_results: int = 50,
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

    max_results: int = 50

    for file_path in files:
        if any(part in DEFAULT_EXCLUDE_DIRS for part in file_path.parts):
            continue

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

                if len(results) >= max_results:
                    return results

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
            "max_results": {
                "type": "integer",
                "minimum": 1,
                "maximum": 100,
                "default": 50,
                "description": "Maximum number of matches to return.",
            },
        },
        "required": ["query"],
        "additionalProperties": False,
    },
    handler=search_code,
)
