from pathlib import Path

from mini_llm.tools.schema import ToolSchema


def read_file(
    path: str,
    start_line: int | None = None,
    end_line: int | None = None,
) -> str:
    """
    Read a UTF-8 text file.

    Line numbers are 1-based and inclusive.

    Args:
        path: Path to the file.
        start_line: Optional starting line number, starting from 1.
        end_line: Optional ending line number, inclusive.

    Returns:
        File contents or the requested line range.

    Raises:
        FileNotFoundError: If the path does not exist.
        ValueError: If the path is not a file or line arguments are invalid.
        UnicodeDecodeError: If the file is not valid UTF-8 text.
    """

    file_path = Path(path)

    if not file_path.exists():
        raise FileNotFoundError(f"File not found: {path}")

    if not file_path.is_file():
        raise ValueError(f"Path is not a file: {path}")

    if start_line is not None and start_line < 1:
        raise ValueError("start_line must be >= 1")

    if end_line is not None and end_line < 1:
        raise ValueError("end_line must be >= 1")

    if start_line is not None and end_line is not None and start_line > end_line:
        raise ValueError("start_line must be <= end_line")

    content = file_path.read_text(encoding="utf-8")

    if start_line is None and end_line is None:
        return content

    lines = content.splitlines(keepends=True)

    start_index = 0 if start_line is None else start_line - 1
    end_index = len(lines) if end_line is None else end_line

    return "".join(lines[start_index:end_index])


READ_FILE_TOOL = ToolSchema(
    name="read_file",
    description=(
        "Read a UTF-8 text file.Optionally read a specific inclusive line range."
    ),
    parameters={
        "type": "object",
        "properties": {
            "path": {
                "type": "string",
                "description": "Path to the file to read.",
            },
            "start_line": {
                "type": "integer",
                "minimum": 1,
                "description": "Optional 1-based starting line number.",
            },
            "end_line": {
                "type": "integer",
                "minimum": 1,
                "description": "Optional 1-based ending line number, inclusive.",
            },
        },
        "required": ["path"],
        "additionalProperties": False,
    },
    handler=read_file,
)
