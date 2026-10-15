import subprocess
from collections.abc import Callable
from pathlib import Path

from mini_llm.tools.schema import ToolSchema
from mini_llm.workspace import Workspace


def git_log(
    path: str | Path = ".",
    max_count: int = 10,
) -> list[dict[str, str]]:
    """
    Read recent Git commit history.

    Args:
        path: Path to a Git repository or a directory inside one.
        max_count: Maximum number of commits to return.

    Returns:
        A list of commits contains hash, author, date and subject.

    Raises:
        FileNotFoundError: If the path does not exists.
        ValueError: If max_count is less than 1.
        RuntimeError: If Git_execution fails.
    """

    repo_path = Path(path)

    if not repo_path.exists():
        raise FileNotFoundError(f"Path not found: {path}")

    if max_count < 1:
        raise ValueError("max_count must be >= 1")

    command = [
        "git",
        "-C",
        str(repo_path),
        "log",
        f"--max-count={max_count}",
        "--pretty=format:%H%x1f%an%x1f%aI%x1f%s",
        "--",
        ".",
    ]

    try:
        result = subprocess.run(
            command,
            capture_output=True,
            text=True,
            encoding="utf-8",
            errors="replace",
            check=False,
            timeout=10,
        )
    except FileNotFoundError as exc:
        raise RuntimeError("Git executable not found") from exc
    except subprocess.TimeoutExpired as exc:
        raise RuntimeError("Git command timed out") from exc

    if result.returncode != 0:
        error = result.stderr.strip() or "Git command failed"
        raise RuntimeError(error)

    commits: list[dict[str, str]] = []

    if not result.stdout.strip():
        return commits

    # Git separates records with LF; Unicode separators can occur in subjects.
    for line in result.stdout.split("\n"):
        commit_hash, author, date, subject = line.split("\x1f", maxsplit=3)

        commits.append(
            {"hash": commit_hash, "author": author, "date": date, "subject": subject}
        )

    return commits


def create_git_log_handler(
    workspace: Workspace,
) -> Callable[..., list[dict[str, str]]]:
    def handler(
        path: str = ".",
        max_count: int = 10,
    ) -> list[dict[str, str]]:
        resolved_path = workspace.resolve_path(path)

        return git_log(
            path=resolved_path,
            max_count=max_count,
        )

    return handler


def create_git_log_tool(
    workspace: Workspace,
) -> ToolSchema:
    return ToolSchema(
        name="git_log",
        description=(
            "Read recent Git commit history affecting files "
            "within the current workspace."
        ),
        parameters={
            "type": "object",
            "properties": {
                "path": {
                    "type": "string",
                    "description": (
                        "File or directory within the workspace "
                        "Defaults to the workspace root."
                    ),
                    "default": ".",
                },
                "max_count": {
                    "type": "integer",
                    "minimum": 1,
                    "description": "Maximum number of commits to return.",
                    "default": 10,
                },
            },
            "additionalProperties": False,
        },
        handler=create_git_log_handler(workspace),
    )
