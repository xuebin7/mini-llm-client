import subprocess
from pathlib import Path

from mini_llm.tools.schema import ToolSchema


def git_log(
    path: str = ".",
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
    ]

    try:
        result = subprocess.run(
            command,
            capture_output=True,
            text=True,
            encoding="utf-8",
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


GIT_LOG_TOOL = ToolSchema(
    name="git_log",
    description="Read recent Git commit history from a repository",
    parameters={
        "type": "object",
        "properties": {
            "path": {
                "type": "string",
                "description": "Path to a Git repository or directory inside one.",
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
    handler=git_log,
)
