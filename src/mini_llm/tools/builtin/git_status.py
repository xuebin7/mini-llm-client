import subprocess
from collections.abc import Callable

from mini_llm.tools.command import CommandResult
from mini_llm.tools.schema import ToolSchema
from mini_llm.workspace import Workspace


def git_status(
    workspace: Workspace,
) -> CommandResult:
    completed = subprocess.run(
        ["git", "status", "--short", "--", "."],
        cwd=workspace.root,
        capture_output=True,
        text=True,
        encoding="utf-8",
        errors="replace",
        check=False,
    )

    return CommandResult(
        returncode=completed.returncode,
        stdout=completed.stdout,
        stderr=completed.stderr,
    )


def create_git_status_handler(
    workspace: Workspace,
) -> Callable[..., CommandResult]:
    def handler() -> CommandResult:
        return git_status(workspace)

    return handler


def create_git_status_tool(
    workspace: Workspace,
) -> ToolSchema:
    return ToolSchema(
        name="git_status",
        description=("Show the current Git working tree status in short format."),
        parameters={
            "type": "object",
            "properties": {},
            "additionalProperties": False,
        },
        handler=create_git_status_handler(workspace),
    )
