import subprocess
from collections.abc import Callable

from mini_llm.tools.command import CommandResult
from mini_llm.tools.schema import ToolSchema
from mini_llm.workspace import Workspace


def git_diff(
    workspace: Workspace,
    path: str | None = None,
) -> CommandResult:
    target = path or "."

    completed = subprocess.run(
        ["git", "diff", "--", target],
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


def create_git_diff_handler(
    workspace: Workspace,
) -> Callable[..., CommandResult]:
    def handler(
        path: str | None = None,
    ) -> CommandResult:
        return git_diff(
            workspace,
            path=path,
        )

    return handler


def create_git_diff_tool(
    workspace: Workspace,
) -> ToolSchema:
    return ToolSchema(
        name="git_diff",
        description=("Show the current unstaged Git diff in the workspace."),
        parameters={
            "type": "object",
            "properties": {
                "path": {
                    "type": "string",
                    "description": (
                        "Optional relative file or directory path to limit the diff."
                    ),
                },
            },
            "additionalProperties": False,
        },
        handler=create_git_diff_handler(workspace),
    )
