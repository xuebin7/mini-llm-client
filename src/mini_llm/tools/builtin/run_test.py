import subprocess
from collections.abc import Callable

from mini_llm.tools.command import CommandResult
from mini_llm.tools.schema import ToolSchema
from mini_llm.workspace import Workspace


def run_test(
    workspace: Workspace,
    target: str | None = None,
    timeout: int = 60,
) -> CommandResult:
    command = [
        "uv",
        "run",
        "pytest",
    ]

    if target is not None:
        command.append(target)

    command.append("-q")

    completed = subprocess.run(
        command,
        cwd=workspace.root,
        capture_output=True,
        text=True,
        timeout=timeout,
        encoding="utf-8",
        errors="replace",
        check=False,
    )

    return CommandResult(
        returncode=completed.returncode,
        stdout=completed.stdout,
        stderr=completed.stderr,
    )


def create_run_test_handler(
    workspace: Workspace,
) -> Callable[..., CommandResult]:
    def handler(
        target: str | None = None,
        timeout: int = 60,
    ) -> CommandResult:
        return run_test(
            workspace=workspace,
            target=target,
            timeout=timeout,
        )

    return handler


def create_run_test_tool(
    workspace: Workspace,
) -> ToolSchema:
    return ToolSchema(
        name="run_test",
        description=(
            "Run pytest in the current workspace."
            "Optionally run a specific test file or test target."
        ),
        parameters={
            "type": "object",
            "properties": {
                "target": {
                    "type": "string",
                    "description": (
                        "Optional pytest target, such as "
                        "'tests/test_demo.py' or "
                        "'tests/test_demp.py::test_example'."
                    ),
                }
            },
            "timeout": {
                "type": "integer",
                "minimum": 1,
                "description": "Timeout in seconds.",
            },
            "additionalProperties": False,
        },
        handler=create_run_test_handler(workspace),
    )
