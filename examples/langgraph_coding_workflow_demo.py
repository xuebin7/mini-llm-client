import difflib
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Literal, TypedDict

from langchain_deepseek import ChatDeepSeek
from langgraph.graph import END, START, StateGraph
from pydantic import BaseModel

from mini_llm.tools.defaults import create_default_registry
from mini_llm.tools.executor import ToolExecutor
from mini_llm.workspace import Workspace

project_root = Path(__file__).resolve().parent.parent
workspace = Workspace(project_root)
registry = create_default_registry(workspace)
executor = ToolExecutor(registry)

model = ChatDeepSeek(
    model="deepseek-flash",
    temperature=0,
    extra_body={
        "thinking": {
            "type": "disabled",
        }
    },
)

MAX_ATTEMPTS = 3

CALCULATOR_PATH = "examples/coding_agent_demo/calculator.py"
TEST_PATH = "examples/coding_agent_demo/test_calculator.py"

INITIAL_CALCULATOR = """def divide(a: float, b: float) -> float:
    return a * b
"""

INITIAL_TEST = """from calculator import divide


def test_divide():
    assert divide(10, 2) == 5


def test_divide_fraction():
    assert divide(9, 3) == 3
"""

STRING_UTILS_PATH = "examples/coding_agent_demo/string_utils.py"
STRING_TEST_PATH = "examples/coding_agent_demo/test_string_utils.py"

INITIAL_STRING_UTILS = """def is_palindrome(text: str) -> bool:
    return False
"""

INITIAL_STRING_TEST = """from string_utils import is_palindrome


def test_palindrome():
    assert is_palindrome("level") is True


def test_not_palindrome():
    assert is_palindrome("hello") is False
"""


@dataclass
class CodingTask:
    description: str
    files: dict[str, str]
    test_target: str


DIVIDE_TASK = CodingTask(
    description=(
        "Fix divide so it performs division correctly, "
        "and add a test verifying that division by zero "
        "raises ZeroDivisionError. "
        "Prefer the smallest implementation change."
    ),
    files={
        CALCULATOR_PATH: INITIAL_CALCULATOR,
        TEST_PATH: INITIAL_TEST,
    },
    test_target=TEST_PATH,
)

PALINDROME_TASK = CodingTask(
    description=(
        "Implement is_palindrome so that it correctly detects "
        "whether a string is a palindrome. "
        "Prefer the smallest implementation change."
    ),
    files={
        STRING_UTILS_PATH: INITIAL_STRING_UTILS,
        STRING_TEST_PATH: INITIAL_STRING_TEST,
    },
    test_target=STRING_TEST_PATH,
)

CODING_TASKS: dict[str, CodingTask] = {
    "divide": DIVIDE_TASK,
    "palindrome": PALINDROME_TASK,
}


def reset_demo_files(task: CodingTask) -> None:
    for path, content in task.files.items():
        file_path = project_root / path
        file_path.write_text(
            content,
            encoding="utf-8",
        )


class FilePatch(BaseModel):
    file_path: str
    old_text: str
    new_text: str


class CodePatchSet(BaseModel):
    patches: list[FilePatch]


patch_model = model.with_structured_output(CodePatchSet)


class CodeReview(BaseModel):
    approved: bool
    reason: str


review_model = model.with_structured_output(CodeReview)


class CodingState(TypedDict):
    task: str
    source_files: list[str]
    files: dict[str, str]
    original_files: dict[str, str]
    test_target: str
    test_returncode: int
    test_output: str
    attempts: int
    applied_patches: list[dict[str, str]]
    code_diff: str
    review_approved: bool | None
    review_result: str


async def read_files(state: CodingState) -> CodingState:
    files: dict[str, str] = {}

    for path in state["source_files"]:
        result = await executor.execute(
            tool_name="read_file",
            arguments={
                "path": path,
            },
        )

        if not result.success:
            raise RuntimeError(result.error or f"read_file failed: {path}")

        files[path] = str(result.output)

    return {
        **state,
        "files": files,
        "original_files": (state["original_files"] or files.copy()),
    }


def format_files(
    files: dict[str, str],
) -> str:
    parts: list[str] = []

    for path, content in files.items():
        parts.append(f"Path: {path}\n\nContent:\n{content}")

    return "\n\n---\n\n".join(parts)


async def run_test(
    state: CodingState,
) -> CodingState:
    result = await executor.execute(
        tool_name="run_test",
        arguments={
            "target": state["test_target"],
        },
    )

    if not result.success:
        return {
            **state,
            "test_returncode": -1,
            "test_output": result.error or "run_test tool failed",
        }

    command_result = result.output

    output = "\n".join(
        part
        for part in [
            command_result.stdout,
            command_result.stderr,
        ]
        if part
    )

    return {
        **state,
        "test_returncode": command_result.returncode,
        "test_output": output,
    }


def route_after_test(
    state: CodingState,
) -> Literal[
    "success",
    "build_agent_diff",
    "fix_code",
    "failed",
]:
    if state["test_returncode"] == 0:
        if state["attempts"] == 0:
            return "success"

        return "build_agent_diff"

    if state["attempts"] >= MAX_ATTEMPTS:
        return "failed"

    return "fix_code"


def route_after_review(
    state: CodingState,
) -> Literal["success", "fix_code", "failed"]:
    if state["review_approved"] is True:
        return "success"

    if state["attempts"] >= MAX_ATTEMPTS:
        return "failed"

    return "fix_code"


def success(state: CodingState) -> CodingState:
    print("Coding workflow succeeded.")
    print(state["test_output"])

    if state["review_result"]:
        print(state["review_result"])

    return state


async def fix_code(state: CodingState) -> CodingState:
    files_text = format_files(state["files"])

    prompt = f"""You are fixing a Python project.

Task:

{state["task"]}
    
Available files:

{files_text}

Test result:

{state["test_output"]}

Review feedback:

{state["review_result"] or "No review feedback yet."}

Identify the smallest exact text replacement needed.

Return:
- patches: a list of file patches
- each patch must contain:
    - file_path: exactly one of the file paths shown above
    - old_text: exact text that currently exists in that file
    - new_text: corrected replacement

Use as few patches as possible.
Do not modify unrelated code.
"""

    raw_patch_set = await patch_model.ainvoke(prompt)

    if isinstance(raw_patch_set, CodePatchSet):
        patch_set = raw_patch_set
    else:
        patch_set = CodePatchSet.model_validate(raw_patch_set)

    if not patch_set.patches:
        raise RuntimeError("LLM returned no patches")

    if len(patch_set.patches) > 3:
        raise RuntimeError(f"Too many patches: {len(patch_set.patches)}")

    for patch in patch_set.patches:
        if patch.file_path not in state["files"]:
            raise RuntimeError(f"Unexpected patch file: {patch.file_path}")

        result = await executor.execute(
            tool_name="apply_patch",
            arguments={
                "path": patch.file_path,
                "old_text": patch.old_text,
                "new_text": patch.new_text,
            },
        )

        if not result.success:
            raise RuntimeError(result.error or "apply patch tool failed")

    new_patches = [
        {
            "target_file": patch.file_path,
            "old_text": patch.old_text,
            "new_text": patch.new_text,
        }
        for patch in patch_set.patches
    ]

    return {
        **state,
        "applied_patches": state["applied_patches"] + new_patches,
        "attempts": state["attempts"] + 1,
    }


def failed(state: CodingState) -> CodingState:
    print("Coding workflow failed.")

    if state["test_returncode"] != 0:
        print(state["test_output"])

    if state["review_result"]:
        print(state["review_result"])

    return state


async def review_code(
    state: CodingState,
) -> CodingState:
    files_text = format_files(state["files"])

    prompt = f"""You are reviewing a code change.
    
Task:

{state["task"]}

Current project files:

{files_text}

Test result:

{state["test_output"]}

Code diff:

{state["code_diff"]}

Review whether the final project state correctly completes the task
and whether the implementation is minimal.

Approve the change only if:
1. The requested task is fully satisfied by the final project state.
2. The tests pass.
3. The diff is relevant to the task.
4. The change is minimal.
5. The implementation does not add unnecessary or redundant logic.
6. The change does not contain unrelated modifications.

A requirement may already have been satisfied before this coding pass.
Do not require it to appear in the diff if the current project state
already satisfies it.

Reject the change if there is a simpler implementation
that satisfies the task and passes the tests.

If rejecting the change, explain exactly what should be simplified
or corrected so that another coding pass can address it.
"""

    raw_review = await review_model.ainvoke(prompt)

    if isinstance(raw_review, CodeReview):
        review = raw_review
    else:
        review = CodeReview.model_validate(raw_review)

    return {
        **state,
        "review_approved": review.approved,
        "review_result": review.reason,
    }


def build_agent_diff(
    state: CodingState,
) -> CodingState:
    diffs: list[str] = []

    for path, current_content in state["files"].items():
        original_content = state["original_files"].get(
            path,
            "",
        )

        diff = "".join(
            difflib.unified_diff(
                original_content.splitlines(
                    keepends=True,
                ),
                current_content.splitlines(
                    keepends=True,
                ),
                fromfile=path + " (before)",
                tofile=path + " (after)",
            )
        )

        if diff:
            diffs.append(diff)

    return {
        **state,
        "code_diff": "\n".join(diffs),
    }


def create_coding_graph():
    builder = StateGraph(CodingState)

    builder.add_node("read_files", read_files)
    builder.add_node("run_test", run_test)
    builder.add_node("fix_code", fix_code)
    builder.add_node("success", success)
    builder.add_node("failed", failed)
    builder.add_node("review_code", review_code)
    builder.add_node("build_agent_diff", build_agent_diff)

    builder.add_edge(START, "read_files")
    builder.add_edge("read_files", "run_test")

    builder.add_conditional_edges(
        "run_test",
        route_after_test,
    )

    builder.add_edge(
        "fix_code",
        "read_files",
    )

    builder.add_edge(
        "build_agent_diff",
        "review_code",
    )

    builder.add_conditional_edges(
        "review_code",
        route_after_review,
    )

    builder.add_edge("success", END)
    builder.add_edge("failed", END)

    return builder.compile()


async def main() -> None:
    task_name = sys.argv[1] if len(sys.argv) > 1 else "divide"

    if task_name not in CODING_TASKS:
        raise ValueError(
            f"Unknown task: {task_name}. Available tasks: {', '.join(CODING_TASKS)}"
        )
    task = CODING_TASKS[task_name]

    reset_demo_files(task)

    graph = create_coding_graph()

    result = await graph.ainvoke(
        {
            "task": task.description,
            "source_files": list(task.files.keys()),
            "files": {},
            "original_files": {},
            "test_target": task.test_target,
            "test_returncode": -1,
            "test_output": "",
            "attempts": 0,
            "applied_patches": [],
            "code_diff": "",
            "review_approved": None,
            "review_result": "",
        }
    )

    print(result)


if __name__ == "__main__":
    import asyncio

    asyncio.run(main())
