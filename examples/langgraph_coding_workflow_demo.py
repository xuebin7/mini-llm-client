import difflib
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


class CodePatch(BaseModel):
    old_text: str
    new_text: str


patch_model = model.with_structured_output(CodePatch)


class CodeReview(BaseModel):
    approved: bool
    reason: str


review_model = model.with_structured_output(CodeReview)


class CodingState(TypedDict):
    file_path: str
    file_content: str
    test_target: str
    test_returncode: int
    test_output: str
    attempts: int
    old_text: str
    new_text: str
    code_diff: str
    review_approved: bool
    review_result: str
    original_file_content: str


async def read_file(state: CodingState) -> CodingState:
    result = await executor.execute(
        tool_name="read_file",
        arguments={
            "path": state["file_path"],
        },
    )

    if not result.success:
        raise RuntimeError(result.error or "read_file tool failed")

    file_content = str(result.output)

    return {
        **state,
        "file_content": file_content,
        "original_file_content": (state["original_file_content"] or file_content),
    }


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
) -> Literal["success", "failed"]:
    if state["review_approved"]:
        return "success"

    return "failed"


def success(state: CodingState) -> CodingState:
    print("Tests passed:")
    print(state["test_output"])

    return state


async def fix_code(state: CodingState) -> CodingState:
    prompt = f"""You are fixing a Python file.
    
Current file:
    
{state["file_content"]}

Test Result:

{state["test_output"]}

Identify the smallest exact text replacement needed to fix the bug.

old_text must exactly match text that currently exists in the file.
new_text must contain the corrected replacement.
Do not rewrite unrelated code.
"""

    raw_patch = await patch_model.ainvoke(prompt)

    if isinstance(raw_patch, CodePatch):
        patch = raw_patch
    else:
        patch = CodePatch.model_validate(raw_patch)

    result = await executor.execute(
        tool_name="apply_patch",
        arguments={
            "path": state["file_path"],
            "old_text": patch.old_text,
            "new_text": patch.new_text,
        },
    )

    if not result.success:
        raise RuntimeError(result.error or "apply patch tool failed")

    return {
        **state,
        "old_text": patch.old_text,
        "new_text": patch.new_text,
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
    prompt = f"""You are reviewing a code change.
    
Test result:

{state["test_output"]}

Code diff:

{state["code_diff"]}

Review whether this change is correct, minimal, and relevant to the bug.

Approve the change only if:
1. The tests pass.
2. The diff fixes the intended bug.
3. The change does not contain unrelated modifications.
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
    diff = "".join(
        difflib.unified_diff(
            state["original_file_content"].splitlines(
                keepends=True,
            ),
            state["file_content"].splitlines(
                keepends=True,
            ),
            fromfile=state["file_path"] + " (before)",
            tofile=state["file_path"] + " (after)",
        )
    )

    return {
        **state,
        "code_diff": diff,
    }


async def main() -> None:
    builder = StateGraph(CodingState)

    builder.add_node(
        "read_file",
        read_file,
    )

    builder.add_node(
        "run_test",
        run_test,
    )

    builder.add_node(
        "fix_code",
        fix_code,
    )

    builder.add_node(
        "success",
        success,
    )

    builder.add_node(
        "failed",
        failed,
    )

    builder.add_node(
        "review_code",
        review_code,
    )

    builder.add_node(
        "build_agent_diff",
        build_agent_diff,
    )

    builder.add_edge(
        START,
        "read_file",
    )

    builder.add_edge(
        "read_file",
        "run_test",
    )

    builder.add_conditional_edges(
        "run_test",
        route_after_test,
    )

    builder.add_edge(
        "fix_code",
        "read_file",
    )

    builder.add_edge(
        "build_agent_diff",
        "review_code",
    )

    builder.add_conditional_edges(
        "review_code",
        route_after_review,
    )

    builder.add_edge(
        "success",
        END,
    )

    builder.add_edge(
        "failed",
        END,
    )

    graph = builder.compile()

    result = await graph.ainvoke(
        {
            "file_path": "examples/coding_agent_demo/calculator.py",
            "test_target": "examples/coding_agent_demo/test_calculator.py",
            "file_content": "",
            "test_returncode": -1,
            "test_output": "",
            "attempts": 0,
            "old_text": "",
            "new_text": "",
            "code_diff": "",
            "review_approved": False,
            "review_result": "",
            "original_file_content": "",
        }
    )

    print(result)


if __name__ == "__main__":
    import asyncio

    asyncio.run(main())
