from examples.langgraph_coding_workflow_demo import (
    CODING_TASKS,
    CodingState,
    project_root,
    reset_demo_files,
    route_after_review,
    route_after_test,
)


def test_coding_tasks_registered():
    assert "divide" in CODING_TASKS
    assert "palindrome" in CODING_TASKS


def make_state() -> CodingState:
    return {
        "task": "",
        "source_files": [],
        "files": {},
        "original_files": {},
        "test_target": "",
        "test_returncode": -1,
        "test_output": "",
        "attempts": 0,
        "applied_patches": [],
        "code_diff": "",
        "review_approved": None,
        "review_result": "",
    }


def test_route_after_test_initial_success():
    state = make_state()
    state["test_returncode"] = 0

    assert route_after_test(state) == "success"


def test_route_after_test_after_fix():
    state = make_state()
    state["test_returncode"] = 0
    state["attempts"] = 1

    assert route_after_test(state) == "build_agent_diff"


def test_route_after_test_failure_goes_to_fix():
    state = make_state()
    state["test_returncode"] = 1

    assert route_after_test(state) == "fix_code"


def test_route_after_test_max_attempts():
    state = make_state()
    state["test_returncode"] = 1
    state["attempts"] = 3

    assert route_after_test(state) == "failed"


def test_route_after_review_approved():
    state = make_state()
    state["review_approved"] = True

    assert route_after_review(state) == "success"


def test_route_after_review_rejected():
    state = make_state()
    state["review_approved"] = False

    assert route_after_test(state) == "fix_code"


def test_route_after_review_rejected_max_attempts():
    state = make_state()
    state["review_approved"] = False
    state["attempts"] = 3

    assert route_after_review(state) == "failed"


def test_reset_demo_files():
    task = CODING_TASKS["divide"]

    reset_demo_files(task)

    for path, expected_content in task.files.items():
        actual_content = (project_root / path).read_text(encoding="utf-8")

        assert actual_content == expected_content
