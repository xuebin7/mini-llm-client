from pathlib import Path

import pytest

from mini_llm.tools.builtin.run_test import run_test
from mini_llm.tools.defaults import create_default_registry
from mini_llm.tools.executor import ToolExecutor
from mini_llm.workspace import Workspace


def test_run_test_success(tmp_path: Path):
    workspace_root = tmp_path / "workspace"
    workspace_root.mkdir()

    test_file = workspace_root / "test_demo.py"
    test_file.write_text(
        "def test_demo():\n    assert 1 + 1 == 2\n",
        encoding="utf-8",
    )

    result = run_test(
        Workspace(workspace_root),
        target="test_demo.py",
    )

    assert result.returncode == 0
    assert "passed" in result.stdout


def test_run_test_fail(tmp_path: Path):
    workspace_root = tmp_path / "workspace"
    workspace_root.mkdir()

    test_file = workspace_root / "test_demo.py"
    test_file.write_text(
        "def test_demo():\n    assert 1 + 1 == 3\n",
        encoding="utf-8",
    )

    result = run_test(
        workspace=Workspace(workspace_root),
        target="test_demo.py",
    )

    assert result.returncode != 0
    assert "failed" in result.stdout


def test_run_test_all_tests(tmp_path: Path):
    workspace_root = tmp_path / "workspace"
    workspace_root.mkdir()

    (workspace_root / "test_one.py").write_text(
        "def test_one():\n    assert True\n",
        encoding="utf-8",
    )

    (workspace_root / "test_two.py").write_text(
        "def test_two():\n   assert True\n",
        encoding="utf-8",
    )

    result = run_test(Workspace(workspace_root))

    assert result.returncode == 0
    assert "2 passed" in result.stdout


@pytest.mark.asyncio
async def test_run_test_tool_with_executor(tmp_path: Path):
    workspace_root = tmp_path / "workspace"
    workspace_root.mkdir()

    test_file = workspace_root / "test_demo.py"
    test_file.write_text(
        "def test_demo():\n    assert True\n",
        encoding="utf-8",
    )

    registry = create_default_registry(Workspace(workspace_root))
    executor = ToolExecutor(registry)

    result = await executor.execute(
        tool_name="run_test",
        arguments={
            "target": "test_demo.py",
        },
    )

    assert result.success is True
    assert result.output.returncode == 0
    assert "passed" in result.output.stdout
