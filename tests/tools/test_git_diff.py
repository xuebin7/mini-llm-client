import asyncio
import subprocess
from pathlib import Path

import pytest

from mini_llm.tools.builtin.git_diff import git_diff
from mini_llm.tools.defaults import create_default_registry
from mini_llm.tools.executor import ToolExecutor
from mini_llm.workspace import Workspace


def prepare_git_repo(workspace_root: Path) -> None:
    subprocess.run(
        ["git", "init"],
        cwd=workspace_root,
        check=True,
        capture_output=True,
    )

    subprocess.run(
        ["git", "config", "user.email", "text@example.com"],
        cwd=workspace_root,
        check=True,
        capture_output=True,
    )

    subprocess.run(
        ["git", "config", "user.name", "Test User"],
        cwd=workspace_root,
        check=True,
        capture_output=True,
    )

    test_file = workspace_root / "demo.txt"
    test_file.write_text("hello world\n", encoding="utf-8")

    subprocess.run(
        ["git", "add", "."],
        cwd=workspace_root,
        check=True,
        capture_output=True,
    )

    subprocess.run(
        ["git", "commit", "-m", "initial"],
        cwd=workspace_root,
        check=True,
        capture_output=True,
    )


def commit_all(
    workspace_root: Path,
    message: str,
) -> None:
    subprocess.run(
        ["git", "add", "."],
        cwd=workspace_root,
        check=True,
        capture_output=True,
    )

    subprocess.run(
        ["git", "commit", "-m", message],
        cwd=workspace_root,
        check=True,
        capture_output=True,
    )


@pytest.mark.asyncio
async def test_git_diff_no_changes(tmp_path: Path):
    workspace_root = tmp_path / "workspace"
    workspace_root.mkdir()

    await asyncio.to_thread(
        prepare_git_repo,
        workspace_root,
    )

    test_file = workspace_root / "test_demo.py"
    test_file.write_text(
        "def test_demo():\n    assert 1 + 1 == 2\n",
        encoding="utf-8",
    )

    await asyncio.to_thread(
        commit_all,
        workspace_root,
        "first commit",
    )

    result = git_diff(workspace=Workspace(workspace_root))

    assert result.returncode == 0
    assert result.stdout == ""


@pytest.mark.asyncio
async def test_git_diff_with_changes(tmp_path: Path):
    workspace_root = tmp_path / "workspace"
    workspace_root.mkdir()

    await asyncio.to_thread(
        prepare_git_repo,
        workspace_root,
    )

    test_file = workspace_root / "test_demo.py"
    test_file.write_text(
        "def test_demo():\n    assert 1 + 1 == 2\n",
        encoding="utf-8",
    )

    await asyncio.to_thread(
        commit_all,
        workspace_root,
        "first commit",
    )

    test_file.write_text(
        "def test_demo():\n    assert 1 + 1 == 3\n",
        encoding="utf-8",
    )

    result = git_diff(
        workspace=Workspace(workspace_root),
    )

    assert result.returncode == 0
    assert "assert 1 + 1 == 2" in result.stdout
    assert "assert 1 + 1 == 3" in result.stdout
    assert result.stderr == ""


@pytest.mark.asyncio
async def test_git_diff_tool_with_executor(tmp_path: Path):
    workspace_root = tmp_path / "workspace"
    workspace_root.mkdir()

    await asyncio.to_thread(
        prepare_git_repo,
        workspace_root,
    )

    test_file = workspace_root / "demo.txt"
    # prepare_git_repo 已经提交了 "hello world"
    # 现在只需要制造一个修改
    test_file.write_text(
        "hello python\n",
        encoding="utf-8",
    )

    registry = create_default_registry(Workspace(workspace_root))
    executor = ToolExecutor(registry)

    result = await executor.execute(
        tool_name="git_diff",
        arguments={},
    )

    assert result.success is True
    assert result.output.returncode == 0
    assert "hello world" in result.output.stdout
    assert "hello python" in result.output.stdout
