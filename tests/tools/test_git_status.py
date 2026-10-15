import subprocess
from pathlib import Path

from mini_llm.tools.builtin.git_status import git_status
from mini_llm.workspace import Workspace


def test_git_status_clean_repo(tmp_path: Path):
    workspace_root = tmp_path / "workspace"
    workspace_root.mkdir()

    subprocess.run(
        ["git", "init"],
        cwd=workspace_root,
        check=True,
        capture_output=True,
    )

    subprocess.run(
        ["git", "config", "user.email", "test@example.com"],
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

    result = git_status(Workspace(workspace_root))

    assert result.returncode == 0
    assert result.stdout == ""


def test_git_status_modified_repo(tmp_path: Path):
    workspace_root = tmp_path / "workspace"
    workspace_root.mkdir()

    subprocess.run(
        ["git", "init"],
        cwd=workspace_root,
        check=True,
        capture_output=True,
    )

    subprocess.run(
        ["git", "config", "user.email", "test@example.com"],
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

    tracked_file = workspace_root / "tracked.txt"
    tracked_file.write_text(
        "hello from tracked.txt",
        encoding="utf-8",
    )

    subprocess.run(
        ["git", "add", "."],
        cwd=workspace_root,
        check=True,
        capture_output=True,
    )

    subprocess.run(
        ["git", "commit", "-m", "first commit"],
        cwd=workspace_root,
        check=True,
        capture_output=True,
    )

    tracked_file.write_text(
        "tracked file changed",
        encoding="utf-8",
    )

    new_file = workspace_root / "new.txt"
    new_file.write_text(
        "hello from new.txt",
        encoding="utf-8",
    )

    result = git_status(Workspace(workspace_root))

    assert result.returncode == 0
    assert "tracked.txt" in result.stdout
    assert "new.txt" in result.stdout
