import subprocess
import sys
from pathlib import Path

import pytest

from mini_llm.tools.builtin.git_log import git_log


def test_import_outside_repository(tmp_path: Path):
    source_path = Path(__file__).resolve().parents[2] / "src"
    result = subprocess.run(
        [
            sys.executable,
            "-c",
            (
                "import sys; sys.path.insert(0, sys.argv[1]); "
                + "from mini_llm.tools.builtin.git_log import GIT_LOG_TOOL",
            ),
            str(source_path),
        ],
        cwd=tmp_path,
        capture_output=True,
        text=True,
        encoding="utf-8",
        timeout=10,
        check=False,
    )

    assert result.returncode == 0, result.stderr
    assert result.stdout == ""
    assert result.stderr == ""


@pytest.mark.parametrize(
    "subject",
    ["Initial commit", "Subject with \x1f delimiter", "Subject with \u2028 separator"],
)
def test_git_log(tmp_path: Path, subject: str):
    subprocess.run(
        ["git", "init"],
        cwd=tmp_path,
        check=True,
        capture_output=True,
    )

    subprocess.run(
        ["git", "config", "user.name", "Test User"], cwd=tmp_path, check=True
    )

    subprocess.run(
        ["git", "config", "user.email", "test@example.com"], cwd=tmp_path, check=True
    )

    file_path = tmp_path / "hello.txt"
    file_path.write_text("hello", encoding="utf-8")

    subprocess.run(["git", "add", "."], cwd=tmp_path, check=True)

    subprocess.run(
        ["git", "commit", "-m", subject],
        cwd=tmp_path,
        check=True,
        capture_output=True,
    )

    result = git_log(str(tmp_path))

    assert len(result) == 1
    assert result[0]["author"] == "Test User"
    assert result[0]["subject"] == subject
    assert result[0]["hash"]
    assert result[0]["date"]
