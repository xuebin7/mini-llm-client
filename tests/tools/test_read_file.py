from pathlib import Path

import pytest

from mini_llm.tools.builtin.read_file import (
    create_read_file_handler,
    create_read_file_tool,
    read_file,
)
from mini_llm.tools.executor import ToolExecutor
from mini_llm.tools.registry import ToolRegistry
from mini_llm.workspace import Workspace


@pytest.fixture
def sample_file(tmp_path: Path) -> Path:
    file_path = tmp_path / "sample.txt"
    file_path.write_text("line 1\nline 2\nline 3\nline 4\nline 5\n", encoding="utf-8")
    return file_path


def test_read_entire_file(sample_file: Path):
    result = read_file(str(sample_file))

    assert result == ("line 1\nline 2\nline 3\nline 4\nline 5\n")


def test_read_line_range(sample_file: Path):
    result = read_file(str(sample_file), start_line=2, end_line=4)

    assert result == ("line 2\nline 3\nline 4\n")


def test_read_from_start_line(sample_file: Path):
    result = read_file(
        str(sample_file),
        start_line=3,
    )

    assert result == ("line 3\nline 4\nline 5\n")


def test_read_until_end_line(sample_file: Path):
    result = read_file(str(sample_file), end_line=2)

    assert result == ("line 1\nline 2\n")


def test_file_not_found(tmp_path: Path):
    path = tmp_path / "missing.txt"

    with pytest.raises(FileNotFoundError):
        read_file(str(path))


def test_path_is_directory(tmp_path: Path):
    with pytest.raises(ValueError, match="Path is not a file"):
        read_file(str(tmp_path))


@pytest.mark.parametrize(
    "start_line,end_line",
    [
        (0, None),
        (-1, None),
        (None, 0),
        (None, -1),
    ],
)
def test_invalid_line_number(
    sample_file: Path,
    start_line: int | None,
    end_line: int | None,
):
    with pytest.raises(ValueError):
        read_file(str(sample_file), start_line=start_line, end_line=end_line)


def test_start_line_greater_than_end_line(sample_file: Path):
    with pytest.raises(ValueError, match="start_line must be <= end_line"):
        read_file(str(sample_file), start_line=4, end_line=2)


def test_read_file_rejects_outside_workspace(tmp_path: Path):
    workspace_root = tmp_path / "workspace"
    workspace_root.mkdir()

    secret_file = tmp_path / "secret.txt"
    secret_file.write_text("secret", encoding="utf-8")

    handler = create_read_file_handler(Workspace(workspace_root))

    with pytest.raises(ValueError):
        handler("../secret.txt")


def test_read_file_handler_reads_inside_workspace(tmp_path: Path):
    file_path = tmp_path / "hello.txt"
    file_path.write_text("hello", encoding="utf-8")

    workspace = Workspace(tmp_path)
    handler = create_read_file_handler(workspace)

    result = handler("hello.txt")

    assert result == "hello"


def test_create_read_file_tool(tmp_path: Path):
    workspace = Workspace(tmp_path)

    tool = create_read_file_tool(workspace)

    assert tool.name == "read_file"
    assert tool.handler is not None


def test_read_file_tool_reads_from_workspace(tmp_path: Path):
    file_path = tmp_path / "hello.txt"
    file_path.write_text("hello", encoding="utf-8")

    tool = create_read_file_tool(Workspace(tmp_path))

    result = tool.handler(path="hello.txt")

    assert result == "hello"


@pytest.mark.asyncio
async def test_executor_reads_file_inside_workspace(tmp_path: Path):
    file_path = tmp_path / "hello.txt"
    file_path.write_text("hello", encoding="utf-8")

    registry = ToolRegistry()
    registry.register(create_read_file_tool(Workspace(tmp_path)))

    executor = ToolExecutor(registry)

    result = await executor.execute(
        "read_file",
        {"path": "hello.txt"},
    )

    assert result.success == True
    assert result.output == "hello"


@pytest.mark.asyncio
async def test_executor_rejects_file_outside_workspace(tmp_path: Path):
    workspace_root = tmp_path / "workspace"
    workspace_root.mkdir()

    secrect_file = tmp_path / "secret.txt"
    secrect_file.write_text("secret", encoding="utf-8")

    registry = ToolRegistry()
    registry.register(create_read_file_tool(Workspace(workspace_root)))

    executor = ToolExecutor(registry)

    result = await executor.execute(
        "read_file",
        {"path": "../secret.txt"},
    )

    assert result.success == False
