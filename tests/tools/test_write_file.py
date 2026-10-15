from pathlib import Path

import pytest

from mini_llm.tools.builtin.read_file import read_file
from mini_llm.tools.builtin.write_file import (
    create_write_file_handler,
    create_write_file_tool,
    write_file,
)
from mini_llm.tools.defaults import create_default_registry
from mini_llm.tools.executor import ToolExecutor
from mini_llm.workspace import Workspace


def test_write_file_creates_file(tmp_path: Path):
    path = tmp_path / "test.txt"
    write_file(path, "hello world")

    assert path.exists()


def test_write_file_overwrites_existing_file(tmp_path: Path):
    path = tmp_path / "test.txt"
    write_file(path, "hello world")

    write_file(path, "hehe")

    content = read_file(path)

    assert content == "hehe"


def test_write_file_creates_parent_directories(tmp_path: Path):
    path = tmp_path / "src" / "demo" / "test.txt"
    write_file(path, "hello world")

    assert path.exists()


def test_write_file_handler_rejects_outside_workspace(tmp_path: Path):
    workspace_root = tmp_path / "workspace"
    workspace_root.mkdir()

    handler = create_write_file_handler(workspace=Workspace(workspace_root))

    with pytest.raises(ValueError):
        handler("../secret.txt", "hello world")


def test_create_write_file_tool(tmp_path: Path):
    tool = create_write_file_tool(Workspace(tmp_path))

    assert tool.name == "write_file"


def test_write_file_tool_writes_file(tmp_path: Path):
    tool = create_write_file_tool(Workspace(tmp_path))

    tool.handler(
        path="hello.txt",
        content="hello",
    )

    assert (tmp_path / "hello.txt").read_text(encoding="utf-8") == "hello"


@pytest.mark.asyncio
async def test_executor_writes_file_inside_workspace(tmp_path: Path):
    workspace = Workspace(tmp_path)
    registry = create_default_registry(workspace)
    executor = ToolExecutor(registry)

    result = await executor.execute(
        "write_file",
        {
            "path": "hello.txt",
            "content": "hello world",
        },
    )

    assert result.success is True
    assert (tmp_path / "hello.txt").read_text(encoding="utf-8") == "hello world"


@pytest.mark.asyncio
async def test_executor_rejects_write_outside_workspace(tmp_path: Path):
    workspace_root = tmp_path / "workspace"
    workspace_root.mkdir()

    registry = create_default_registry(Workspace(workspace_root))
    executor = ToolExecutor(registry)

    result = await executor.execute(
        tool_name="write_file",
        arguments={
            "path": "../secret.txt",
            "content": "secret",
        },
    )

    assert result.success is False
    assert not (tmp_path / "secret.txt").exists()
