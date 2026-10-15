from pathlib import Path

import pytest

from mini_llm.tools.defaults import create_default_registry
from mini_llm.tools.executor import ToolExecutor
from mini_llm.workspace import Workspace


@pytest.mark.asyncio
async def test_apply_patch_replaces_text(tmp_path: Path):
    workspace_root = tmp_path / "workspace"
    workspace_root.mkdir()

    registry = create_default_registry(Workspace(workspace_root))
    executor = ToolExecutor(registry)

    path = "test.txt"

    await executor.execute(
        tool_name="write_file",
        arguments={
            "path": path,
            "content": "hello world",
        },
    )

    apply_patch_result = await executor.execute(
        tool_name="apply_patch",
        arguments={
            "path": path,
            "old_text": "world",
            "new_text": "python",
        },
    )

    assert apply_patch_result.success is True

    content = (workspace_root / path).read_text(encoding="utf-8")
    assert content == "hello python"


@pytest.mark.asyncio
async def test_apply_patch_preserves_other_content(tmp_path: Path):
    workspace_root = tmp_path / "workspace"
    workspace_root.mkdir()

    registry = create_default_registry(Workspace(workspace_root))
    executor = ToolExecutor(registry)

    path = "./test.txt"

    await executor.execute(
        tool_name="write_file",
        arguments={
            "path": path,
            "content": "line1\nhello world\nline3",
        },
    )

    result = await executor.execute(
        tool_name="apply_patch",
        arguments={
            "path": path,
            "old_text": "hello world",
            "new_text": "hello python",
        },
    )

    assert result.success is True

    content = (workspace_root / path).read_text(encoding="utf-8")

    assert content == "line1\nhello python\nline3"


@pytest.mark.asyncio
async def test_apply_patch_rejects_missing_text(tmp_path: Path):
    workspace_root = tmp_path / "workspace"
    workspace_root.mkdir()

    registry = create_default_registry(Workspace(workspace_root))
    executor = ToolExecutor(registry)

    path = "./test.txt"

    await executor.execute(
        tool_name="write_file",
        arguments={
            "path": path,
            "content": "hello world",
        },
    )

    result = await executor.execute(
        tool_name="apply_patch",
        arguments={
            "path": path,
            "old_text": "goodbye",
            "new_text": "python",
        },
    )

    assert result.success is False
    assert result.error == "old_text not found in file"

    content = (workspace_root / path).read_text(encoding="utf-8")
    assert content == "hello world"


@pytest.mark.asyncio
async def test_apply_patch_rejects_multiple_matches(tmp_path: Path):
    workspace_root = tmp_path / "workspace"
    workspace_root.mkdir()

    registry = create_default_registry(Workspace(workspace_root))
    executor = ToolExecutor(registry)

    path = "./test.txt"

    await executor.execute(
        tool_name="write_file",
        arguments={
            "path": path,
            "content": "hello\nmiddle\nhello\n",
        },
    )

    result = await executor.execute(
        tool_name="apply_patch",
        arguments={
            "path": path,
            "old_text": "hello",
            "new_text": "hi",
        },
    )

    assert result.success is False
    assert result.error == "old_text matched multiple locations"

    content = (workspace_root / path).read_text(encoding="utf-8")
    assert content == "hello\nmiddle\nhello\n"


@pytest.mark.asyncio
async def test_apply_patch_handler_rejects_outside_workspace(
    tmp_path: Path,
):
    workspace_root = tmp_path / "workspace"
    workspace_root.mkdir()

    secret_file = tmp_path / "secret.txt"
    secret_file.write_text("hello world", encoding="utf-8")

    registry = create_default_registry(Workspace(workspace_root))
    executor = ToolExecutor(registry)

    result = await executor.execute(
        tool_name="apply_patch",
        arguments={
            "path": "../secret.txt",
            "old_text": "world",
            "new_text": "python",
        },
    )

    assert result.success is False
    assert secret_file.read_text(encoding="utf-8") == "hello world"

    assert result.error is not None
    assert "not relative to the root" in result.error


@pytest.mark.asyncio
async def test_apply_patch_rejects_empty_old_text(tmp_path: Path):
    workspace_root = tmp_path / "workspace"
    workspace_root.mkdir()

    registry = create_default_registry(Workspace(workspace_root))
    executor = ToolExecutor(registry)

    path = "./test.txt"

    await executor.execute(
        tool_name="write_file",
        arguments={
            "path": path,
            "content": "hello",
        },
    )

    result = await executor.execute(
        tool_name="apply_patch",
        arguments={
            "path": path,
            "old_text": "",
            "new_text": "new",
        },
    )

    assert result.success is False
    assert "old_text must not be empty" == result.error
