from pathlib import Path

import pytest

from mini_llm.workspace import Workspace


def test_workspace_resolves_root(tmp_path: Path):
    workspace = Workspace(tmp_path)

    assert workspace.root == tmp_path


def test_resolve_relative_path(tmp_path: Path):
    workspace = Workspace(tmp_path)

    resolved_path = workspace.resolve_path("src/main.py")

    assert resolved_path == (tmp_path / "src/main.py").resolve()


def test_resolve_normalized_path(tmp_path: Path):
    workspace = Workspace(tmp_path)

    resolved_path = workspace.resolve_path("src/../README.md")

    assert resolved_path == (tmp_path / "README.md").resolve()


def test_reject_path_outside_workspace(tmp_path: Path):
    workspace = Workspace(tmp_path)

    with pytest.raises(ValueError):
        workspace.resolve_path("../secret.txt")
