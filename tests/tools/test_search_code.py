from pathlib import Path

import pytest

from mini_llm.tools.builtin.search_code import search_code


def test_search_code_found(tmp_path: Path):
    file_path = tmp_path / "example.py"

    file_path.write_text("class ToolExecutor:\n   pass\n", encoding="utf-8")

    result = search_code(
        "ToolExecutor",
        str(tmp_path),
    )

    assert len(result) == 1
    assert result[0]["path"] == str(file_path)
    assert result[0]["line"] == 1
    assert result[0]["content"] == "class ToolExecutor:"


def test_search_code_multiple_matches(tmp_path: Path):
    file_path = tmp_path / "example.py"

    file_path.write_text(
        "我有一只小毛驴\n我从来也不骑\n有一天我心血来潮\n骑着去赶集",
        encoding="utf-8",
    )

    result = search_code(
        "一",
        str(tmp_path),
    )

    assert len(result) == 2
    assert result[0]["path"] == str(file_path)
    assert result[0]["line"] == 1
    assert result[0]["content"] == "我有一只小毛驴"

    assert result[1]["line"] == 3
    assert result[1]["content"] == "有一天我心血来潮"


def test_search_code_recursive(tmp_path: Path):
    file_path = tmp_path / "src/mini_llm/executor.py"
    file_path.parent.mkdir(parents=True, exist_ok=True)

    file_path.write_text(
        "# this is a comment line.\nprint('hello world!')", encoding="utf-8"
    )

    result = search_code(
        "world!",
        str(tmp_path),
    )

    assert len(result) == 1
    assert result[0]["path"] == str(file_path)
    assert result[0]["line"] == 2
    assert result[0]["content"] == "print('hello world!')"


def test_search_code_no_match(tmp_path: Path):
    file_path = tmp_path / "example.py"

    file_path.write_text("class ToolExecutor:\n   pass\n", encoding="utf-8")

    result = search_code(
        "not exist text",
        str(tmp_path),
    )

    assert result == []


def test_search_code_empty_query(tmp_path: Path):
    file_path = tmp_path / "example.py"

    file_path.write_text("class ToolExecutor:\n   pass\n", encoding="utf-8")

    with pytest.raises(ValueError) as exc_info:
        result = search_code(
            "",
            str(tmp_path),
        )

    assert "query must not be empty" in str(exc_info.value)


def test_search_code_path_not_found(tmp_path: Path):
    missing_path = tmp_path / "missing"

    with pytest.raises(FileNotFoundError) as exc_info:
        search_code(
            "path doesn't exist",
            str(missing_path),
        )

    assert "Path not found" in str(exc_info.value)


def test_search_code_single_file(tmp_path: Path):
    target_path = tmp_path / "target.py"
    target_path.write_text("Contains ToolExecutor in target.py\n", encoding="utf-8")

    other_path = tmp_path / "other.py"
    other_path.write_text("Contains ToolExecutor in other.py.\n", encoding="utf-8")

    result = search_code("ToolExecutor", str(target_path))
    assert len(result) == 1
    assert result[0]["path"] == str(target_path)
    assert result[0]["line"] == 1
    assert result[0]["content"] == "Contains ToolExecutor in target.py"


def test_search_code_ignore_binary_file(tmp_path: Path):
    text_file = tmp_path / "example.py"
    text_file.write_text(
        "class ToolExecutor:\n    pass\n",
        encoding="utf-8",
    )

    binary_path = tmp_path / "binary.bin"
    binary_path.write_bytes(b"\xff\xfe\xfd\x00\x01\x02")

    result = search_code(
        "ToolExecutor",
        str(tmp_path),
    )

    assert len(result) == 1
    assert result[0]["path"] == str(text_file)
