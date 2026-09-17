from pathlib import Path

import pytest

from mini_llm.tools.builtin.read_file import read_file


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
