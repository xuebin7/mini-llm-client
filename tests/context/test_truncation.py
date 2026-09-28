from mini_llm.context.truncation import ToolResultTruncator, TruncationConfig


def test_empty_text():
    empty_str = ""
    tool_result_truncator = ToolResultTruncator(
        TruncationConfig(max_chars=100, head_chars=20, tail_chars=20)
    )
    truncate_result = tool_result_truncator.truncate(empty_str)
    assert truncate_result == empty_str


def test_text_within_limit():
    tool_result_truncator = ToolResultTruncator(
        TruncationConfig(max_chars=100, head_chars=20, tail_chars=20)
    )
    hello = "hello"
    truncate_result = tool_result_truncator.truncate(hello)
    assert truncate_result == hello


def test_truncate_long_text():
    tool_result_truncator = ToolResultTruncator(
        TruncationConfig(max_chars=10, head_chars=3, tail_chars=3)
    )

    test_str = "abcdefghijklmno"
    truncate_result = tool_result_truncator.truncate(test_str)
    assert truncate_result.startswith("abc")
    assert truncate_result.endswith("mno")
    assert "truncated" in truncate_result
