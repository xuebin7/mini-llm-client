from mini_llm.streaming import StreamParser

def test_parse_stream():
    lines = [
        'data: {"choices":[{"delta":{"content":"Redis"}}]}',
        "",
        'data: {"choices":[{"delta":{"content":" 是"}}]}',
        "",
        'data: {"choices":[{"delta":{"content":" 一个数据库"}}]}',
        "",
        'data: [DONE]'
    ]

    parser = StreamParser()

    result = list(parser.parse(lines))

    assert result == [
        "Redis",
        " 是",
        " 一个数据库"
    ]

    empty_lines = [
        "",
        "data: [DONE]"
    ]
    result_empty = list(parser.parse(empty_lines))
    assert result_empty == []

    error_lines = [
        'notdata: {"choices":[{"delta":{"content":"Redis"}}]}',
        "",
        "data: [DONE]"
    ]
    result_error = list(parser.parse(error_lines))
    assert result_error == []

    redunant_lines = [
        'data: {"choices":[{"delta":{"content":"Redis"}}]}',
        "",
        'data: {"choices":[{"delta":{"content":" 是"}}]}',
        "",
        "data: [DONE]",
        "",
            'data: {"choices":[{"delta":{"content":" 额外的内容"}}]}',
    ]

    result_redundant = list(parser.parse(redunant_lines))
    assert result_redundant == ["Redis", " 是"]
