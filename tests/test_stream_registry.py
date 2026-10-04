from mini_llm.harness.stream_registry import StreamRegistry


def test_stream_registry_can_register_and_get():
    registry = StreamRegistry()

    registry.register(
        stream_id="stream-1",
        thread_id="thread-1",
        user_input="hello",
    )

    stream = registry.get("stream-1")

    assert stream is not None
    assert stream.thread_id == "thread-1"
    assert stream.user_input == "hello"


def test_stream_registry_returns_none_for_unknown_stream():
    registry = StreamRegistry()

    assert registry.get("unknown") is None


def test_stream_registry_can_remove_stream():
    registry = StreamRegistry()

    registry.register(
        stream_id="stream-1",
        thread_id="thread-1",
        user_input="hello",
    )

    removed = registry.remove("stream-1")

    assert removed is not None
    assert registry.get("stream-1") is None
