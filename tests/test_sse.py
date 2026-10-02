from mini_llm.harness.events import Event, EventType
from mini_llm.harness.sse import encode_sse


def test_encode_sse():
    event = Event(
        type=EventType.TURN_STARTED,
        thread_id="thread-123",
        turn_id="turn-123",
    )

    encoded = encode_sse(event)

    assert encoded.startswith("event: turn_started\n")
    assert '"thread_id": "thread-123"' in encoded
    assert '"turn_id": "turn-123"' in encoded
    assert encoded.endswith("\n\n")
