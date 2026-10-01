import pytest

from mini_llm.harness.events import Event, EventType
from mini_llm.harness.queue_sink import QueueEventSink


@pytest.mark.asyncio
async def test_queue_event_sink():
    sink = QueueEventSink()

    event = Event(
        type=EventType.TURN_STARTED,
        thread_id="thread-123",
        turn_id="turn-123",
    )

    await sink.emit(event)

    received = await sink.next_event()

    assert received == event
