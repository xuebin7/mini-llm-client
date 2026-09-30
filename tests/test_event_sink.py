import pytest

from mini_llm.harness.collector import CollectingEventSink
from mini_llm.harness.events import Event, EventType


def test_init_empty():
    sink = CollectingEventSink()

    assert sink.events == []


@pytest.mark.asyncio
async def test_emit_collects_event():
    sink = CollectingEventSink()

    event1 = Event(
        type=EventType.TURN_STARTED,
        thread_id="thread_123",
        turn_id="turn_123",
        data={"foo": "bar"},
    )

    await sink.emit(event1)

    assert sink.events[0] == event1


@pytest.mark.asyncio
async def test_multi_sink_emits_keep_order():
    sink = CollectingEventSink()

    event1 = Event(
        type=EventType.TURN_STARTED,
        thread_id="thread_123",
        turn_id="turn_123",
    )

    event2 = Event(
        type=EventType.TURN_COMPLETED,
        thread_id="thread_456",
        turn_id="turn_456",
    )

    event3 = Event(
        type=EventType.TURN_FAILED,
        thread_id="thread_789",
        turn_id="turn_789",
    )

    await sink.emit(event1)
    await sink.emit(event2)
    await sink.emit(event3)

    assert sink.events == [event1, event2, event3]
