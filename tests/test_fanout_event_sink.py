import pytest

from mini_llm.harness.collector import CollectingEventSink
from mini_llm.harness.events import Event, EventType
from mini_llm.harness.fanout_sink import FanoutEventSink


@pytest.mark.asyncio
async def test_fanout_event_sink_emits_to_all_sinks():
    sink1 = CollectingEventSink()
    sink2 = CollectingEventSink()

    fanout = FanoutEventSink(
        sink1,
        sink2,
    )

    event = Event(
        type=EventType.TURN_STARTED,
        thread_id="thread-123",
        turn_id="turn-123",
    )

    await fanout.emit(event)

    assert sink1.events == [event]
    assert sink2.events == [event]
