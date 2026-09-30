from datetime import UTC, datetime

from mini_llm.harness.events import Event, EventType


def test_events_can_create():
    event = Event(
        type=EventType.TURN_STARTED,
        thread_id="thread_123",
        turn_id="turn_123",
        data={},
    )

    assert event.type == EventType.TURN_STARTED
    assert event.thread_id == "thread_123"
    assert event.turn_id == "turn_123"


def test_events_timestamp_can_auto_gen():
    before = datetime.now(UTC)

    event = Event(
        type=EventType.TURN_STARTED,
        thread_id="thread_123",
        turn_id="turn_123",
        data={},
    )

    after = datetime.now(UTC)

    assert event.timestamp.tzinfo is not None
    assert before <= event.timestamp <= after


def test_events_data_default_empty_dict():
    event = Event(
        type=EventType.TURN_STARTED,
        thread_id="thread_123",
        turn_id="turn_123",
    )

    assert event.data == {}


def test_events_data_not_shared_between_instances():
    event1 = Event(
        type=EventType.TURN_STARTED,
        thread_id="thread_123",
        turn_id="turn_123",
    )

    event2 = Event(
        type=EventType.TURN_STARTED,
        thread_id="thread_456",
        turn_id="turn_456",
    )

    event1.data["foo"] = "bar"

    assert event1.data["foo"] == "bar"
    assert event2.data == {}
