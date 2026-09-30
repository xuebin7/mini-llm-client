import pytest

from mini_llm.harness.events import Event, EventType
from mini_llm.harness.models import Thread, Turn, TurnStatus
from mini_llm.harness.runtime import MiniHarness
from mini_llm.harness.sink import EventSink


class FakeAgentLoop:
    def __init__(self):
        self.calls = []

    async def run(
        self,
        user_input: str,
        thread_id: str,
        turn_id: str,
        event_sink: EventSink | None = None,
    ) -> str:
        if event_sink is not None:
            await event_sink.emit(
                Event(
                    type=EventType.TURN_STARTED,
                    thread_id=thread_id,
                    turn_id=turn_id,
                )
            )
        self.calls.append(
            {
                "user_input": user_input,
                "thread_id": thread_id,
                "turn_id": turn_id,
            }
        )
        if event_sink is not None:
            await event_sink.emit(
                Event(
                    type=EventType.TURN_COMPLETED,
                    thread_id=thread_id,
                    turn_id=turn_id,
                )
            )
        return "ok"


class FailingAgentLoop:
    async def run(
        self,
        user_input: str,
        thread_id: str,
        turn_id: str,
        event_sink: EventSink | None = None,
    ) -> str:
        raise RuntimeError("boom")


def test_harness_can_create_thread():
    agent_loop = FakeAgentLoop()

    harness = MiniHarness(agent_loop=agent_loop)

    thread = harness.create_thread()

    assert thread.id.startswith("thread-")


@pytest.mark.asyncio
async def test_harness_run_turn_calls_agent_loop():
    agent_loop = FakeAgentLoop()
    harness = MiniHarness(agent_loop=agent_loop)

    thread = harness.create_thread()

    result = await harness.run_turn(
        thread=thread,
        user_input="hello",
    )

    assert result.output == "ok"
    assert result.turn.thread_id == thread.id
    assert result.turn.id.startswith("turn-")
    assert len(agent_loop.calls) == 1

    call = agent_loop.calls[0]

    assert call["thread_id"] == result.turn.thread_id
    assert call["turn_id"] == result.turn.id
    assert call["user_input"] == "hello"


def test_thread_turns_empty_when_created():
    thread = Thread()

    assert thread.turns == []


@pytest.mark.asyncio
async def test_harness_turns_after_run_turn():
    agent_loop = FakeAgentLoop()
    harness = MiniHarness(agent_loop=agent_loop)

    thread = Thread()

    result = await harness.run_turn(
        thread=thread,
        user_input="hello",
    )

    assert len(thread.turns) == 1
    assert thread.turns[0].id == result.turn.id


@pytest.mark.asyncio
async def test_harness_run_turn_multi_time():
    agent_loop = FakeAgentLoop()
    harness = MiniHarness(agent_loop=agent_loop)

    thread = Thread()

    result1 = await harness.run_turn(
        thread=thread,
        user_input="hello",
    )

    result2 = await harness.run_turn(
        thread=thread,
        user_input="world",
    )

    assert len(thread.turns) == 2
    assert thread.turns[0].id == result1.turn.id
    assert thread.turns[1].id == result2.turn.id


def test_turn_status_running_when_created():
    thread = Thread()
    turn = Turn(thread_id=thread.id)

    assert turn.status == TurnStatus.RUNNING
    assert turn.completed_at is None


@pytest.mark.asyncio
async def test_turn_status_succeeded_after_run():
    agent_loop = FakeAgentLoop()
    harness = MiniHarness(agent_loop=agent_loop)

    thread = Thread()

    result = await harness.run_turn(
        thread=thread,
        user_input="hello",
    )

    assert result.turn.status == TurnStatus.SUCCEEDED
    assert thread.turns[0].status == TurnStatus.SUCCEEDED

    assert result.turn.completed_at is not None


@pytest.mark.asyncio
async def test_turn_status_failed_when_agent_loop_raises():
    agent_loop = FailingAgentLoop()
    harness = MiniHarness(agent_loop=agent_loop)
    thread = harness.create_thread()

    with pytest.raises(RuntimeError, match="boom"):
        await harness.run_turn(
            thread=thread,
            user_input="hello",
        )

    assert len(thread.turns) == 1

    turn = thread.turns[0]

    assert turn.status == TurnStatus.FAILED
    assert turn.completed_at is not None


@pytest.mark.asyncio
async def test_harness_returns_events_after_succeed():
    agent_loop = FakeAgentLoop()
    harness = MiniHarness(agent_loop=agent_loop)

    thread = harness.create_thread()

    result = await harness.run_turn(
        thread=thread,
        user_input="hello",
    )

    assert result.events
    assert result.events[0].type == EventType.TURN_STARTED
    assert result.events[-1].type == EventType.TURN_COMPLETED

    result1 = await harness.run_turn(
        thread=thread,
        user_input="test result1",
    )

    result2 = await harness.run_turn(
        thread=thread,
        user_input="test result2",
    )

    assert result1.turn.id != result2.turn.id
    assert all(event.turn_id == result1.turn.id for event in result1.events)
    assert all(event.turn_id == result2.turn.id for event in result2.events)

    assert result1.events is not result2.events
