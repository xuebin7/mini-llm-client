from typing import TypedDict

from mini_llm.harness.events import Event, EventType
from mini_llm.harness.sink import EventSink


class AgentLoopCall(TypedDict):
    user_input: str
    thread_id: str
    turn_id: str


class FakeAgentLoop:
    def __init__(self, output: str = "ok") -> None:
        self.output = output
        self.calls: list[AgentLoopCall] = []

    async def run(
        self,
        user_input: str,
        thread_id: str | None = None,
        turn_id: str | None = None,
        event_sink: EventSink | None = None,
    ) -> str:
        if thread_id is None:
            raise ValueError("thread_id is required")

        if turn_id is None:
            raise ValueError("turn_id is required")

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

        return self.output


class FailingAgentLoop:
    async def run(
        self,
        user_input: str,
        thread_id: str | None = None,
        turn_id: str | None = None,
        event_sink: EventSink | None = None,
    ) -> str:
        raise RuntimeError("boom")
