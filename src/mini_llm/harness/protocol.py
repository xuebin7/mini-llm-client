from typing import Protocol

from mini_llm.harness.sink import EventSink


class AgentLoopProtocol(Protocol):
    async def run(
        self,
        user_input: str,
        thread_id: str | None = None,
        turn_id: str | None = None,
        event_sink: EventSink | None = None,
    ) -> str: ...
