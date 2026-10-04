import asyncio
from collections.abc import AsyncIterator

from mini_llm.harness.events import EventType
from mini_llm.harness.models import TurnResult
from mini_llm.harness.queue_sink import QueueEventSink
from mini_llm.harness.runtime import MiniHarness
from mini_llm.harness.sse import encode_sse
from mini_llm.harness.stream_registry import StreamRegistry


class AppServer:
    def __init__(
        self,
        harness: MiniHarness,
        stream_registry: StreamRegistry | None = None,
    ) -> None:
        self.harness = harness
        self.stream_registry = stream_registry or StreamRegistry()

    def create_thread(self) -> str:
        thread = self.harness.create_thread()
        return thread.id

    async def run_turn(
        self,
        thread_id: str,
        user_input: str,
    ) -> TurnResult:
        return await self.harness.run_turn_by_id(
            thread_id=thread_id,
            user_input=user_input,
        )

    async def stream_turn(
        self,
        thread_id: str,
        user_input: str,
    ) -> AsyncIterator[str]:
        sink = QueueEventSink()

        task = asyncio.create_task(
            self.harness.run_turn_by_id(
                thread_id=thread_id,
                user_input=user_input,
                event_sink=sink,
            )
        )

        while True:
            event = await sink.next_event()

            yield encode_sse(event)

            if event.type in [
                EventType.TURN_COMPLETED,
                EventType.TURN_FAILED,
            ]:
                break

        await task

    def create_stream(
        self,
        stream_id: str,
        thread_id: str,
        user_input: str,
    ) -> None:
        self.stream_registry.register(
            stream_id=stream_id,
            thread_id=thread_id,
            user_input=user_input,
        )

    async def open_stream(
        self,
        stream_id: str,
    ) -> AsyncIterator[str]:
        stream = self.stream_registry.get(stream_id)

        if stream is None:
            raise ValueError(f"Stream not found: {stream_id}")

        try:
            async for chunk in self.stream_turn(
                thread_id=stream.thread_id,
                user_input=stream.user_input,
            ):
                yield chunk

        finally:
            self.stream_registry.remove(stream_id)
