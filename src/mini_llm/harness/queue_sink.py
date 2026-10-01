import asyncio

from mini_llm.harness.events import Event


class QueueEventSink:
    def __init__(self) -> None:
        self.queue: asyncio.Queue[Event] = asyncio.Queue()

    async def emit(self, event: Event) -> None:
        await self.queue.put(event)

    async def next_event(self) -> Event:
        return await self.queue.get()
