from mini_llm.harness.events import Event
from mini_llm.harness.sink import EventSink


class FanoutEventSink:
    def __init__(self, *sinks: EventSink) -> None:
        self.sinks = sinks

    async def emit(self, event: Event) -> None:
        for sink in self.sinks:
            await sink.emit(event)
