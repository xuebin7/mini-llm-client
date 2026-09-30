from __future__ import annotations

from typing import Protocol

from mini_llm.harness.events import Event


class EventSink(Protocol):
    async def emit(self, event: Event) -> None: ...
