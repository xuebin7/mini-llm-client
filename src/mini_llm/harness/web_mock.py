from collections.abc import AsyncIterator
from typing import TypedDict

from mini_llm.harness.app_server import AppServer
from mini_llm.harness.events import Event
from mini_llm.harness.models import TurnStatus


class MockWebResponse(TypedDict):
    thread_id: str
    turn_id: str
    output: str
    status: TurnStatus
    events: list[Event]


class MockWebClient:
    def __init__(self, server: AppServer) -> None:
        self.server = server

    def create_thread(self) -> str:
        return self.server.create_thread()

    async def send_message(
        self,
        thread_id: str,
        message: str,
    ) -> MockWebResponse:
        result = await self.server.run_turn(
            thread_id=thread_id,
            user_input=message,
        )

        return {
            "thread_id": thread_id,
            "turn_id": result.turn.id,
            "output": result.output,
            "status": result.turn.status,
            "events": result.events,
        }

    async def stream_sse(
        self,
        thread_id: str,
        message: str,
    ) -> AsyncIterator[str]:
        async for chunk in self.server.stream_turn(
            thread_id=thread_id,
            user_input=message,
        ):
            yield chunk
