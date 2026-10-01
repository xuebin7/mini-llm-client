from typing import TypedDict

from mini_llm.harness.events import Event
from mini_llm.harness.models import TurnStatus
from mini_llm.harness.runtime import MiniHarness


class MockWebResponse(TypedDict):
    thread_id: str
    turn_id: str
    output: str
    status: TurnStatus
    events: list[Event]


class MockWebClient:
    def __init__(self, harness: MiniHarness) -> None:
        self.harness = harness

    def create_thread(self) -> str:
        thread = self.harness.create_thread()
        return thread.id

    async def send_message(
        self,
        thread_id: str,
        message: str,
    ) -> MockWebResponse:
        result = await self.harness.run_turn_by_id(
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
