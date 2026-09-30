from datetime import UTC, datetime

from mini_llm.agent.loop import AgentLoop
from mini_llm.harness.collector import CollectingEventSink
from mini_llm.harness.models import Thread, Turn, TurnResult, TurnStatus


class MiniHarness:
    def __init__(self, agent_loop: AgentLoop) -> None:
        self.agent_loop = agent_loop

    def create_thread(self) -> Thread:
        return Thread()

    async def run_turn(
        self,
        thread: Thread,
        user_input: str,
    ) -> TurnResult:
        turn = Turn(thread_id=thread.id)
        thread.turns.append(turn)

        sink = CollectingEventSink()

        try:
            output = await self.agent_loop.run(
                user_input=user_input,
                thread_id=thread.id,
                turn_id=turn.id,
                event_sink=sink,
            )
        except Exception:
            turn.status = TurnStatus.FAILED
            turn.completed_at = datetime.now(UTC)
            raise

        turn.status = TurnStatus.SUCCEEDED
        turn.completed_at = datetime.now(UTC)

        return TurnResult(
            turn=turn,
            output=output,
            events=sink.events,
        )
