from datetime import UTC, datetime

from mini_llm.harness.collector import CollectingEventSink
from mini_llm.harness.fanout_sink import FanoutEventSink
from mini_llm.harness.models import Thread, Turn, TurnResult, TurnStatus
from mini_llm.harness.protocol import AgentLoopProtocol
from mini_llm.harness.sink import EventSink


class MiniHarness:
    def __init__(self, agent_loop: AgentLoopProtocol) -> None:
        self.agent_loop = agent_loop
        self.threads: dict[str, Thread] = {}

    def create_thread(self) -> Thread:
        thread = Thread()
        self.threads[thread.id] = thread
        return thread

    def get_thread(
        self,
        thread_id: str,
    ) -> Thread | None:
        return self.threads.get(thread_id)

    def list_threads(self) -> list[Thread]:
        return list(self.threads.values())

    async def run_turn(
        self,
        thread: Thread,
        user_input: str,
        event_sink: EventSink | None = None,
    ) -> TurnResult:
        turn = Turn(thread_id=thread.id)
        thread.turns.append(turn)

        collector = CollectingEventSink()

        if event_sink is None:
            sink: EventSink = collector
        else:
            sink = FanoutEventSink(
                collector,
                event_sink,
            )

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
            events=collector.events,
        )

    async def run_turn_by_id(
        self,
        thread_id: str,
        user_input: str,
        event_sink: EventSink | None = None,
    ) -> TurnResult:
        thread = self.get_thread(thread_id)

        if thread is None:
            raise ValueError(f"Thread not found: {thread_id}")

        return await self.run_turn(
            thread=thread,
            user_input=user_input,
            event_sink=event_sink,
        )

    def list_turns(self, thread: Thread) -> list[Turn]:
        return thread.turns

    def get_turn(self, thread: Thread, turn_id: str) -> Turn | None:
        for turn in thread.turns:
            if turn.id == turn_id:
                return turn
        return None
