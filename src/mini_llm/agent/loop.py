from uuid import uuid4

from mini_llm.agent.protocols import ContextManagerProtocol, ResponsesClient
from mini_llm.agent.runtime import (
    execute_tool_call,
    tool_result_to_output,
)
from mini_llm.harness.events import Event, EventType
from mini_llm.harness.sink import EventSink
from mini_llm.models import ResponseRequest
from mini_llm.tools.executor import ToolExecutor
from mini_llm.tools.registry import ToolRegistry


class AgentLoop:
    def __init__(
        self,
        client: ResponsesClient,
        registry: ToolRegistry,
        context_manager: ContextManagerProtocol,
        *,
        model: str,
        event_sink: EventSink | None = None,
        max_steps: int = 10,
    ) -> None:
        self.client = client
        self.registry = registry
        self.context_manager = context_manager
        self.executor = ToolExecutor(registry)
        self.model = model
        self.event_sink = event_sink
        self.max_steps = max_steps

    async def emit(
        self,
        event: Event,
        event_sink: EventSink | None = None,
    ) -> None:
        sink = event_sink or self.event_sink

        if sink is None:
            return

        await sink.emit(event)

    async def run(
        self,
        user_input: str,
        thread_id: str | None = None,
        turn_id: str | None = None,
        event_sink: EventSink | None = None,
    ) -> str:
        actual_thread_id = thread_id or f"thread-{uuid4()}"
        actual_turn_id = turn_id or f"turn-{uuid4()}"

        await self.emit(
            Event(
                type=EventType.TURN_STARTED,
                thread_id=actual_thread_id,
                turn_id=actual_turn_id,
            ),
            event_sink=event_sink,
        )

        input_items: str | list[dict[str, object]] = user_input
        final_result: str | None = None

        for step in range(self.max_steps):
            prepared_input = input_items

            if isinstance(input_items, list):
                prepared_input = self.context_manager.prepare(items=input_items)

            request = ResponseRequest(
                model=self.model,
                input=prepared_input,
                tools=self.registry.to_openai_tools(),
            )

            response = self.client.responses(request)

            print(f"\n[agent] step={step + 1}")
            print(f"[agent] text={response.text}")
            print(f"[agent] tool_calls={response.tool_calls}")

            if not response.tool_calls:
                final_result = response.text or ""

                await self.emit(
                    Event(
                        type=EventType.AGENT_MESSAGE,
                        thread_id=actual_thread_id,
                        turn_id=actual_turn_id,
                        data={"text": final_result},
                    ),
                    event_sink=event_sink,
                )

                break

            if isinstance(input_items, str):
                input_items = [
                    {
                        "role": "user",
                        "content": input_items,
                    }
                ]

            input_items.extend(response.output)

            for tool_call in response.tool_calls:
                await self.emit(
                    Event(
                        type=EventType.TOOL_STARTED,
                        thread_id=actual_thread_id,
                        turn_id=actual_turn_id,
                        data={
                            "call_id": tool_call.call_id,
                            "tool_name": tool_call.name,
                            "arguments": tool_call.arguments,
                        },
                    ),
                    event_sink=event_sink,
                )

                print(f"[tool] calling {tool_call.name} with {tool_call.arguments}")
                tool_result = await execute_tool_call(
                    tool_call,
                    self.executor,
                )

                print(
                    f"[tool] success={tool_result.success} "
                    f"output={tool_result.output} "
                    f"error={tool_result.error}"
                )

                if tool_result.success and tool_result.output is not None:
                    await self.emit(
                        Event(
                            type=EventType.TOOL_COMPLETED,
                            thread_id=actual_thread_id,
                            turn_id=actual_turn_id,
                            data={
                                "call_id": tool_call.call_id,
                                "tool_name": tool_call.name,
                            },
                        ),
                        event_sink=event_sink,
                    )
                    tool_result.output = self.context_manager.truncate_tool_result(
                        str(tool_result.output)
                    )
                else:
                    await self.emit(
                        Event(
                            type=EventType.TOOL_FAILED,
                            thread_id=actual_thread_id,
                            turn_id=actual_turn_id,
                            data={
                                "call_id": tool_call.call_id,
                                "tool_name": tool_call.name,
                                "error": tool_result.error,
                            },
                        ),
                        event_sink=event_sink,
                    )

                input_items.append(
                    tool_result_to_output(
                        tool_call,
                        tool_result,
                    )
                )

        if final_result is None:
            await self.emit(
                Event(
                    type=EventType.TURN_FAILED,
                    thread_id=actual_turn_id,
                    turn_id=actual_turn_id,
                    data={
                        "reason": "max_steps_exceeded",
                        "max_steps": self.max_steps,
                    },
                ),
                event_sink=event_sink,
            )

            raise RuntimeError(f"Agent exceeded max_steps={self.max_steps}")

        await self.emit(
            Event(
                type=EventType.TURN_COMPLETED,
                thread_id=actual_thread_id,
                turn_id=actual_turn_id,
            ),
            event_sink=event_sink,
        )

        return final_result
