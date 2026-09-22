from mini_llm.agent.protocols import ResponsesClient
from mini_llm.agent.runtime import (
    execute_tool_call,
    tool_result_to_output,
)
from mini_llm.models import ResponseRequest
from mini_llm.tools.executor import ToolExecutor
from mini_llm.tools.registry import ToolRegistry


class AgentLoop:
    def __init__(
        self,
        client: ResponsesClient,
        registry: ToolRegistry,
        *,
        model: str,
        max_steps: int = 10,
    ) -> None:
        self.client = client
        self.registry = registry
        self.executor = ToolExecutor(registry)
        self.model = model
        self.max_steps = max_steps

    def run(self, user_input: str) -> str:
        input_items: str | list[dict] = user_input

        for step in range(self.max_steps):
            request = ResponseRequest(
                model=self.model,
                input=input_items,
                tools=self.registry.to_openai_tools(),
            )

            response = self.client.responses(request)

            print(f"\n[agent] step={step + 1}")
            print(f"[agent] text={response.text}")
            print(f"[agent] tool_calls={response.tool_calls}")

            if not response.tool_calls:
                return response.text or ""

            if isinstance(input_items, str):
                input_items = [
                    {
                        "role": "user",
                        "content": input_items,
                    }
                ]

            input_items.extend(response.output)

            for tool_call in response.tool_calls:
                print(f"[tool] calling {tool_call.name} with {tool_call.arguments}")
                result = execute_tool_call(
                    tool_call,
                    self.executor,
                )

                print(
                    f"[tool] success={result.success} "
                    f"output={result.output} "
                    f"error={result.error}"
                )

                input_items.append(
                    tool_result_to_output(
                        tool_call,
                        result,
                    )
                )

        raise RuntimeError(f"Agent exceeded max_steps={self.max_steps}")
