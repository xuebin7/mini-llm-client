import asyncio
import os
from pathlib import Path

from mini_llm.agent.loop import AgentLoop
from mini_llm.client import LLMClient
from mini_llm.context.factory import create_context_manager
from mini_llm.tools.defaults import create_default_registry
from mini_llm.workspace import Workspace


async def main() -> None:
    workspace_root = Path(__file__).resolve().parent / "coding_agent_demo"

    workspace = Workspace(workspace_root)

    client = LLMClient(
        base_url="https://api.deepseek.com",
        api_key=os.environ["LLM_API_KEY"],
        timeout=60,
    )

    registry = create_default_registry(workspace)

    context_manager = create_context_manager(
        client=client,
        model="deepseek-flash",
    )

    agent = AgentLoop(
        client=client,
        registry=registry,
        context_manager=context_manager,
        model="deepseek-flash",
        max_steps=15,
    )

    result = await agent.run(
        "There are failing tests in this repository. "
        "Find the bug and fix it. "
        "Run the test to verify your change. "
        "After the tests pass, inspect the git diff "
        "and summarize what you changed."
    )

    print(result)


if __name__ == "__main__":
    asyncio.run(main())
