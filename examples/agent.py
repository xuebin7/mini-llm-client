import asyncio
import os

from mini_llm.agent.loop import AgentLoop
from mini_llm.client import LLMClient
from mini_llm.context.factory import create_context_manager
from mini_llm.tools.defaults import create_default_registry


async def main() -> None:
    client = LLMClient(
        base_url="https://api.deepseek.com",
        api_key=os.environ["LLM_API_KEY"],
        timeout=60,
    )

    registry = create_default_registry()

    context_manager = create_context_manager(client=client, model="deepseek-flash")

    agent = AgentLoop(
        client=client,
        registry=registry,
        context_manager=context_manager,
        model="deepseek-flash",
        max_steps=10,
    )

    result = await agent.run(
        "请先尝试读取文件 "
        "`src/mini_llm/tools/builtin/tool_executor.py`。"
        "如果这个文件不存在，不要停止，"
        "请自己查找 ToolExecutor 的真实定义位置，"
        "读取相关代码，并告诉我 ToolExecutor 的职责。"
    )

    print(result)


if __name__ == "__main__":
    asyncio.run(main())
