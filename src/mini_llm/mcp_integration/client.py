import asyncio
import sys
from typing import Any

from mcp import Client, StdioServerParameters


class MCPClient:
    def __init__(
        self,
        server_params: StdioServerParameters,
    ) -> None:
        self.server_params = server_params
        self._client: Client | None = None

    async def connect(self) -> None:
        if self._client is not None:
            return

        client = Client(self.server_params)
        await client.__aenter__()
        self._client = client

    async def list_tools(self):
        if self._client is None:
            raise RuntimeError("MCP client is not connected")

        return await self._client.list_tools()

    async def call_tool(
        self,
        name: str,
        arguments: dict[str, Any],
        timeout: float = 10.0,
    ):
        if self._client is None:
            raise RuntimeError("MCP client is not connected")

        try:
            return await asyncio.wait_for(
                self._client.call_tool(
                    name,
                    arguments,
                ),
                timeout=timeout,
            )
        except TimeoutError as exc:
            raise RuntimeError(
                f"MCP tool '{name}' timed out after {timeout} seconds"
            ) from exc

    async def close(self) -> None:
        if self._client is None:
            return

        await self._client.__aexit__(None, None, None)
        self._client = None

    async def __aenter__(self):
        await self.connect()
        return self

    async def __aexit__(
        self,
        exc_type,
        exc_val,
        exc_tb,
    ):
        await self.close()


async def main() -> None:
    server_params = StdioServerParameters(
        command=sys.executable,
        args=["-m", "mini_llm.mcp_integration.server"],
    )

    async with Client(server_params) as client:
        print(f"Protocol version: {client.protocol_version}")

        tools_result = await client.list_tools()

        for tool in tools_result.tools:
            print(f"Tool name: {tool.name}")
            print(f"Description: {tool.description}")
            print(f"Input schema: {tool.input_schema}")

        result = await client.call_tool(
            "count_lines",
            {"text": "hello\nworld\nmcp"},
        )

        print(f"Result content: {result.content}")
        print(f"Structured content: {result.structured_content}")
        print(f"Is error: {result.is_error}")


if __name__ == "__main__":
    asyncio.run(main())
