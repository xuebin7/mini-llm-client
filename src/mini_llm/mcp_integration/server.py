import asyncio

from mcp.server import MCPServer

mcp = MCPServer("mini-llm-demo")


@mcp.tool()
def count_lines(text: str) -> int:
    if not text:
        return 0

    return len(text.splitlines())


@mcp.tool()
async def slow_tool(seconds: float) -> str:
    await asyncio.sleep(seconds)
    return "done"


@mcp.tool()
def always_fail() -> str:
    raise RuntimeError("intentional MCP tool failure")


if __name__ == "__main__":
    mcp.run()
