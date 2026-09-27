from typing import Any

from mini_llm.tools.schema import ToolSchema


def build_tool_name(namespace: str, tool_name: str) -> str:
    return namespace + "." + tool_name


def normalize_description(description: str | None) -> str:
    """Normalize an optional MCP tool description to a string."""
    return description or ""


def mcp_tool_parameters(tool) -> dict[str, Any]:
    return tool.input_schema


def extract_mcp_error_message(content) -> str:
    messages: list[str] = []

    for item in content:
        text = getattr(item, "text", None)
        if text:
            messages.append(text)

    if messages:
        return "\n".join(messages)

    return str(content)


def create_mcp_handler(
    client,
    remote_tool_name: str,
    timeout: float = 10.0,
):
    async def handler(**arguments):
        result = await client.call_tool(
            remote_tool_name,
            arguments,
            timeout=timeout,
        )
        if result.is_error:
            raise RuntimeError(extract_mcp_error_message(result.content))
        if result.structured_content is not None:
            return result.structured_content

        return result.content

    return handler


def mcp_tool_to_schema(
    tool,
    client,
    namespace: str,
    timeout: float = 10.0,
) -> ToolSchema:
    return ToolSchema(
        name=build_tool_name(
            namespace=namespace,
            tool_name=tool.name,
        ),
        description=normalize_description(tool.description),
        parameters=mcp_tool_parameters(tool),
        handler=create_mcp_handler(
            client=client,
            remote_tool_name=tool.name,
            timeout=timeout,
        ),
    )


async def register_mcp_tools(
    client,
    registry,
    namespace: str,
    timeout: float = 10.0,
) -> list[ToolSchema]:
    tools_result = await client.list_tools()

    schemas = []

    for tool in tools_result.tools:
        schema = mcp_tool_to_schema(
            tool=tool,
            client=client,
            namespace=namespace,
            timeout=timeout,
        )

        registry.register(schema)
        schemas.append(schema)

    return schemas
