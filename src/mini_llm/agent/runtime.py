import json
from typing import Any, cast

from mini_llm.models import ToolCall
from mini_llm.tools.executor import ToolExecutor, ToolResult


async def execute_tool_call(
    tool_call: ToolCall,
    executor: ToolExecutor,
) -> ToolResult:
    try:
        raw_arguments = json.loads(tool_call.arguments)
    except json.JSONDecodeError as exc:
        return ToolResult(
            tool_name=tool_call.name,
            success=False,
            error=f"Invalid tool arguments JSON: {exc}",
        )

    if not isinstance(raw_arguments, dict):
        return ToolResult(
            tool_name=tool_call.name,
            success=False,
            error="Tool arguments must be a JSON object",
        )

    arguments = cast(dict[str, Any], raw_arguments)

    return await executor.execute(
        tool_name=tool_call.name,
        arguments=arguments,
    )


def tool_result_to_output(
    tool_call: ToolCall,
    result: ToolResult,
) -> dict[str, Any]:
    if result.success:
        output = json.dumps(
            {
                "success": True,
                "output": result.output,
            },
            ensure_ascii=False,
        )
    else:
        output = json.dumps(
            {
                "success": False,
                "error": result.error,
            },
            ensure_ascii=False,
        )

    return {
        "type": "function_call_output",
        "call_id": tool_call.call_id,
        "output": output,
    }
