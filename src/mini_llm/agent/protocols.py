from typing import Protocol

from mini_llm.models import ResponseRequest, ResponseResult, ToolCall
from mini_llm.retry import retry


class ResponsesClient(Protocol):
    def responses(
        self,
        request: ResponseRequest,
    ) -> ResponseResult:
        """Create a Responses API request and parse function calls."""

        url = f"{self.base_url}/responses"
        headers = self._headers()

        payload = request.model_dump(mode="json", exclude_none=True)

        response = retry(
            lambda: self._do_request(
                url,
                headers,
                payload,
            ),
            max_attempts=3,
        )

        data = response.json()

        tool_calls: list[ToolCall] = []
        text_parts: list[str] = []

        for item in data.get("output", []):
            if item.get("type") == "function_call":
                tool_calls.append(
                    ToolCall(
                        call_id=item["call_id"],
                        name=item["name"],
                        arguments=item["arguments"],
                    )
                )

            elif item.get("type") == "message":
                for content in item.get("content", []):
                    if content.get("type") == "output_text":
                        text_parts.append(content["text"])

        text = "".join(text_parts) or None

        return ResponseResult(
            text=text,
            tool_calls=tool_calls,
            model=data.get("model"),
            response_id=data.get("id"),
            output=data.get("output", []),
        )
