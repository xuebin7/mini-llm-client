"""Synchronous client for Chat Completions-compatible APIs."""

from collections.abc import Iterator
from contextlib import contextmanager
from typing import Any, Literal, TypeVar

import httpx
from pydantic import BaseModel

from .exceptions import LLMAPIError, LLMTimeoutError
from .models import ChatRequest, ChatResponse, ResponseRequest, ResponseResult, ToolCall
from .retry import retry
from .streaming import StreamParser
from .structured import build_response_format, parse_structured_output

T = TypeVar("T", bound=BaseModel)


class LLMClient:
    """Provide chat, streaming, and locally validated structured output."""

    def __init__(
        self,
        base_url: str,
        api_key: str,
        timeout: float = 30.0,
    ) -> None:
        self.base_url = base_url.rstrip("/")
        self.api_key = api_key
        self.timeout = timeout

    def _headers(self) -> dict[str, str]:
        return {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json",
        }

    @staticmethod
    def _payload(request: ChatRequest, *, stream: bool = False) -> dict[str, Any]:
        payload = request.model_dump(mode="json", exclude_none=True)
        if stream:
            payload["stream"] = True
        return payload

    @contextmanager
    def _translate_timeout(self) -> Iterator[None]:
        """Preserve the HTTPX cause while exposing a consistent SDK exception."""
        try:
            yield
        except httpx.TimeoutException as exc:
            raise LLMTimeoutError(
                f"LLM request timed out (HTTP timeout: {self.timeout}s)"
            ) from exc

    @staticmethod
    def _raise_for_status(response: httpx.Response) -> None:
        if response.status_code >= 400:
            # Streaming responses must be read before accessing response.text.
            response.read()
            raise LLMAPIError(
                status_code=response.status_code,
                message=f"LLM API error: {response.status_code} {response.text}",
            )

    def _do_request(
        self,
        url: str,
        headers: dict[str, str],
        payload: dict[str, Any],
    ) -> httpx.Response:
        with self._translate_timeout():
            response = httpx.post(
                url,
                headers=headers,
                json=payload,
                timeout=self.timeout,
            )
            self._raise_for_status(response)
        return response

    def chat(self, request: ChatRequest) -> ChatResponse:
        """Return a complete response, retrying transient request failures."""
        url = f"{self.base_url}/chat/completions"
        headers = self._headers()
        payload = self._payload(request)
        response = retry(
            lambda: self._do_request(url, headers, payload),
            max_attempts=3,
        )
        data = response.json()
        return ChatResponse(
            content=data["choices"][0]["message"]["content"],
            model=data.get("model"),
        )

    def stream(self, request: ChatRequest) -> Iterator[str]:
        """Yield text without retrying; replay could duplicate delivered chunks.

        Exhaust or explicitly close the iterator to release the connection.
        """
        with self._translate_timeout():
            with httpx.stream(
                "POST",
                f"{self.base_url}/chat/completions",
                headers=self._headers(),
                json=self._payload(request, stream=True),
                timeout=self.timeout,
            ) as response:
                self._raise_for_status(response)
                yield from StreamParser().parse(response.iter_lines())

    @staticmethod
    def schema_to_response_format(schema: type[BaseModel]) -> dict[str, Any]:
        """Build a strict JSON Schema response format for supporting providers."""
        return build_response_format(schema)

    def structured(
        self,
        request: ChatRequest,
        schema: type[T],
        *,
        mode: Literal["json_object", "json_schema"] = "json_object",
    ) -> T:
        """Validate JSON locally; optionally request server-side schema enforcement.

        JSON mode requires instructions describing the expected JSON in messages.
        Schema mode requires provider support. Neither mode mutates the request.
        """
        if mode == "json_object":
            response_format = {"type": "json_object"}
        elif mode == "json_schema":
            response_format = self.schema_to_response_format(schema)
        else:
            raise ValueError("mode must be 'json_object' or 'json_schema'")

        structured_request = request.model_copy(
            update={"response_format": response_format}
        )
        response = self.chat(structured_request)
        return parse_structured_output(response.content, schema)

    def responses(self, request: ResponseRequest) -> ResponseResult:
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
