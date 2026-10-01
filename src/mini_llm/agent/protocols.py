from typing import Any, Protocol

from mini_llm.models import ResponseRequest, ResponseResult


class ResponsesClient(Protocol):
    def responses(
        self,
        request: ResponseRequest,
    ) -> ResponseResult: ...


class ContextManagerProtocol(Protocol):
    def prepare(
        self,
        items: list[dict[str, Any]],
    ) -> list[dict[str, Any]]: ...

    def truncate_tool_result(self, text: str) -> str: ...
