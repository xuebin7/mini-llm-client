from typing import Any

from pydantic import BaseModel


class Message(BaseModel):
    role: str
    content: str


class ChatRequest(BaseModel):
    messages: list[Message]
    model: str
    temperature: float | None = None
    response_format: dict | None = None


class ChatResponse(BaseModel):
    content: str
    model: str | None = None


class ResponseRequest(BaseModel):
    model: str | None = None
    input: str | list[dict[str, Any]]
    tools: list[dict[str, Any]] | None = None
    temerature: float | None = None


class ToolCall(BaseModel):
    call_id: str
    name: str
    arguments: str


class ResponseResult(BaseModel):
    text: str | None = None
    tool_calls: list[ToolCall]
    model: str | None = None
    response_id: str | None = None
    output: list[dict[str, Any]]
