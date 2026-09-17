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
