from datetime import UTC, datetime
from enum import StrEnum
from uuid import uuid4

from pydantic import BaseModel, Field

from mini_llm.harness.events import Event


class TurnStatus(StrEnum):
    RUNNING = "running"
    SUCCEEDED = "succeeded"
    FAILED = "failed"


class Turn(BaseModel):
    id: str = Field(default_factory=lambda: f"turn-{uuid4()}")
    thread_id: str
    status: TurnStatus = TurnStatus.RUNNING
    created_at: datetime = Field(default_factory=lambda: datetime.now(UTC))
    completed_at: datetime | None = None


class Thread(BaseModel):
    id: str = Field(default_factory=lambda: f"thread-{uuid4()}")
    created_at: datetime = Field(default_factory=lambda: datetime.now(UTC))
    turns: list[Turn] = Field(default_factory=list)


class TurnResult(BaseModel):
    turn: Turn
    output: str
    events: list[Event] = Field(default_factory=list)
