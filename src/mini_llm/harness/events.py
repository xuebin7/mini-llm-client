from __future__ import annotations

from datetime import UTC, datetime
from enum import StrEnum
from typing import Any

from pydantic import BaseModel, Field


class EventType(StrEnum):
    TURN_STARTED = "turn_started"
    TURN_COMPLETED = "turn_completed"
    TURN_FAILED = "turn_failed"

    AGENT_MESSAGE = "agent_message"

    TOOL_STARTED = "tool_started"
    TOOL_COMPLETED = "tool_completed"
    TOOL_FAILED = "tool_failed"

    STATE_CHANGED = "state_changed"


class Event(BaseModel):
    type: EventType
    timestamp: datetime = Field(default_factory=lambda: datetime.now(UTC))
    thread_id: str
    turn_id: str
    data: dict[str, Any] = Field(default_factory=dict)
