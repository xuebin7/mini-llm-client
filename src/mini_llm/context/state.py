from dataclasses import dataclass, field
from typing import Any


def _new_input_items() -> list[dict[str, Any]]:
    return []


@dataclass
class TaskState:
    input_items: list[dict[str, Any]] = field(default_factory=_new_input_items)
    step: int = 0
    input_tokens: int = 0
    compact_count: int = 0
