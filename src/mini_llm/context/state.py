from dataclasses import dataclass, field
from typing import Any


@dataclass
class TaskState:
    input_items: list[dict[str, Any]] = field(default_factory=list)
    step: int = 0
    input_tokens: int = 0
    compact_count: int = 0
