from dataclasses import dataclass


@dataclass(frozen=True)
class TruncationConfig:
    max_chars: int
    head_chars: int
    tail_chars: int


class ToolResultTruncator:
    def __init__(self, config: TruncationConfig) -> None:
        self.config = config

    def truncate(self, text: str) -> str:
        if not text:
            return ""

        if len(text) <= self.config.max_chars:
            return text
        else:
            return (
                text[: self.config.head_chars]
                + "\n"
                + "truncated"
                + "\n"
                + text[self.config.tail_chars :]
            )
