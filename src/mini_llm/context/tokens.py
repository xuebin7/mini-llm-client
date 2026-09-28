from typing import Protocol


class TokenCounter(Protocol):
    def count(self, text: str) -> int: ...


class ApproxTokenCounter:
    def count(self, text: str) -> int:
        if not text:
            return 0

        return max(1, len(text) // 4)
