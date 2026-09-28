from dataclasses import dataclass


@dataclass(frozen=True)
class TokenBudget:
    max_tokens: int
    reserve_tokens: int

    @property
    def input_limit(self) -> int:
        return self.max_tokens - self.reserve_tokens

    def exceeded(self, input_tokens: int) -> bool:
        return input_tokens > self.input_limit
