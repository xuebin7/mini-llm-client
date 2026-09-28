import json
from typing import Any

from mini_llm.context.budget import TokenBudget
from mini_llm.context.compaction import SlidingWindowCompactor
from mini_llm.context.tokens import TokenCounter
from mini_llm.context.truncation import ToolResultTruncator


class ContextManager:
    def __init__(
        self,
        budget: TokenBudget,
        token_counter: TokenCounter,
        truncator: ToolResultTruncator,
        compactor: SlidingWindowCompactor,
    ) -> None:
        self.budget = budget
        self.token_counter = token_counter
        self.truncator = truncator
        self.compactor = compactor

    def count_tokens(
        self,
        items: list[dict[str, Any]],
    ) -> int:
        text = json.dumps(
            items,
            ensure_ascii=False,
            default=str,
        )

        return self.token_counter.count(text)

    def needs_compaction(
        self,
        items: list[dict[str, Any]],
    ) -> bool:
        tokens = self.count_tokens(items)

        return self.budget.exceeded(tokens)

    def truncate_tool_result(self, text: str) -> str:
        return self.truncator.truncate(text)

    def prepare(
        self,
        items: list[dict[str, Any]],
    ) -> list[dict[str, Any]]:
        if not self.needs_compaction(items):
            return items

        return self.compactor.compact(items)
