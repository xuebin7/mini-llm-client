from typing import Any, Protocol


class ConversationSummarizer(Protocol):
    def summarize(
        self,
        items: list[dict[str, Any]],
    ) -> str: ...
