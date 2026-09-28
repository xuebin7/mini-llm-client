from dataclasses import dataclass
from typing import Any

from mini_llm.context.summarizer import ConversationSummarizer


@dataclass(frozen=True)
class CompactionConfig:
    keep_recent: int  # 压缩时，最近 keep_recent 个 item 原样保留。


class SlidingWindowCompactor:
    def __init__(
        self,
        config: CompactionConfig,
        summarizer: ConversationSummarizer,
    ) -> None:
        self.config = config
        self.summarizer = summarizer

    def compact(
        self,
        items: list[dict[str, Any]],
    ) -> list[dict[str, Any]]:
        if not items:
            return []

        if len(items) <= self.config.keep_recent:
            return items

        old_items = items[: -self.config.keep_recent]
        recent_items = items[-self.config.keep_recent :]

        summary = self.summarizer.summarize(items=old_items)
        summary_item = {
            "role": "system",
            "content": summary,
        }
        return [summary_item] + recent_items
