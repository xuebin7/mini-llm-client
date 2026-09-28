from typing import Any

from mini_llm.context.compaction import CompactionConfig, SlidingWindowCompactor


class FakeSummarizer:
    def summarize(
        self,
        items: list[dict[str, Any]],
    ) -> str:
        return "Old conversation summary"


class RecordingSummarizer:
    def summarize(
        self,
        items: list[dict[str, Any]],
    ) -> str:
        self.received_items = items
        return "summary"


def test_compact_empty_lines():
    summarizer = FakeSummarizer()
    sliding_window_compactor = SlidingWindowCompactor(
        config=CompactionConfig(keep_recent=5),
        summarizer=summarizer,
    )
    empty_items = []
    compacted_result = sliding_window_compactor.compact(items=empty_items)
    assert compacted_result == empty_items


def test_compact_items_within_window():
    summarizer = FakeSummarizer()
    sliding_window_compactor = SlidingWindowCompactor(
        config=CompactionConfig(keep_recent=5),
        summarizer=summarizer,
    )
    items = [
        {"role": "user", "content": "1"},
        {"role": "assistant", "content": "2"},
    ]
    compacted_result = sliding_window_compactor.compact(items=items)

    assert compacted_result == items


def test_compact_keeps_recent_items():
    summarizer = FakeSummarizer()
    sliding_window_compactor = SlidingWindowCompactor(
        config=CompactionConfig(keep_recent=3),
        summarizer=summarizer,
    )
    items = [
        {"role": "user", "content": "1"},
        {"role": "assistant", "content": "2"},
        {"role": "tool", "content": "3"},
        {"role": "assistant", "content": "4"},
        {"role": "tool", "content": "5"},
    ]

    compacted_result = sliding_window_compactor.compact(items=items)
    assert len(compacted_result) == 4

    assert compacted_result[0]["role"] == "system"
    assert compacted_result[0]["content"] == "Old conversation summary"
    assert compacted_result[1:] == items[-3:]


def test_compact_summarizes_old_items():
    summarizer = FakeSummarizer()
    sliding_window_compactor = SlidingWindowCompactor(
        config=CompactionConfig(keep_recent=2),
        summarizer=summarizer,
    )
    items = [
        {"role": "user", "content": "1"},
        {"role": "assistant", "content": "2"},
        {"role": "tool", "content": "3"},
        {"role": "assistant", "content": "4"},
        {"role": "tool", "content": "5"},
    ]

    compacted_result = sliding_window_compactor.compact(items=items)

    assert len(compacted_result) == 3
    assert compacted_result[0]["role"] == "system"
    assert "Old conversation summary" in compacted_result[0]["content"]
    assert compacted_result[1:] == items[-2:]


def test_compact_received_items():
    summarizer = RecordingSummarizer()
    sliding_window_compactor = SlidingWindowCompactor(
        config=CompactionConfig(keep_recent=2),
        summarizer=summarizer,
    )

    items = [
        {"role": "user", "content": "1"},
        {"role": "assistant", "content": "2"},
        {"role": "tool", "content": "3"},
        {"role": "assistant", "content": "4"},
        {"role": "tool", "content": "5"},
    ]

    sliding_window_compactor.compact(items=items)
    assert summarizer.received_items == items[:-2]
