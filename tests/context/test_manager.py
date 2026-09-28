from mini_llm.context.budget import TokenBudget
from mini_llm.context.compaction import CompactionConfig, SlidingWindowCompactor
from mini_llm.context.manager import ContextManager
from mini_llm.context.tokens import ApproxTokenCounter
from mini_llm.context.truncation import ToolResultTruncator, TruncationConfig


class FakeSummarizer:
    def summarize(self, items):
        return "summary"


def test_prepare_returns_original_items_when_within_budget():
    max_tokens = 1000
    reserve_tokens = 100
    context_manager = ContextManager(
        budget=TokenBudget(
            max_tokens=max_tokens,
            reserve_tokens=reserve_tokens,
        ),
        token_counter=ApproxTokenCounter(),
        truncator=ToolResultTruncator(
            config=TruncationConfig(
                max_chars=1000,
                head_chars=100,
                tail_chars=100,
            )
        ),
        compactor=SlidingWindowCompactor(
            config=CompactionConfig(keep_recent=3),
            summarizer=FakeSummarizer(),
        ),
    )

    items = [
        {"role": "user", "content": "hello"},
        {"role": "assistant", "content": "hi"},
    ]

    result = context_manager.prepare(items=items)
    assert result is items


def test_prepare_compacts_when_budget_exceeded():
    max_tokens = 10
    reserve_tokens = 2
    keep_recent = 2
    context_manager = ContextManager(
        budget=TokenBudget(
            max_tokens=max_tokens,
            reserve_tokens=reserve_tokens,
        ),
        token_counter=ApproxTokenCounter(),
        truncator=ToolResultTruncator(
            config=TruncationConfig(
                max_chars=1000,
                head_chars=100,
                tail_chars=100,
            )
        ),
        compactor=SlidingWindowCompactor(
            config=CompactionConfig(keep_recent=keep_recent),
            summarizer=FakeSummarizer(),
        ),
    )

    items = [
        {"role": "user", "content": "1"},
        {"role": "assistant", "content": "2"},
        {"role": "tool", "content": "3"},
        {"role": "assistant", "content": "4"},
        {"role": "tool", "content": "5"},
    ]

    result = context_manager.prepare(items=items)

    assert len(result) == keep_recent + 1
    assert result[0]["role"] == "system"
    assert result[1:] == items[-keep_recent:]


def test_truncate_tool_result():
    text = "x" * 100

    max_tokens = 1000
    reserve_tokens = 100
    keep_recent = 100
    context_manager = ContextManager(
        budget=TokenBudget(
            max_tokens=max_tokens,
            reserve_tokens=reserve_tokens,
        ),
        token_counter=ApproxTokenCounter(),
        truncator=ToolResultTruncator(
            config=TruncationConfig(
                max_chars=20,
                head_chars=5,
                tail_chars=5,
            )
        ),
        compactor=SlidingWindowCompactor(
            config=CompactionConfig(keep_recent=keep_recent),
            summarizer=FakeSummarizer(),
        ),
    )

    result = context_manager.truncate_tool_result(text=text)

    assert result.startswith("xxxxx")
    assert result.endswith("xxxxx")
    assert "truncated" in result
