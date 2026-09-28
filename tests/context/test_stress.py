from typing import Any

from mini_llm.context.budget import TokenBudget
from mini_llm.context.compaction import CompactionConfig, SlidingWindowCompactor
from mini_llm.context.manager import ContextManager
from mini_llm.context.tokens import ApproxTokenCounter
from mini_llm.context.truncation import ToolResultTruncator, TruncationConfig


class FakeSummarizer:
    def summarize(
        self,
        items: list[dict[str, Any]],
    ) -> str:
        return (
            "The user asked to analyze a large test failure. "
            "Earlier tool output was compacted."
        )


def test_context_manager_compacts_100k_character_history():
    max_tokens = 10_000
    reserve_tokens = 2_000
    max_chars = 20_000
    head_chars = 5_000
    tail_chars = 10_000
    keep_recent = 2

    context_manager = ContextManager(
        budget=TokenBudget(
            max_tokens=max_tokens,
            reserve_tokens=reserve_tokens,
        ),
        token_counter=ApproxTokenCounter(),
        truncator=ToolResultTruncator(
            config=TruncationConfig(
                max_chars=max_chars,
                head_chars=head_chars,
                tail_chars=tail_chars,
            )
        ),
        compactor=SlidingWindowCompactor(
            config=CompactionConfig(keep_recent=keep_recent),
            summarizer=FakeSummarizer(),
        ),
    )

    huge_log = "x" * 100_000

    items = [
        {
            "role": "user",
            "content": "Analyze the test failure.",
        },
        {
            "role": "tool",
            "content": huge_log,
        },
        {
            "role": "assistant",
            "content": "I found an error in the test output.",
        },
        {
            "role": "tool",
            "content": "Latest short tool result.",
        },
    ]

    result = context_manager.prepare(items=items)

    assert result
    assert len(result) == 3
    assert result[0]["role"] == "system"
    assert len(str(result)) < len(str(items))
    assert result[1:] == items[-2:]


def test_tool_result_truncated_100k_characters():
    max_tokens = 128_000
    reserve_tokens = 16_000

    max_chars = 20_000
    head_chars = 5_000
    tail_chars = 10_000

    keep_recent = 6

    context_manager = ContextManager(
        budget=TokenBudget(
            max_tokens=max_tokens,
            reserve_tokens=reserve_tokens,
        ),
        token_counter=ApproxTokenCounter(),
        truncator=ToolResultTruncator(
            config=TruncationConfig(
                max_chars=max_chars,
                head_chars=head_chars,
                tail_chars=tail_chars,
            )
        ),
        compactor=SlidingWindowCompactor(
            config=CompactionConfig(keep_recent=keep_recent),
            summarizer=FakeSummarizer(),
        ),
    )

    huge_log = "x" * 100_000

    result = context_manager.truncate_tool_result(huge_log)

    assert len(result) < len(huge_log)
    assert result.startswith("x" * 5_000)
    assert result.endswith("x" * 10_000)
    assert "truncated" in result
