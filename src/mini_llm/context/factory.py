from mini_llm.agent.protocols import ResponsesClient
from mini_llm.context.budget import TokenBudget
from mini_llm.context.compaction import CompactionConfig, SlidingWindowCompactor
from mini_llm.context.llm_summarizer import LLMSummarizer
from mini_llm.context.manager import ContextManager
from mini_llm.context.tokens import ApproxTokenCounter
from mini_llm.context.truncation import ToolResultTruncator, TruncationConfig


def create_context_manager(
    client: ResponsesClient,
    *,
    model: str,
) -> ContextManager:
    summarizer = LLMSummarizer(
        client=client,
        model=model,
    )

    max_tokens = 128_000
    reserve_tokens = 16_000
    keep_recent = 6

    compactor = SlidingWindowCompactor(
        config=CompactionConfig(keep_recent=keep_recent),
        summarizer=summarizer,
    )

    context_manager = ContextManager(
        budget=TokenBudget(
            max_tokens=max_tokens,
            reserve_tokens=reserve_tokens,
        ),
        token_counter=ApproxTokenCounter(),
        truncator=ToolResultTruncator(
            config=TruncationConfig(
                max_chars=20_000,
                head_chars=5_000,
                tail_chars=10_000,
            )
        ),
        compactor=compactor,
    )

    return context_manager
