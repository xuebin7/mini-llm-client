from mini_llm.context.llm_summarizer import LLMSummarizer


class FakeResponse:
    def __init__(self, text: str) -> None:
        self.text = text


class FakeClient:
    def __init__(self) -> None:
        self.last_request = None

    def responses(self, request):
        self.last_request = request

        return FakeResponse("The user wants to fix a login bug.")


def test_llm_summarizer_returns_response_text():
    client = FakeClient()

    summarizer = LLMSummarizer(
        client=client,
        model="test-model",
    )

    items = [
        {
            "role": "user",
            "content": "Fix the login bug.",
        },
        {
            "role": "assistant",
            "content": "I found LoginService.",
        },
    ]

    result = summarizer.summarize(items)

    assert result == "The user wants to fix a login bug."


def test_llm_summarizer_includes_history_in_request():
    client = FakeClient()

    summarizer = LLMSummarizer(
        client=client,
        model="test-model",
    )

    items = [
        {
            "role": "user",
            "content": "Fix the login bug.",
        }
    ]

    summarizer.summarize(items)

    assert client.last_request is not None


def test_llm_summarizer_prompt_contains_summary_instructions():
    client = FakeClient()

    summarizer = LLMSummarizer(
        client=client,
        model="test-model",
    )

    summarizer.summarize([{"role": "user", "content": "hello"}])

    prompt = client.last_request.input

    assert "user's goal" in prompt
    assert "failed attempts" in prompt
    assert "unresolved issues" in prompt
