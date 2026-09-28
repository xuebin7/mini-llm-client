import json
from typing import Any

from mini_llm.agent.protocols import ResponsesClient
from mini_llm.models import ResponseRequest


class LLMSummarizer:
    def __init__(
        self,
        client: ResponsesClient,
        *,
        model: str,
    ) -> None:
        self.client = client
        self.model = model

    def summarize(
        self,
        items: list[dict[str, Any]],
    ) -> str:
        history = json.dumps(
            items,
            ensure_ascii=False,
            default=str,
        )

        prompt = (
            "Summarize the following agent history for future continuation.\n\n"
            "Preserve:\n"
            "- the user's goal\n"
            "- important findings\n"
            "- completed actions\n"
            "- failed attempts\n"
            "- unresolved issues\n"
            "- information needed for next steps\n\n"
            "Be concise and factual.\n\n"
            f"History:\n{history}"
        )

        request = ResponseRequest(
            model=self.model,
            input=prompt,
        )

        response = self.client.responses(request=request)

        return response.text or ""
