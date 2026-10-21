from typing import Protocol

from mini_llm.models import ChatRequest, ChatResponse, Message
from mini_llm.rag.retriever import RetrieverLike


class ChatClient(Protocol):
    def chat(
        self,
        request: ChatRequest,
    ) -> ChatResponse: ...


class RAGPipeline:
    def __init__(
        self,
        retriever: RetrieverLike,
        client: ChatClient,
        model: str,
    ) -> None:
        self._retriever = retriever
        self._client = client
        self._model = model

    def build_context(
        self,
        query: str,
        top_k: int = 3,
    ) -> str:
        chunks = self._retriever.retrieve(
            query=query,
            top_k=top_k,
        )

        return "\n\n".join(chunk.text for chunk in chunks)

    def answer(
        self,
        query: str,
        top_k: int = 3,
    ) -> str:
        context = self.build_context(
            query=query,
            top_k=top_k,
        )

        request = ChatRequest(
            model=self._model,
            messages=[
                Message(
                    role="system",
                    content=(
                        "Answer the user's question using only the provided context. "
                        "If the context does not contain the answer, say you do not know."
                    ),
                ),
                Message(
                    role="user",
                    content=(f"Context:\n{context}\n\nQuestion:\n{query}"),
                ),
            ],
        )

        response = self._client.chat(request)

        return response.content
