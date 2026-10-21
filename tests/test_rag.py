from mini_llm.models import ChatRequest, ChatResponse
from mini_llm.rag.models import Chunk
from mini_llm.rag.rag import RAGPipeline


class StubRetriever:
    def retrieve(
        self,
        query: str,
        top_k: int = 3,
    ) -> list[Chunk]:
        return [
            Chunk(
                id="doc-1:0",
                document_id="doc-1",
                text="Python is used in AI.",
                index=0,
            ),
            Chunk(
                id="doc-2:0",
                document_id="doc-2",
                text="Java is used in backend systems.",
                index=0,
            ),
        ]


class FakeLLMClient:
    def __init__(self) -> None:
        self.request: ChatRequest | None = None

    def chat(
        self,
        request: ChatRequest,
    ) -> ChatResponse:
        self.request = request

        return ChatResponse(
            content="Python is widely used in AI.",
            model="test-model",
        )


def test_rag_pipeline_builds_context():
    client = FakeLLMClient()

    pipeline = RAGPipeline(
        retriever=StubRetriever(),
        client=client,
        model="test-model",
    )

    context = pipeline.build_context(
        query="What is Python used for?",
        top_k=2,
    )

    assert context == ("Python is used in AI.\n\nJava is used in backend systems.")


def test_rag_pipeline_answers_using_llm():
    client = FakeLLMClient()

    pipeline = RAGPipeline(
        retriever=StubRetriever(),
        client=client,
        model="test-model",
    )

    answer = pipeline.answer(
        query="What is Python used for?",
        top_k=2,
    )

    assert answer == "Python is widely used in AI."
    assert client.request is not None

    user_message = client.request.messages[1]

    assert "Python is used in AI." in user_message.content
    assert "What is Python used for?" in user_message.content
