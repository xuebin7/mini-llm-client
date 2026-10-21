import os

from mini_llm.client import LLMClient
from mini_llm.rag.indexer import Indexer
from mini_llm.rag.models import Document
from mini_llm.rag.rag import RAGPipeline
from mini_llm.rag.retriever import Retriever
from mini_llm.rag.vector_store import InMemoryVectorStore


class KeywordEmbeddingModel:
    def embed(
        self,
        texts: list[str],
    ) -> list[list[float]]:
        return [self._embed_one(text) for text in texts]

    def _embed_one(
        self,
        text: str,
    ) -> list[float]:
        text = text.lower()

        return [
            float("python" in text),
            float("java" in text),
            float("ai" in text),
            float("backend" in text),
            float("cooking" in text),
        ]


def main() -> None:
    embedding_model = KeywordEmbeddingModel()
    vector_store = InMemoryVectorStore()

    indexer = Indexer(
        embedding_model=embedding_model,
        vector_store=vector_store,
    )

    documents = [
        Document(
            id="python",
            text=(
                "Python is widely used in artificial intelligence, "
                "data science, automation, and web development."
            ),
        ),
        Document(
            id="java",
            text=(
                "Java is widely used in enterprise backend systems "
                "and large-scale business applications."
            ),
        ),
        Document(
            id="cooking",
            text=(
                "Cooking involves ingredients, heat, and food preparation techniques."
            ),
        ),
    ]

    for document in documents:
        indexer.index(
            document=document,
            chunk_size=200,
            chunk_overlap=0,
        )

    retriever = Retriever(
        embedding_model=embedding_model,
        vector_store=vector_store,
    )

    client = LLMClient(
        base_url=os.environ["LLM_BASE_URL"],
        api_key=os.environ["LLM_API_KEY"],
    )

    pipeline = RAGPipeline(
        retriever=retriever,
        client=client,
        model=os.environ["LLM_MODEL"],
    )

    answer = pipeline.answer(
        query="What is Python commonly used for?",
        top_k=2,
    )

    print(answer)


if __name__ == "__main__":
    main()
