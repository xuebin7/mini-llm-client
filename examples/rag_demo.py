from mini_llm.rag.embeddings import FakeEmbeddingModel
from mini_llm.rag.indexer import Indexer
from mini_llm.rag.models import Document
from mini_llm.rag.retriever import Retriever
from mini_llm.rag.vector_store import InMemoryVectorStore


class KeywordEmbeddingModel:
    def embed(
        self,
        texts: list[str],
    ) -> list[list[float]]:
        return [self._embed_one(text) for text in texts]

    def _embed_one(self, text: str) -> list[float]:
        text = text.lower()

        return [
            float("python" in text),
            float("java" in text),
            float("ai" in text),
            float("backend" in text),
            float("cooking" in text),
        ]


def main() -> None:
    embedding_model = FakeEmbeddingModel()
    vector_store = InMemoryVectorStore()

    indexer = Indexer(
        embedding_model=embedding_model,
        vector_store=vector_store,
    )

    retriever = Retriever(
        embedding_model=embedding_model,
        vector_store=vector_store,
    )

    documents = [
        Document(
            id="python",
            text="Python is a programming language. "
            "It is widely used in AI and data science.",
            metadata={"source": "demo"},
        ),
        Document(
            id="java",
            text="Java is a programming language. "
            "It is widely used in enterprise backend systems.",
            metadata={"source": "demo"},
        ),
        Document(
            id="cooking",
            text="Cooking requires ingredients, heat, "
            "and different preparation techniques.",
            metadata={"source": "demo"},
        ),
    ]

    for document in documents:
        indexer.index(
            document=document,
            chunk_size=200,
            chunk_overlap=0,
        )

    results = retriever.retrieve(
        query="Python AI",
        top_k=2,
    )

    for chunk in results:
        print(f"id={chunk.id}, document_id={chunk.document_id}, text={chunk.text}")


if __name__ == "__main__":
    main()
