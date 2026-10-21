from typing import Protocol

from mini_llm.rag.embeddings import EmbeddingModel
from mini_llm.rag.models import Chunk
from mini_llm.rag.vector_store import InMemoryVectorStore


class Retriever:
    def __init__(
        self,
        embedding_model: EmbeddingModel,
        vector_store: InMemoryVectorStore,
    ) -> None:
        self._embedding_model = embedding_model
        self._vector_store = vector_store

    def retrieve(
        self,
        query: str,
        top_k: int = 3,
    ) -> list[Chunk]:
        embeddings = self._embedding_model.embed([query])

        if not embeddings:
            return []

        query_embedding = embeddings[0]

        return self._vector_store.search(
            query_embedding=query_embedding,
            top_k=top_k,
        )


class RetrieverLike(Protocol):
    def retrieve(
        self,
        query: str,
        top_k: int = 3,
    ) -> list[Chunk]: ...
