import pytest

from mini_llm.rag.indexer import Indexer
from mini_llm.rag.models import Document
from mini_llm.rag.vector_store import InMemoryVectorStore


class StubEmbeddingModel:
    def embed(
        self,
        texts: list[str],
    ) -> list[list[float]]:
        return [[float(index), 1.0] for index, _ in enumerate(texts)]


def test_indexer_indexes_document_chunks():
    store = InMemoryVectorStore()

    indexer = Indexer(
        embedding_model=StubEmbeddingModel(),
        vector_store=store,
    )

    document = Document(
        id="doc-1",
        text="abcdefgh",
    )

    indexer.index(
        document=document,
        chunk_size=4,
        chunk_overlap=0,
    )

    results = store.search(
        query_embedding=[0.0, 1.0],
        top_k=10,
    )

    assert len(results) == 2
    assert [chunk.text for chunk in results] == [
        "abcd",
        "efgh",
    ]


class BrokenEmbeddingModel:
    def embed(
        self,
        texts: list[str],
    ) -> list[list[float]]:
        return [[1.0, 0.0]]


def test_indexer_rejects_embedding_count_mismatch():
    store = InMemoryVectorStore()

    indexer = Indexer(
        embedding_model=BrokenEmbeddingModel(),
        vector_store=store,
    )

    document = Document(
        id="doc-1",
        text="abcdefgh",
    )

    with pytest.raises(ValueError):
        indexer.index(
            document=document,
            chunk_size=4,
            chunk_overlap=0,
        )
