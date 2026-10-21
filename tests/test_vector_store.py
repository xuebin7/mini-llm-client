import pytest

from mini_llm.rag.models import Chunk
from mini_llm.rag.vector_store import InMemoryVectorStore, cosine_similarity


def test_cosine_similarity_identical_vectors():
    result = cosine_similarity(
        [1.0, 0.0],
        [1.0, 0.0],
    )

    assert result == pytest.approx(1.0)


def test_cosine_similarity_orthogonal_vectors():
    result = cosine_similarity(
        [1.0, 0.0],
        [0.0, 1.0],
    )

    assert result == pytest.approx(0.0)


def test_cosine_similarity_rejects_different_dimensions():
    with pytest.raises(ValueError):
        cosine_similarity(
            [1.0, 0.0],
            [1.0, 0.0, 0.0],
        )


def test_cosine_similaryty_rejects_zero_vector():
    with pytest.raises(ValueError):
        cosine_similarity(
            [0.0, 0.0],
            [1.0, 0.0],
        )


def test_vector_store_add_and_search():
    store = InMemoryVectorStore()

    chunk = Chunk(
        id="doc-1:0",
        document_id="doc-1",
        text="hello",
        index=0,
    )

    store.add(
        chunk=chunk,
        embedding=[1.0, 0.0],
    )

    results = store.search(
        query_embedding=[1.0, 0.0],
    )

    assert results == [chunk]


def test_vector_store_returns_most_similar_fist():
    store = InMemoryVectorStore()

    chunk_a = Chunk(
        id="doc-1:0",
        document_id="doc-1",
        text="A",
        index=0,
    )

    chunk_b = Chunk(
        id="doc-1:1",
        document_id="doc-1",
        text="B",
        index=1,
    )

    chunk_c = Chunk(
        id="doc-1:2",
        document_id="doc-1",
        text="C",
        index=2,
    )

    store.add(chunk_a, [1.0, 0.0])
    store.add(chunk_b, [0.8, 0.2])
    store.add(chunk_c, [0.0, 1.0])

    results = store.search(
        query_embedding=[1.0, 0.0],
        top_k=3,
    )

    assert results == [
        chunk_a,
        chunk_b,
        chunk_c,
    ]


def test_vector_store_respects_top_k():
    store = InMemoryVectorStore()

    chunk_a = Chunk(
        id="doc-1:0",
        document_id="doc-1",
        text="A",
        index=0,
    )

    chunk_b = Chunk(
        id="doc-1:1",
        document_id="doc-1",
        text="B",
        index=1,
    )

    chunk_c = Chunk(
        id="doc-1:2",
        document_id="doc-1",
        text="C",
        index=2,
    )

    store.add(chunk_a, [1.0, 0.0])
    store.add(chunk_b, [0.8, 0.2])
    store.add(chunk_c, [0.0, 1.0])

    results = store.search(
        query_embedding=[1.0, 0.0],
        top_k=2,
    )

    assert results == [
        chunk_a,
        chunk_b,
    ]
