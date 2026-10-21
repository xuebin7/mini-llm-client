from mini_llm.rag.models import Chunk
from mini_llm.rag.retriever import Retriever
from mini_llm.rag.vector_store import InMemoryVectorStore


class StubEmbeddingModel:
    def embed(
        self,
        texts: list[str],
    ) -> list[list[float]]:
        vectors = {
            "python": [1.0, 0.0],
            "java": [0.8, 0.2],
            "cooking": [0.0, 1.0],
        }

        return [vectors[text] for text in texts]


def test_retriever_returns_relevant_chunks():
    store = InMemoryVectorStore()

    python_chunk = Chunk(
        id="doc-1:0",
        document_id="doc-1",
        text="Python content",
        index=0,
    )

    java_chunk = Chunk(
        id="doc-2:0",
        document_id="doc-2",
        text="Java content",
        index=0,
    )

    cooking_chunk = Chunk(
        id="doc-3:0",
        document_id="doc-3",
        text="Cooking content",
        index=0,
    )

    store.add(
        python_chunk,
        [1.0, 0.0],
    )
    store.add(
        java_chunk,
        [0.8, 0.2],
    )
    store.add(
        cooking_chunk,
        [0.0, 1.0],
    )

    retriever = Retriever(
        embedding_model=StubEmbeddingModel(),
        vector_store=store,
    )

    results = retriever.retrieve(
        query="python",
        top_k=3,
    )

    assert results == [
        python_chunk,
        java_chunk,
        cooking_chunk,
    ]


def test_retriever_respects_top_k():
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

    store.add(
        chunk_a,
        [1.0, 0.0],
    )
    store.add(
        chunk_b,
        [0.8, 0.2],
    )

    retriever = Retriever(
        embedding_model=StubEmbeddingModel(),
        vector_store=store,
    )

    results = retriever.retrieve(
        "python",
        top_k=1,
    )

    assert results == [chunk_a]
