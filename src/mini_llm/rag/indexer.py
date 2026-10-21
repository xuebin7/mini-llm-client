from mini_llm.rag.chunking import chunk_document
from mini_llm.rag.embeddings import EmbeddingModel
from mini_llm.rag.models import Document
from mini_llm.rag.vector_store import InMemoryVectorStore


class Indexer:
    def __init__(
        self,
        embedding_model: EmbeddingModel,
        vector_store: InMemoryVectorStore,
    ) -> None:
        self._embedding_model = embedding_model
        self._vector_store = vector_store

    def index(
        self,
        document: Document,
        chunk_size: int = 500,
        chunk_overlap: int = 50,
    ) -> None:
        chunks = chunk_document(
            document=document,
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap,
        )

        if not chunks:
            return

        embeddings = self._embedding_model.embed([chunk.text for chunk in chunks])

        if len(embeddings) != len(chunks):
            raise ValueError("embedding count must match chunk count")

        for chunk, embedding in zip(
            chunks,
            embeddings,
            strict=True,
        ):
            self._vector_store.add(
                chunk=chunk,
                embedding=embedding,
            )
