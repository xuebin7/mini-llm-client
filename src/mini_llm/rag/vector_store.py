import math
from dataclasses import dataclass

from mini_llm.rag.models import Chunk


@dataclass(frozen=True)
class VectorEntry:
    chunk: Chunk
    embedding: list[float]


def cosine_similarity(
    a: list[float],
    b: list[float],
) -> float:
    if len(a) != len(b):
        raise ValueError("vectors must have the same dimension")

    if not a:
        raise ValueError("vectors must not be empty")

    dot_product = sum(x * y for x, y in zip(a, b, strict=True))

    norm_a = math.sqrt(sum(x * x for x in a))
    norm_b = math.sqrt(sum(y * y for y in b))

    if norm_a == 0 or norm_b == 0:
        raise ValueError("zero vector is not allowed")

    return dot_product / (norm_a * norm_b)


class InMemoryVectorStore:
    def __init__(self) -> None:
        self._entries: list[VectorEntry] = []

    def add(
        self,
        chunk: Chunk,
        embedding: list[float],
    ) -> None:
        self._entries.append(
            VectorEntry(
                chunk=chunk,
                embedding=embedding,
            )
        )

    def search(
        self,
        query_embedding: list[float],
        top_k: int = 3,
    ) -> list[Chunk]:
        scored = [
            (
                cosine_similarity(
                    query_embedding,
                    entry.embedding,
                ),
                entry.chunk,
            )
            for entry in self._entries
        ]

        scored.sort(
            key=lambda item: item[0],
            reverse=True,
        )

        return [chunk for _, chunk in scored[:top_k]]
