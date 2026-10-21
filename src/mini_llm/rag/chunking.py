from mini_llm.rag.models import Chunk, Document


def chunk_text(
    text: str,
    chunk_size: int = 500,
    chunk_overlap: int = 50,
) -> list[str]:
    if chunk_size <= 0:
        raise ValueError("chunk_size must be greater than 0")

    if chunk_overlap < 0:
        raise ValueError("chunk_overlap must be >= 0")

    if chunk_overlap >= chunk_size:
        raise ValueError("chunk_overlap must be less than chunk_size")

    if text.strip() == "":
        return []

    chunks: list[str] = []

    step = chunk_size - chunk_overlap
    start = 0

    while start < len(text):
        end = start + chunk_size

        chunks.append(text[start:end])

        if end >= len(text):
            break

        start += step

    return chunks


def chunk_document(
    document: Document,
    chunk_size: int = 500,
    chunk_overlap: int = 50,
) -> list[Chunk]:
    texts = chunk_text(
        document.text,
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
    )

    return [
        Chunk(
            id=f"{document.id}:{index}",
            document_id=document.id,
            text=text,
            index=index,
            metadata=dict(document.metadata),
        )
        for index, text in enumerate(texts)
    ]


if __name__ == "__main__":
    chunk_text(
        text="abcdefghij",
        chunk_size=4,
        chunk_overlap=1,
    )
