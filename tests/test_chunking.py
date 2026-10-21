import pytest

from mini_llm.rag.chunking import chunk_document, chunk_text
from mini_llm.rag.models import Document


def test_chunk_text_empty():
    text = ""
    chunks = chunk_text(
        text=text,
        chunk_size=4,
        chunk_overlap=2,
    )

    assert chunks == []


def test_chunk_text_short_text():
    text = "abc"
    chunks = chunk_text(
        text=text,
        chunk_size=4,
        chunk_overlap=1,
    )

    assert chunks == [
        "abc",
    ]


def test_chunk_text_without_overlap():
    text = "abcdefgh"
    chunks = chunk_text(
        text=text,
        chunk_size=4,
        chunk_overlap=0,
    )

    assert chunks == [
        "abcd",
        "efgh",
    ]


def test_chunk_text_with_overlap():
    text = "abcdefghij"
    chunks = chunk_text(
        text=text,
        chunk_size=4,
        chunk_overlap=1,
    )

    assert chunks == [
        "abcd",
        "defg",
        "ghij",
    ]


def test_chunk_text_rejects_invalid_arguments():
    text = "abcd"

    with pytest.raises(ValueError):
        chunk_text(
            text=text,
            chunk_size=0,
            chunk_overlap=0,
        )

    with pytest.raises(ValueError):
        chunk_text(
            text=text,
            chunk_size=4,
            chunk_overlap=-1,
        )

    with pytest.raises(ValueError):
        chunk_text(
            text=text,
            chunk_size=4,
            chunk_overlap=4,
        )


def test_chunk_document_creates_chunks():
    document = Document(
        id="doc-1",
        text="abcdefgh",
        metadata={"source": "demo"},
    )

    chunks = chunk_document(
        document=document,
        chunk_size=4,
        chunk_overlap=1,
    )

    assert [chunk.text for chunk in chunks] == [
        "abcd",
        "defg",
        "gh",
    ]


def test_chunk_document_preserves_document_id():
    document = Document(
        id="doc-1",
        text="abcdefgh",
        metadata={"source": "demo"},
    )

    chunks = chunk_document(
        document=document,
        chunk_size=4,
        chunk_overlap=1,
    )

    assert chunks[0].document_id == "doc-1"
    assert chunks[0].id == "doc-1:0"
    assert chunks[0].index == 0

    assert chunks[1].id == "doc-1:1"
    assert chunks[1].index == 1


def test_chunk_document_copies_metadata():
    document = Document(
        id="doc-1",
        text="abcdefgh",
        metadata={"source": "demo"},
    )

    chunks = chunk_document(
        document,
        chunk_size=4,
        chunk_overlap=0,
    )

    assert chunks[0].metadata == {"source": "demo"}
    assert chunks[1].metadata == {"source": "demo"}
