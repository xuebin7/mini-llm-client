from mini_llm.rag.embeddings import FakeEmbeddingModel


def test_fake_embedding_returns_one_vector_per_text():
    model = FakeEmbeddingModel()

    result = model.embed(["hello", "world"])

    assert len(result) == 2


def test_fake_embedding_vectors_have_consistent_dimensions():
    model = FakeEmbeddingModel()

    result = model.embed(["hello", "abc", "python"])

    assert all(len(vector) == 3 for vector in result)


def test_fake_embedding_is_deterministic():
    model = FakeEmbeddingModel()

    first = model.embed(["hello"])
    second = model.embed(["hello"])

    assert first == second
