from typing import Protocol


class EmbeddingModel(Protocol):
    def embed(self, texts: list[str]) -> list[list[float]]: ...


class FakeEmbeddingModel:
    def embed(self, texts: list[str]) -> list[list[float]]:
        return [
            [
                float(len(text)),
                float(text.count("a")),
                float(text.count("e")),
            ]
            for text in texts
        ]
