from dataclasses import dataclass, field


@dataclass(frozen=True)
class Document:
    id: str
    text: str
    metadata: dict[str, str] = field(default_factory=dict)


@dataclass(frozen=True)
class Chunk:
    id: str
    document_id: str
    text: str
    index: int
    metadata: dict[str, str] = field(default_factory=dict)
