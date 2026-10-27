from llama_index.core import (
    Document,
    MockEmbedding,
    Settings,
    VectorStoreIndex,
)
from llama_index.core.node_parser import SentenceSplitter


def main() -> None:
    Settings.embed_model = MockEmbedding(
        embed_dim=8,
    )

    Settings.text_splitter = SentenceSplitter(
        chunk_size=200,
        chunk_overlap=0,
    )

    documents = [
        Document(
            text=(
                "Python is widely used in artificial intelligence, "
                "data science, automation, and web development."
            ),
            metadata={"source": "python"},
        ),
        Document(
            text=(
                "Java is widely used in enterprise backend systems "
                "and large-scale business applications."
            ),
            metadata={"source": "java"},
        ),
        Document(
            text=(
                "Cooking involves ingredients, heat, and food preparation techniques."
            ),
            metadata={"source": "cooking"},
        ),
    ]

    index = VectorStoreIndex.from_documents(documents)

    retriever = index.as_retriever(
        similarity_top_k=2,
    )

    results = retriever.retrieve("What is Python commonly used for?")

    for result in results:
        print(f"score={result.score}, source={result.node.metadata.get('source')}")
        print(result.node.get_content())
        print()


if __name__ == "__main__":
    main()
