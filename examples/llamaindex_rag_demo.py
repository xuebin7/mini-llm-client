from llama_index.core import Document, VectorStoreIndex


def main() -> None:
    documents = [
        Document(
            text=(
                "Python is widely used in artificial intelligence, "
                "data science, automation, and web development."
            ),
            metadata={
                "source": "python",
            },
        ),
        Document(
            text=(
                "Java is widely used in enterprise backend systems "
                "and large-scale business applications."
            ),
            metadata={
                "source": "java",
            },
        ),
        Document(
            text=(
                "Cooking involves ingredients, heat, and food preparation techniques."
            ),
            metadata={
                "source": "cooking",
            },
        ),
    ]

    index = VectorStoreIndex.from_documents(
        documents=documents,
    )

    query_engine = index.as_query_engine(
        similarity_top_k=2,
    )

    response = query_engine.query("What is Python commonly used for?")

    print(response)


if __name__ == "__main__":
    main()
