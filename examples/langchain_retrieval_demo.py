from langchain_core.documents import Document
from langchain_core.embeddings import DeterministicFakeEmbedding
from langchain_core.vectorstores import InMemoryVectorStore
from langchain_text_splitters import RecursiveCharacterTextSplitter


def main() -> None:
    documents = [
        Document(
            page_content=(
                "Python is widely used in artificial intelligence, "
                "data science, automation, and web development."
            ),
            metadata={"source": "python"},
        ),
        Document(
            page_content=(
                "Java is widely used in enterprise backend systems "
                "and large-scale business applications."
            ),
            metadata={"source": "java"},
        ),
        Document(
            page_content=(
                "Cooking involves ingredients, heat, and food preparation techniques."
            ),
            metadata={"source": "cooking"},
        ),
    ]

    splitter = RecursiveCharacterTextSplitter(
        chunk_size=200,
        chunk_overlap=0,
    )

    chunks = splitter.split_documents(documents)

    embeddings = DeterministicFakeEmbedding(
        size=8,
    )

    vector_store = InMemoryVectorStore(
        embedding=embeddings,
    )

    vector_store.add_documents(chunks)

    retriever = vector_store.as_retriever(
        search_kwargs={"k": 2},
    )

    results = retriever.invoke("What is Python commonly used for?")

    for document in results:
        print(f"source={document.metadata.get('source')}")
        print(document.page_content)
        print()


if __name__ == "__main__":
    main()
