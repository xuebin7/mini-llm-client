from langchain_deepseek import ChatDeepSeek


def main() -> None:
    model = ChatDeepSeek(
        model="deepseek-chat",
        temperature=0,
    )

    response = model.invoke(
        [
            (
                "system",
                "You are a concise programming assistant.",
            ),
            (
                "human",
                "Explain what a Python decorator is in one sentence.",
            ),
        ]
    )

    print(response.content)


if __name__ == "__main__":
    main()
