from typing import Literal, TypedDict

from langgraph.graph import END, START, StateGraph


class State(TypedDict):
    message: str


def uppercase(state: State) -> State:
    return {
        "message": state["message"].upper(),
    }


def add_prefix(state: State) -> State:
    return {"message": f"Processed: {state['message']}"}


def route_message(
    state: State,
) -> Literal["uppercase", "add_prefix"]:
    if "hello" in state["message"].lower():
        return "uppercase"

    return "add_prefix"


def main() -> None:
    builder = StateGraph(State)

    builder.add_node(
        "uppercase",
        uppercase,
    )

    builder.add_node(
        "add_prefix",
        add_prefix,
    )

    builder.add_conditional_edges(
        START,
        route_message,
    )

    builder.add_edge(
        "uppercase",
        END,
    )

    builder.add_edge(
        "add_prefix",
        END,
    )

    graph = builder.compile()

    result = graph.invoke(
        {
            "message": "hello",
        }
    )

    print(result)


if __name__ == "__main__":
    main()
