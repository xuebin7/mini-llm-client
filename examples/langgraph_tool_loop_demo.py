from typing import Annotated, TypedDict

from langchain_core.messages import AIMessage, AnyMessage, HumanMessage
from langchain_core.tools import tool
from langchain_deepseek import ChatDeepSeek
from langgraph.graph import END, START, StateGraph
from langgraph.graph.message import add_messages
from langgraph.prebuilt import ToolNode


class State(TypedDict):
    messages: Annotated[list[AnyMessage], add_messages]


@tool
def get_weather(city: str) -> str:
    """Get the weather for a city."""
    return f"The weather in {city} is sunny."


model = ChatDeepSeek(
    model="deepseek-flash",
    temperature=0,
)

tools = [get_weather]

model_with_tools = model.bind_tools(tools=tools)

tool_node = ToolNode(tools=tools)


def call_model(state: State) -> State:
    response = model_with_tools.invoke(state["messages"])

    return {
        "messages": [response],
    }


def route_after_model(
    state: State,
) -> str:
    last_message = state["messages"][-1]

    if isinstance(last_message, AIMessage) and last_message.tool_calls:
        return "tools"

    return END


def main() -> None:
    builder = StateGraph(State)

    builder.add_node(
        "llm",
        call_model,
    )

    builder.add_node(
        "tools",
        tool_node,
    )

    builder.add_edge(
        START,
        "llm",
    )

    builder.add_conditional_edges(
        "llm",
        route_after_model,
    )

    builder.add_edge(
        "tools",
        "llm",
    )

    graph = builder.compile()

    result = graph.invoke(
        {"messages": [HumanMessage(content="What's the weather in Beijing?")]}
    )

    for message in result["messages"]:
        print(type(message).__name__)
        print(message)
        print()


if __name__ == "__main__":
    main()
