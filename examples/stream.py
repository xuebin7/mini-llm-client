import os

from mini_llm import ChatRequest, LLMClient, Message

client = LLMClient(
    base_url=os.environ["LLM_BASE_URL"],
    api_key=os.environ["LLM_API_KEY"],
)

request = ChatRequest(
    model=os.environ["LLM_MODEL"],
    messages=[Message(role="user", content="用一句话解释什么是 Redis。")],
)

for text in client.stream(request):
    print(text, end="", flush=True)
print()
