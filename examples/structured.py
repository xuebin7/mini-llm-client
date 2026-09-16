import os

from pydantic import BaseModel

from mini_llm.client import LLMClient
from mini_llm.models import ChatRequest, Message

class Person(BaseModel):
    name: str
    age: int
    job: str

client = LLMClient(
    base_url=os.environ["LLM_BASE_URL"],
    api_key=os.environ["LLM_API_KEY"],
)

request = ChatRequest(
    model=os.environ["LLM_MODEL"],
    messages=[
        Message(
            role="user",
            content=(
                "虚构一个Java后端工程师。"
                "请以 JSON 格式返回结果。"
                "只返回 JSON，不要 Markdown，不要使用 ```。"
                '字段必须为 name、age、job。'
                '例如：{"name":"张三","age":28,"job":"Java后端工程师"}'
            )
        )
    ]
)

person = client.structured(
    request,
    Person
)

print(person)
print(person.name)
print(person.age)
print(person.job)