import json
from typing import TypeVar

from pydantic import BaseModel, schema

T = TypeVar("T", bound=BaseModel)

def parse_structured_output(
    content: str,
    schema: type[T]
) -> T:
    print("========== LLM RAW OUTPUT ==========")
    print(repr(content))
    print("====================================")
    data = json.loads(content)

    return schema.model_validate(data)

