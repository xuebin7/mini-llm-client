import json

from pydantic import BaseModel, ValidationError
import pytest
from mini_llm.structured import parse_structured_output

class Person(BaseModel):
    name: str
    age: int
    job: str

def test_parse_structured_output():
    content = '{"name": "张三", "age": "30", "job": "Java后端工程师"}'
    result = parse_structured_output(content, Person)

    assert isinstance(result, Person)
    assert result.name == "张三"
    assert result.age == 30
    assert result.job == "Java后端工程师"

from pydantic import BaseModel, ValidationError
import pytest
from mini_llm.structured import parse_structured_output

class Person(BaseModel):
    name: str
    age: int
    job: str

def test_parse_structured_output():
    content = '{"name": "张三", "age": 30, "job": "Java后端工程师"}'
    result = parse_structured_output(content, Person)

    assert isinstance(result, Person)
    assert result.name == "张三"
    assert result.age == 30
    assert result.job == "Java后端工程师"

def test_invalid_structured_output():
    content = '{"name": "张三", "age": "abc", "job": "Java后端工程师"}'

    with pytest.raises(ValidationError):
        parse_structured_output(
            content,
            Person,
        )
        
def test_invalid_json():
    content = """
    {
        "name": "张三",
        "age": 30,
    }
    """
    
    with pytest.raises(json.JSONDecodeError):
        parse_structured_output(
            content,
            Person
        )
    