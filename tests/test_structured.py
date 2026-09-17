import json

import httpx
import pytest
from pydantic import BaseModel, ConfigDict, RootModel, ValidationError

from mini_llm import ChatRequest, LLMClient, Message
from mini_llm.structured import parse_structured_output


class Person(BaseModel):
    name: str
    age: int
    job: str


@pytest.mark.parametrize("age", [30, "30"])
def test_parse_structured_output(age):
    content = json.dumps({"name": "Alice", "age": age, "job": "Engineer"})
    result = parse_structured_output(content, Person)
    assert result == Person(name="Alice", age=30, job="Engineer")


def test_invalid_structured_output():
    with pytest.raises(ValidationError):
        parse_structured_output(
            '{"name": "Alice", "age": "abc", "job": "Engineer"}', Person
        )


def test_invalid_json():
    with pytest.raises(json.JSONDecodeError):
        parse_structured_output('{"name": "Alice", "age": 30,}', Person)


def test_schema_format_is_callable_on_class_and_instance():
    client = LLMClient("https://example.com", "test-key")
    result = LLMClient.schema_to_response_format(Person)
    assert isinstance(result, dict)
    assert result == client.schema_to_response_format(Person)
    assert result["type"] == "json_schema"
    assert result["json_schema"]["strict"] is True
    schema = result["json_schema"]["schema"]
    assert schema["additionalProperties"] is False
    assert schema["required"] == ["name", "age", "job"]


def test_strict_schema_handles_nested_models_and_nullable_defaults():
    class Team(BaseModel):
        members: list[Person]
        note: str | None = None

    original = Team.model_json_schema()
    schema = LLMClient.schema_to_response_format(Team)["json_schema"]["schema"]
    assert schema["required"] == ["members", "note"]
    assert "default" not in schema["properties"]["note"]
    assert {"type": "null"} in schema["properties"]["note"]["anyOf"]
    assert schema["$defs"]["Person"]["additionalProperties"] is False
    assert Team.model_json_schema() == original


def test_strict_schema_handles_recursive_models():
    class Node(BaseModel):
        name: str
        children: list["Node"]

    schema = LLMClient.schema_to_response_format(Node)["json_schema"]["schema"]
    assert schema["$defs"]["Node"]["additionalProperties"] is False
    assert schema["$defs"]["Node"]["required"] == ["name", "children"]


def test_strict_schema_preserves_fields_named_like_schema_keywords():
    class Payload(BaseModel):
        default: str
        properties: str

    schema = LLMClient.schema_to_response_format(Payload)["json_schema"]["schema"]
    assert set(schema["properties"]) == {"default", "properties"}


def test_strict_schema_rejects_open_dictionaries():
    class Metadata(BaseModel):
        values: dict[str, str]

    with pytest.raises(ValueError, match="open dictionaries"):
        LLMClient.schema_to_response_format(Metadata)


def test_strict_schema_rejects_extra_allow():
    class OpenModel(BaseModel):
        model_config = ConfigDict(extra="allow")
        name: str

    with pytest.raises(ValueError, match="extra='allow'"):
        LLMClient.schema_to_response_format(OpenModel)


def test_strict_schema_rejects_non_object_root():
    with pytest.raises(ValueError, match="object model"):
        LLMClient.schema_to_response_format(RootModel[list[str]])


@pytest.mark.parametrize("mode", ["json_object", "json_schema"])
def test_structured_sends_correct_format_without_mutation_or_logging(
    monkeypatch, capsys, mode
):
    original_format = {"type": "text"}
    request = ChatRequest(
        model="test-model",
        messages=[Message(role="user", content="Return a person as JSON")],
        temperature=0,
        response_format=original_format,
    )
    before = request.model_dump()
    calls = []

    def fake_post(url, **kwargs):
        calls.append(kwargs["json"])
        return httpx.Response(
            200,
            json={
                "choices": [
                    {
                        "message": {
                            "content": '{"name":"Alice","age":30,"job":"Engineer"}'
                        }
                    }
                ]
            },
        )

    monkeypatch.setattr("mini_llm.client.httpx.post", fake_post)
    client = LLMClient("https://example.com", "test-key")
    # Exercise the default mode as well as the explicit schema mode.
    options = {} if mode == "json_object" else {"mode": mode}
    result = client.structured(request, Person, **options)
    assert result == Person(name="Alice", age=30, job="Engineer")
    expected = (
        {"type": "json_object"}
        if mode == "json_object"
        else (LLMClient.schema_to_response_format(Person))
    )
    assert calls[0]["response_format"] == expected
    assert calls[0]["temperature"] == 0
    assert request.model_dump() == before
    assert capsys.readouterr().out == ""


def test_structured_rejects_invalid_mode_before_request(monkeypatch):
    def unexpected_post(*args, **kwargs):
        pytest.fail("Invalid modes must not issue an HTTP request")

    monkeypatch.setattr("mini_llm.client.httpx.post", unexpected_post)
    request = ChatRequest(model="test-model", messages=[])
    with pytest.raises(ValueError, match="mode must be"):
        LLMClient("https://example.com", "test-key").structured(
            request, Person, mode="unsupported"
        )


@pytest.mark.parametrize(
    ("content", "error_type"),
    [
        ("not JSON", json.JSONDecodeError),
        ('{"name":"Alice","age":"invalid","job":"Engineer"}', ValidationError),
    ],
)
def test_structured_does_not_retry_invalid_output(monkeypatch, content, error_type):
    calls = []

    def fake_post(*args, **kwargs):
        calls.append(1)
        return httpx.Response(
            200,
            json={
                "choices": [{"message": {"content": content}}],
            },
        )

    monkeypatch.setattr("mini_llm.client.httpx.post", fake_post)
    request = ChatRequest(model="test-model", messages=[])
    with pytest.raises(error_type):
        LLMClient("https://example.com", "test-key").structured(request, Person)
    assert len(calls) == 1
