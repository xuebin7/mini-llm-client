import httpx
import pytest

from mini_llm import ChatRequest, LLMAPIError, LLMClient, Message
from mini_llm.models import ResponseRequest, ResponseResult


@pytest.mark.parametrize(
    "options",
    [
        {},
        {"temperature": 0, "response_format": {"type": "json_object"}},
    ],
)
def test_chat_serializes_optional_parameters(monkeypatch, options):
    captured = []

    def fake_post(url, **kwargs):
        captured.append((url, kwargs))
        return httpx.Response(
            200,
            json={
                "choices": [{"message": {"content": "hello"}}],
            },
        )

    monkeypatch.setattr("mini_llm.client.httpx.post", fake_post)
    request = ChatRequest(
        model="test-model",
        messages=[Message(role="user", content="Hello")],
        **options,
    )
    client = LLMClient("https://example.com/", "test-key", timeout=5)
    assert client.chat(request).content == "hello"
    url, kwargs = captured[0]
    assert url == "https://example.com/chat/completions"
    assert kwargs["json"] == {
        "model": "test-model",
        "messages": [{"role": "user", "content": "Hello"}],
        **options,
    }
    assert kwargs["timeout"] == 5
    assert kwargs["headers"]["Authorization"] == "Bearer test-key"


def test_chat_retry_503_then_success(monkeypatch):
    attempts = 0
    delays = []

    def fake_post(*args, **kwargs):
        nonlocal attempts
        attempts += 1

        if attempts < 3:
            return httpx.Response(
                status_code=503,
                text="Service Unavailable",
            )

        return httpx.Response(
            status_code=200,
            json={
                "model": "deepseek-flash",
                "choices": [{"message": {"content": "hello"}}],
            },
        )

    def fake_sleep(seconds):
        delays.append(seconds)

    monkeypatch.setattr(
        "mini_llm.client.httpx.post",
        fake_post,
    )

    monkeypatch.setattr("mini_llm.retry.time.sleep", fake_sleep)

    client = LLMClient(
        api_key="test-key",
        base_url="https://example.com",
        timeout=30.0,
    )

    request = ChatRequest(
        model="deepseek-flash", messages=[Message(role="user", content="hello")]
    )

    response = client.chat(request)

    assert response.content == "hello"
    assert attempts == 3
    assert len(delays) == 2


def test_chat_does_not_retry_401(monkeypatch):
    attempts = 0
    delays = []

    def fake_post(*args, **kwargs):
        nonlocal attempts
        attempts += 1

        return httpx.Response(
            status_code=401,
            text="Unauthorized",
        )

    def fake_sleep(seconds):
        delays.append(seconds)

    monkeypatch.setattr(
        "mini_llm.client.httpx.post",
        fake_post,
    )

    monkeypatch.setattr(
        "mini_llm.retry.time.sleep",
        fake_sleep,
    )

    client = LLMClient(
        api_key="test-key",
        base_url="https://example.com",
    )

    request = ChatRequest(
        model="deepseek-flash",
        messages=[
            Message(
                role="user",
                content="hello",
            )
        ],
    )

    with pytest.raises(LLMAPIError) as exc_info:
        client.chat(request)

    assert exc_info.value.status_code == 401
    assert attempts == 1
    assert len(delays) == 0


def test_chat_retry_timeout_then_success(monkeypatch):
    attempts = 0
    delays = []

    def fake_post(*args, **kwargs):
        nonlocal attempts
        attempts += 1

        if attempts == 1:
            raise httpx.TimeoutException("Request timed out")

        return httpx.Response(
            status_code=200,
            json={
                "model": "deepseek-flash",
                "choices": [{"message": {"content": "hello"}}],
            },
        )

    def fake_sleep(seconds):
        delays.append(seconds)

    monkeypatch.setattr(
        "mini_llm.client.httpx.post",
        fake_post,
    )

    monkeypatch.setattr(
        "mini_llm.retry.time.sleep",
        fake_sleep,
    )

    client = LLMClient(
        api_key="test-key",
        base_url="https://example.com",
    )

    request = ChatRequest(
        model="deepseek-flash",
        messages=[
            Message(
                role="user",
                content="hello",
            )
        ],
    )

    response = client.chat(request)

    assert response.content == "hello"
    assert attempts == 2
    assert len(delays) == 1


def test_responses_parses_text_response(monkeypatch):
    captured = []

    def fake_post(url, **kwargs):
        captured.append((url, kwargs))

        return httpx.Response(
            200,
            json={
                "id": "resp_123",
                "model": "test-model",
                "output": [
                    {
                        "type": "message",
                        "content": [{"type": "output_text", "text": "hello"}],
                    }
                ],
            },
        )

    monkeypatch.setattr("mini_llm.client.httpx.post", fake_post)

    request = ResponseRequest(
        input="Hello",
        model="test-model",
    )

    client = LLMClient("https://example.com/", "test-key", timeout=5)
    result = client.responses(request)

    assert result.text == "hello"
    assert result.model == "test-model"
    assert result.response_id == "resp_123"
    assert result.tool_calls == []

    url, kwargs = captured[0]

    assert url == "https://example.com/responses"

    assert kwargs["json"] == {
        "input": "Hello",
        "model": "test-model",
    }

    assert kwargs["timeout"] == 5
    assert kwargs["headers"]["Authorization"] == "Bearer test-key"


def test_responses_parses_tool_call(monkeypatch):
    captured = []

    def fake_post(url, **kwargs):
        captured.append((url, kwargs))

        return httpx.Response(
            status_code=200,
            json={
                "id": "resp_123",
                "model": "test-model",
                "output": [
                    {
                        "type": "function_call",
                        "call_id": "call_123",
                        "name": "git_log",
                        "arguments": '{"path": ".", "max_count": 3}',
                    }
                ],
            },
        )

    monkeypatch.setattr(
        "mini_llm.client.httpx.post",
        fake_post,
    )

    request = ResponseRequest(
        model="test-model",
        input="Show me the latest 3 git commits.",
        tools=[
            {
                "type": "function",
                "name": "git_log",
                "description": "Read recent Git commit history",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "path": {
                            "type": "string",
                        },
                        "max_count": {
                            "type": "integer",
                        },
                    },
                    "additionalProperties": False,
                },
            }
        ],
    )

    client = LLMClient(base_url="https://example.com/", api_key="test-key", timeout=5)

    result = client.responses(request)

    assert result.text is None
    assert len(result.tool_calls) == 1

    tool_call = result.tool_calls[0]

    assert tool_call.call_id == "call_123"
    assert tool_call.name == "git_log"
    assert tool_call.arguments == '{"path": ".", "max_count": 3}'

    assert result.response_id == "resp_123"
    assert result.model == "test-model"

    url, kwargs = captured[0]

    assert url == "https://example.com/responses"
    assert kwargs["json"]["tools"][0]["name"] == "git_log"
