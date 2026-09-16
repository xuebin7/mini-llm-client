import httpx
import pytest

from mini_llm.client import LLMClient
from mini_llm.exceptions import LLMAPIError
from mini_llm.models import ChatRequest, Message

def test_chat_retry_503_then_success(monkeypatch):
    attempts = 0
    delays = []

    def fake_post(*args, **kwargs):
        nonlocal attempts
        attempts += 1

        print(f"http attempt: {attempts}")

        if attempts < 3:
            return httpx.Response(
                status_code=503,
                text="Service Unavailable",
            )

        return httpx.Response(
            status_code=200,
            json={
                "model": "deepseek-flash",
                "choices": [
                    {
                        "message": {
                            "content": "hello"
                        }
                    }
                ],
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
        fake_sleep
    )

    client = LLMClient(
        api_key="test-key",
        base_url="https://example.com",
        timeout=30.0,
    )

    request = ChatRequest(
        model="deepseek-flash",
        messages=[
            Message(
                role="user",
                content="hello"
            )
        ]
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

        print(f"http attempt: {attempts}")

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

        print(f"http attempt: {attempts}")

        if attempts == 1:
            raise httpx.TimeoutException(
                "Request timed out"
            )
        
        return httpx.Response(
            status_code=200,
            json={
                "model": "deepseek-flash",
                "choices": [
                    {
                        "message": {
                            "content": "hello"
                        }
                    }
                ],
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
    