from contextlib import contextmanager

import httpx
import pytest

from mini_llm import (
    ChatRequest,
    LLMAPIError,
    LLMClient,
    LLMTimeoutError,
    Message,
)


class ResponseStream(httpx.SyncByteStream):
    """An unread HTTPX body, optionally failing after its first chunk."""

    def __init__(self, chunks=(), error=None):
        self.chunks = chunks
        self.error = error
        self.closed = False

    def __iter__(self):
        yield from self.chunks
        if self.error is not None:
            raise self.error

    def close(self):
        self.closed = True


@pytest.fixture
def chat_request():
    return ChatRequest(
        model="test-model",
        messages=[Message(role="user", content="Hello")],
        temperature=0,
        response_format={"type": "json_object"},
    )


def install_stream(monkeypatch, response):
    calls = []

    @contextmanager
    def fake_stream(method, url, **kwargs):
        calls.append((method, url, kwargs))
        try:
            yield response
        finally:
            response.close()

    monkeypatch.setattr("mini_llm.client.httpx.stream", fake_stream)
    return calls


def test_stream_preserves_request_options_and_closes(monkeypatch, chat_request):
    body = ResponseStream(
        [
            b'data: {"choices":[{"delta":{"content":"Hello"}}]}\n\n',
            b"data: [DONE]\n\n",
        ]
    )
    calls = install_stream(monkeypatch, httpx.Response(200, stream=body))
    before = chat_request.model_dump()
    client = LLMClient("https://example.com/", "test-key", timeout=7)
    assert list(client.stream(chat_request)) == ["Hello"]
    method, url, kwargs = calls[0]
    assert method == "POST"
    assert url == "https://example.com/chat/completions"
    assert kwargs["json"] == {**before, "stream": True}
    assert kwargs["headers"]["Authorization"] == "Bearer test-key"
    assert kwargs["timeout"] == 7
    assert chat_request.model_dump() == before
    assert body.closed


@pytest.mark.parametrize("status", [401, 503])
def test_stream_reads_error_body_and_does_not_retry(monkeypatch, chat_request, status):
    body = ResponseStream([b"provider error details"])
    response = httpx.Response(status, stream=body)
    calls = install_stream(monkeypatch, response)
    with pytest.raises(LLMAPIError) as caught:
        list(LLMClient("https://example.com", "test-key").stream(chat_request))
    assert caught.value.status_code == status
    assert "provider error details" in str(caught.value)
    assert len(calls) == 1
    assert body.closed


def test_stream_translates_connection_timeout(monkeypatch, chat_request):
    error = httpx.ConnectTimeout("connect timed out")
    calls = []

    @contextmanager
    def fake_stream(*args, **kwargs):
        calls.append(1)
        raise error
        yield  # Make this a context manager without opening a connection.

    monkeypatch.setattr("mini_llm.client.httpx.stream", fake_stream)
    with pytest.raises(LLMTimeoutError) as caught:
        list(LLMClient("https://example.com", "test-key").stream(chat_request))
    assert caught.value.__cause__ is error
    assert len(calls) == 1


@pytest.mark.parametrize("status", [200, 503])
def test_stream_translates_body_timeout_and_closes(monkeypatch, chat_request, status):
    error = httpx.ReadTimeout("read timed out")
    chunks = [b'data: {"choices":[{"delta":{"content":"Hello"}}]}\n\n']
    body = ResponseStream(chunks if status == 200 else [], error=error)
    calls = install_stream(monkeypatch, httpx.Response(status, stream=body))
    iterator = LLMClient("https://example.com", "test-key").stream(chat_request)
    if status == 200:
        assert next(iterator) == "Hello"
    with pytest.raises(LLMTimeoutError) as caught:
        next(iterator)
    assert caught.value.__cause__ is error
    assert len(calls) == 1
    assert body.closed


def test_explicit_iterator_close_releases_response(monkeypatch, chat_request):
    body = ResponseStream(
        [
            b'data: {"choices":[{"delta":{"content":"Hello"}}]}\n\n',
            b'data: {"choices":[{"delta":{"content":" world"}}]}\n\n',
        ]
    )
    install_stream(monkeypatch, httpx.Response(200, stream=body))
    iterator = LLMClient("https://example.com", "test-key").stream(chat_request)
    assert next(iterator) == "Hello"
    iterator.close()
    assert body.closed
