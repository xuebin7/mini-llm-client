import pytest

from mini_llm import (
    LLMAPIError,
    LLMRetryExhaustedError,
)
from mini_llm.retry import retry


def test_retry_until_success(monkeypatch):
    attempts = 0
    delays = []

    def fake_sleep(seconds):
        delays.append(seconds)

    monkeypatch.setattr("mini_llm.retry.time.sleep", fake_sleep)

    def fake_request():
        nonlocal attempts

        attempts += 1

        if attempts < 3:
            raise LLMAPIError(503, "Service Unavailable")

        return "success"

    result = retry(fake_request, max_attempts=3)

    assert result == "success"
    assert attempts == 3
    assert len(delays) == 2
    assert 1.0 <= delays[0] <= 1.5
    assert 2.0 <= delays[1] <= 2.5


def test_do_not_retry_401(monkeypatch):
    attempts = 0
    delays = []

    def fake_sleep(seconds):
        delays.append(seconds)

    monkeypatch.setattr(
        "mini_llm.retry.time.sleep",
        fake_sleep,
    )

    def fake_request():
        nonlocal attempts
        attempts += 1

        raise LLMAPIError(
            401,
            "Unauthorized",
        )

    with pytest.raises(LLMAPIError):
        retry(fake_request, max_attempts=3)

    # 注意最后：这是整个测试的重点，401 是不会重试的，所以这里返回的次数是1
    assert attempts == 1
    assert len(delays) == 0


def test_retry_429(monkeypatch):
    attempts = 0
    delays = []

    def fake_sleep(seconds):
        delays.append(seconds)

    monkeypatch.setattr(
        "mini_llm.retry.time.sleep",
        fake_sleep,
    )

    def fake_request():
        nonlocal attempts
        attempts += 1

        if attempts < 2:
            raise LLMAPIError(
                429,
                "Too Many Requests",
            )

        return "success"

    result = retry(
        fake_request,
        max_attempts=3,
    )

    assert result == "success"
    assert attempts == 2
    assert len(delays) == 1


def test_retry_exhausted(monkeypatch):
    attempts = 0
    delays = []

    def fake_sleep(seconds):
        delays.append(seconds)

    monkeypatch.setattr(
        "mini_llm.retry.time.sleep",
        fake_sleep,
    )

    def fake_request():
        nonlocal attempts
        attempts += 1

        raise LLMAPIError(503, "Service Unavailable")

    with pytest.raises(LLMRetryExhaustedError) as exc_info:
        retry(
            fake_request,
            max_attempts=3,
        )

    assert attempts == 3
    assert len(delays) == 2

    assert exc_info.value.attempts == 3

    assert isinstance(
        exc_info.value.last_error,
        LLMAPIError,
    )

    assert exc_info.value.last_error.status_code == 503


def test_invalid_max_attempts():
    def fake_request():
        return "success"

    with pytest.raises(ValueError) as exc_info:
        retry(
            fake_request,
            max_attempts=0,
        )

    assert str(exc_info.value) == "max_attempts must be at least 1"
