import random
import time
from collections.abc import Callable
from typing import TypeVar

from .exceptions import (
    LLMAPIError,
    LLMRetryExhaustedError,
    LLMTimeoutError,
)

T = TypeVar("T")

RETRYABLE_STATUS_CODES = {
    408,
    429,
    500,
    502,
    503,
    504,
}


def is_retryable(exc: Exception) -> bool:
    if isinstance(exc, LLMTimeoutError):
        return True

    if isinstance(exc, LLMAPIError):
        return exc.status_code in RETRYABLE_STATUS_CODES

    return False


def retry[T](
    func: Callable[[], T],
    max_attempts: int = 3,
    base_delay: float = 1.0,
    jitter: float = 0.5,
) -> T:
    if max_attempts < 1:
        raise ValueError("max_attempts must be at least 1")
    last_error: Exception | None = None

    for attempt in range(max_attempts):
        try:
            return func()
        except Exception as exc:
            last_error = exc

            if not is_retryable(exc):
                raise

            if attempt < max_attempts - 1:
                delay = base_delay * (2**attempt) + random.uniform(0, jitter)
                time.sleep(delay)

    assert last_error is not None

    raise LLMRetryExhaustedError(
        attempts=max_attempts,
        last_error=last_error,
    ) from last_error
