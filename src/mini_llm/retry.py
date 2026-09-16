import random
import time
from collections.abc import Callable
from typing import TypeVar

from mini_llm.exceptions import LLMAPIError, LLMTimeoutError

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

def retry(
        func: Callable[[], T],
        max_attempts: int = 3,
        base_delay: float = 1.0,
) -> T:
    last_error: Exception | None = None

    for attempt in range(max_attempts):
        try:
            return func()
        except Exception as exc:
            last_error = exc

            if not is_retryable(exc):
                raise

            if attempt < max_attempts - 1:
                jitter = random.uniform(0, 0.5)
                delay = (base_delay * (2 ** attempt) + jitter)
                print(f"retry after {delay:.2f}s")
                time.sleep(delay)

    raise last_error
