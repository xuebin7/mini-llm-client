class LLMError(Exception):
    """LLM client base exception."""

class LLMTimeoutError(LLMError):
    """LLM request timed out."""

class LLMAPIError(LLMError):
    """LLM API returned an error"""

    def __init__(
            self,
            status_code: int,
            message: str,
    ):
        super().__init__(message)
        self.status_code = status_code

class LLMRetryExhaustedError(LLMError):
    """All retry attempts were exhausted."""