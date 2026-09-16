import httpx

from mini_llm.streaming import StreamParser

from .models import ChatRequest, ChatResponse
from .exceptions import (
    LLMAPIError,
    LLMTimeoutError,
)
from typing import TypeVar
from pydantic import BaseModel
from .structured import parse_structured_output
from .retry import retry

T = TypeVar("T", bound=BaseModel)

class LLMClient:

    def __init__(
            self,
            base_url: str,
            api_key: str,
            timeout: float = 30.0,
    ):
        self.base_url = base_url.rstrip("/")
        self.api_key = api_key
        self.timeout = timeout

    def _do_request(
        self,
        url: str,
        headers: dict,
        payload: dict,
    ) -> httpx.Response:
        try:
            response = httpx.post(
                url,
                headers=headers,
                json=payload,
                timeout=self.timeout,
            )
        except httpx.TimeoutException as exc:
            raise LLMTimeoutError(
                f"LLM request timed out after "
                f"{self.timeout}s"
            ) from exc

        if response.status_code >= 400:
            raise LLMAPIError(
                status_code=response.status_code,
                message=(
                    f"LLM API error: "
                    f"{response.status_code} "
                    f"{response.text}"
                ),
            )

        return response

    def chat(self, request: ChatRequest) -> ChatResponse:
        
        url = f"{self.base_url}/chat/completions"
        
        headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json",
        }
        
        payload = {
            "model": request.model,
            "messages": [
                message.model_dump()
                for message in request.messages
            ],
        }
        
        if request.temperature is not None:
            payload["temperature"] = request.temperature
            
        if request.response_format is not None:
            payload["response_format"] = request.response_format

        print("========== REQUEST PAYLOAD ==========")
        print(payload)
        print("=====================================")

        response = retry(
                        lambda: self._do_request(
                            url,
                            headers,
                            payload,
                        ),
                        max_attempts=3
                    )
        
        data = response.json()
        
        content = data["choices"][0]["message"]["content"]

        return ChatResponse(
            content=content,
            model=data.get("model"),
        )

    def stream(self, request: ChatRequest):
        url = f"{self.base_url}/chat/completions"

        headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json",
        }

        payload = {
            "model": request.model,
            "messages": [
                message.model_dump()
                for message in request.messages
            ],
            "stream": True
        }

        with httpx.stream(
            "POST",
            url,
            headers=headers,
            json=payload,
            timeout=self.timeout,
        ) as response:

            if response.status_code >= 400:
                raise LLMAPIError(
                                status_code=response.status_code,
                                message=(f"LLM API error: "
                                              f"{response.status_code}"
                                              f"{response.text}"
                                        ),
                            )

            parser = StreamParser()

            for text in parser.parse(response.iter_lines()):
                yield text

    def schema_to_response_format(
            schema: type[BaseModel],
    ) -> dict:
        return {
            "type": "json_schema",
            "json_schema": {
                "name": schema.__name__,
                "schema": schema.model_json_schema(),
                "strict": True
            }
        },

    def structured(
            self,
            request: ChatRequest,
            schema: type[T],
    ) -> T:
        request_data = request.model_copy(
            update={
                "response_format": {
                    "type": "json_object",
                    "json_schema": {
                        "name": schema.__name__,
                        "schema": schema.model_json_schema(),
                        "strict": True
                    }
                }
            }
        )

        print("STRUCTURED REQUEST:")
        print(request_data.model_dump())
        
        response = self.chat(request_data)

        return parse_structured_output(
            response.content, 
            schema
        )

    