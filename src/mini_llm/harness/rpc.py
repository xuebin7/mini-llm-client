from typing import Any

from pydantic import BaseModel, Field

from mini_llm.harness.app_server import AppServer


class RPCRequest(BaseModel):
    id: str
    method: str
    params: dict[str, Any] = Field(default_factory=dict)


class RPCResponse(BaseModel):
    id: str
    result: dict[str, Any] | None = None
    error: str | None = None


class RPCDispatcher:
    def __init__(self, server: AppServer) -> None:
        self.server = server

    async def handle(
        self,
        request: RPCRequest,
    ) -> RPCResponse:
        if request.method == "thread.create":
            thread_id = self.server.create_thread()

            return RPCResponse(
                id=request.id,
                result={
                    "thread_id": thread_id,
                },
            )

        if request.method == "turn.run":
            thread_id = request.params.get("thread_id")
            user_input = request.params.get("user_input")

            if not isinstance(thread_id, str):
                return RPCResponse(
                    id=request.id,
                    error="Invalid params: thread_id must be a string",
                )

            if not isinstance(user_input, str):
                return RPCResponse(
                    id=request.id,
                    error="Invalid params: user_input must be a string",
                )

            result = await self.server.run_turn(
                thread_id=thread_id,
                user_input=user_input,
            )

            return RPCResponse(
                id=request.id,
                result={
                    "thread_id": thread_id,
                    "turn_id": result.turn.id,
                    "output": result.output,
                    "status": result.turn.status,
                },
            )

        if request.method == "turn.stream":
            thread_id = request.params.get("thread_id")
            user_input = request.params.get("user_input")

            if not isinstance(thread_id, str):
                return RPCResponse(
                    id=request.id,
                    error="Invalid params: thread_id must be a string",
                )

            if not isinstance(user_input, str):
                return RPCResponse(
                    id=request.id,
                    error="Invalid params: user_input must be a string",
                )

            stream_id = f"stream-{request.id}"

            self.server.create_stream(
                stream_id=stream_id,
                thread_id=thread_id,
                user_input=user_input,
            )

            return RPCResponse(
                id=request.id,
                result={
                    "stream_id": stream_id,
                    "thread_id": thread_id,
                    "user_input": user_input,
                },
            )

        return RPCResponse(
            id=request.id,
            error=f"Unknown method: {request.method}",
        )
