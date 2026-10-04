from pydantic import BaseModel


class StreamRequest(BaseModel):
    thread_id: str
    user_input: str


class StreamRegistry:
    def __init__(self) -> None:
        self.streams: dict[str, StreamRequest] = {}

    def register(self, stream_id: str, thread_id: str, user_input: str) -> None:
        self.streams[stream_id] = StreamRequest(
            thread_id=thread_id,
            user_input=user_input,
        )

    def get(
        self,
        stream_id: str,
    ) -> StreamRequest | None:
        return self.streams.get(stream_id)

    def remove(
        self,
        stream_id: str,
    ) -> StreamRequest | None:
        return self.streams.pop(stream_id, None)
