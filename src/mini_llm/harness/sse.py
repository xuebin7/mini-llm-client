import json

from mini_llm.harness.events import Event


def encode_sse(event: Event) -> str:
    data = event.model_dump(
        mode="json",
        exclude={"type"},
    )

    payload = json.dumps(
        data,
        ensure_ascii=False,
    )

    return f"event: {event.type}\ndata: {payload}\n\n"
