import json
from collections.abc import Iterable, Iterator

class StreamParser:

    def parse(self, lines: Iterable[str]) -> Iterator[str]:
        for line in lines:
            line = line.strip()

            if not line:
                continue

            if not line.startswith("data:"):
                continue

            data = line[len("data:"):].strip()

            if data == "[DONE]":
                break

            try:
                event = json.loads(data)
            except json.JSONDecodeError:
                continue

            choices = event.get("choices", [])

            if not choices:
                continue

            delta = choices[0].get("delta", {})

            content = delta.get("content")

            if content:
                yield content