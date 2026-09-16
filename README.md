# mini-llm-client

A minimal Python LLM client built for learning LLM API fundamentals.

## Features

- Chat completion
- Streaming output
- Structured output with Pydantic
- Request timeout handling
- Retry for transient failures
- Exponential backoff with jitter

## Installation

Requires Python 3.13+ and uv. From the project root, run:

```bash
uv sync
```

## Configuration

Copy `.env.example` to `.env` in the project root and set your API key:

```ini
LLM_API_KEY=your-api-key
LLM_BASE_URL=https://api.deepseek.com
LLM_MODEL=deepseek-flash
```

Use a model available from your provider; the model name above is an example.
The client uses a Chat Completions-compatible API and appends
`/chat/completions` to `LLM_BASE_URL`.

The client requires an explicit base URL and API key. It does not load `.env`
automatically; the commands below use uv to load it into the environment.
Keep real credentials in `.env`, which is ignored by Git. Commit only the
placeholder values in `.env.example`.

## Basic Chat

```python
import os

from mini_llm import ChatRequest, LLMClient, Message

client = LLMClient(
    base_url=os.environ["LLM_BASE_URL"],
    api_key=os.environ["LLM_API_KEY"],
    timeout=30.0,
)

request = ChatRequest(
    model=os.environ["LLM_MODEL"],
    messages=[Message(role="user", content="Hello")],
)

response = client.chat(request)
print(response.content)
```

Run the [basic example](examples/basic.py) from the project root:

```bash
uv run --env-file .env python -m examples.basic
```

## Streaming

`client.stream(request)` yields text chunks:

```python
for text in client.stream(request):
    print(text, end="", flush=True)
print()
```

Run the [streaming example](examples/stream.py):

```bash
uv run --env-file .env python -m examples.stream
```

For an offline demonstration of the SSE parser, see
[examples/stream_parser.py](examples/stream_parser.py).

## Structured Output

`client.structured(request, schema)` parses the response as JSON and validates
it against a Pydantic model, returning an instance of that model. By default,
it sends `{"type": "json_object"}` for providers with JSON mode support.
Your messages must explicitly request JSON and describe the expected fields.
This mode validates the schema locally; it does not enforce it on the server.

For providers and models supporting strict JSON Schema output, opt in explicitly:

```python
person = client.structured(request, Person, mode="json_schema")
```

Here `Person` is the Pydantic model defined in the example below. Schema mode
closes object schemas with `additionalProperties: false` and requires all fields,
including fields with defaults. Nullable fields may contain `null`, but cannot
be omitted. Open dictionaries, models with `extra="allow"`, and non-object root
models are rejected locally. Other schema constraints remain subject to the
provider's supported JSON Schema subset. There is no automatic mode fallback.

See [examples/structured.py](examples/structured.py):

```bash
uv run --env-file .env python -m examples.structured
```

Invalid JSON raises `json.JSONDecodeError`; data that fails schema validation
raises `pydantic.ValidationError`.

## Retry Policy

`chat()` and `structured()` retry transient request failures including:

- HTTP 408
- HTTP 429
- HTTP 500
- HTTP 502
- HTTP 503
- HTTP 504
- Request timeout

Non-retryable HTTP errors such as 400, 401, 403 and 404 fail immediately
with `LLMAPIError`.

Requests make at most three attempts, including the initial attempt. Retry
uses exponential backoff with jitter: 1–1.5 seconds before the second attempt
and 2–2.5 seconds before the third. Exhausted retries raise
`LLMRetryExhaustedError`, which exposes `attempts` and `last_error`.

Streaming requests currently do not retry automatically. JSON parsing and
Pydantic validation errors are not retried.

Streaming connection and read timeouts raise `LLMTimeoutError`, with the
original HTTPX exception retained as the cause. The timeout is an HTTPX
network-operation timeout, not a deadline for the entire generation.
Consume the stream completely or close it explicitly when stopping early:

```python
from contextlib import closing

with closing(client.stream(request)) as chunks:
    for text in chunks:
        print(text, end="", flush=True)
        # It is safe to break here: closing() releases the connection.
```

## Tests

```bash
uv run pytest
```

Tests use mocked requests and local sample data; no API key is required.
