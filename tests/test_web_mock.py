import pytest

from mini_llm.harness.app_server import AppServer
from mini_llm.harness.runtime import MiniHarness
from mini_llm.harness.web_mock import MockWebClient
from tests.fakes import FakeAgentLoop


def test_mock_web_client_can_create_thread():
    harness = MiniHarness(agent_loop=FakeAgentLoop())
    server = AppServer(harness=harness)
    client = MockWebClient(server=server)

    thread_id = client.create_thread()

    assert thread_id.startswith("thread-")
    assert harness.get_thread(thread_id) is not None


@pytest.mark.asyncio
async def test_mock_web_client_can_send_message():
    harness = MiniHarness(agent_loop=FakeAgentLoop())
    server = AppServer(harness=harness)
    client = MockWebClient(server=server)

    thread_id = client.create_thread()

    response = await client.send_message(
        thread_id=thread_id,
        message="hello",
    )

    assert response["thread_id"] == thread_id
    assert response["output"] == "ok"
    assert response["turn_id"].startswith("turn-")
    assert response["events"]


@pytest.mark.asyncio
async def test_mock_web_client_can_stream_sse():
    harness = MiniHarness(agent_loop=FakeAgentLoop())
    server = AppServer(harness=harness)
    client = MockWebClient(server=server)

    thread_id = client.create_thread()

    chunks: list[str] = []

    async for chunk in client.stream_sse(
        thread_id=thread_id,
        message="hello",
    ):
        chunks.append(chunk)

    assert len(chunks) == 2

    assert chunks[0].startswith("event: turn_started\n")
    assert chunks[1].startswith("event: turn_completed\n")

    assert all(f'"thread_id": "{thread_id}' in chunk for chunk in chunks)
    assert all(chunk.endswith("\n\n") for chunk in chunks)
