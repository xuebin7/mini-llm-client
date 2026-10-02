import pytest

from mini_llm.harness.app_server import AppServer
from mini_llm.harness.runtime import MiniHarness
from tests.fakes import FakeAgentLoop


def test_app_server_can_create_thread():
    harness = MiniHarness(agent_loop=FakeAgentLoop())
    server = AppServer(harness=harness)

    thread_id = server.create_thread()

    assert thread_id.startswith("thread-")
    assert harness.get_thread(thread_id) is not None


@pytest.mark.asyncio
async def test_app_server_can_run_turn():
    harness = MiniHarness(agent_loop=FakeAgentLoop())
    server = AppServer(harness=harness)

    thread_id = server.create_thread()

    result = await server.run_turn(
        thread_id=thread_id,
        user_input="hello",
    )

    assert result.output == "ok"
    assert result.turn.thread_id == thread_id

    thread = harness.get_thread(thread_id=thread_id)

    assert thread is not None
    assert len(thread.turns) == 1


@pytest.mark.asyncio
async def test_app_server_can_stream_turn():
    harness = MiniHarness(agent_loop=FakeAgentLoop())
    server = AppServer(harness=harness)

    thread_id = server.create_thread()

    chunks: list[str] = []

    async for chunk in server.stream_turn(
        thread_id=thread_id,
        user_input="hello",
    ):
        chunks.append(chunk)

    assert len(chunks) == 2

    assert chunks[0].startswith("event: turn_started\n")

    assert chunks[-1].startswith("event: turn_completed\n")
