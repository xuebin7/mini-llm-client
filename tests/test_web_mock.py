import pytest

from mini_llm.harness.runtime import MiniHarness
from mini_llm.harness.web_mock import MockWebClient
from tests.fakes import FakeAgentLoop


def test_mock_web_client_can_create_thread():
    harness = MiniHarness(agent_loop=FakeAgentLoop())
    client = MockWebClient(harness=harness)

    thread_id = client.create_thread()

    assert thread_id.startswith("thread-")
    assert harness.get_thread(thread_id) is not None


@pytest.mark.asyncio
async def test_mock_web_client_can_send_message():
    harness = MiniHarness(agent_loop=FakeAgentLoop())
    client = MockWebClient(harness=harness)

    thread_id = client.create_thread()

    response = await client.send_message(
        thread_id=thread_id,
        message="hello",
    )

    assert response["thread_id"] == thread_id
    assert response["output"] == "ok"
    assert response["turn_id"].startswith("turn-")
    assert response["events"]
