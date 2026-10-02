import pytest

from mini_llm.harness.app_server import AppServer
from mini_llm.harness.rpc import RPCDispatcher, RPCRequest
from mini_llm.harness.runtime import MiniHarness
from tests.fakes import FakeAgentLoop


@pytest.mark.asyncio
async def test_rpc_can_create_thread():
    harness = MiniHarness(agent_loop=FakeAgentLoop())
    server = AppServer(harness=harness)
    dispatcher = RPCDispatcher(server=server)

    request = RPCRequest(
        id="req-1",
        method="thread.create",
    )

    response = await dispatcher.handle(request)

    assert response.id == "req-1"
    assert response.error is None
    assert response.result is not None

    thread_id = response.result["thread_id"]

    assert isinstance(thread_id, str)
    assert thread_id.startswith("thread-")


@pytest.mark.asyncio
async def test_rpc_unkonwn_method_returns_error():
    harness = MiniHarness(agent_loop=FakeAgentLoop())
    server = AppServer(harness=harness)
    dispatcher = RPCDispatcher(server=server)

    request = RPCRequest(
        id="req-2",
        method="unknown",
    )

    response = await dispatcher.handle(request)

    assert response.id == "req-2"
    assert response.result is None
    assert response.error == "Unknown method: unknown"


@pytest.mark.asyncio
async def test_rpc_can_run_turn():
    harness = MiniHarness(agent_loop=FakeAgentLoop())
    server = AppServer(harness=harness)
    dispatcher = RPCDispatcher(server=server)

    thread_id = server.create_thread()

    request = RPCRequest(
        id="req-3",
        method="turn.run",
        params={
            "thread_id": thread_id,
            "user_input": "hello",
        },
    )

    response = await dispatcher.handle(request)

    assert response.error is None
    assert response.result is not None

    assert response.result["thread_id"] == thread_id
    assert response.result["output"] == "ok"


@pytest.mark.asyncio
async def test_rpc_run_turn_rejects_invalid_params():
    harness = MiniHarness(agent_loop=FakeAgentLoop())
    server = AppServer(harness=harness)
    dispatcher = RPCDispatcher(server=server)

    request = RPCRequest(
        id="req-4",
        method="turn.run",
        params={
            "thread_id": 123,
            "user_input": "hello",
        },
    )

    response = await dispatcher.handle(request)

    assert response.result is None
    assert response.error == "Invalid params: thread_id must be a string"


@pytest.mark.asyncio
async def test_rpc_can_create_turn_stream():
    harness = MiniHarness(agent_loop=FakeAgentLoop())
    server = AppServer(harness=harness)
    dispatcher = RPCDispatcher(server=server)

    thread_id = server.create_thread()

    request = RPCRequest(
        id="req-5",
        method="turn.stream",
        params={
            "thread_id": thread_id,
            "user_input": "hello",
        },
    )

    response = await dispatcher.handle(request)

    assert response.error is None
    assert response.result is not None

    assert response.result["stream_id"] == "stream-req-5"
    assert response.result["thread_id"] == thread_id
