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


@pytest.mark.asyncio
async def test_rpc_turn_stream_registers_stream():
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

    assert response.result is not None

    stream_id = response.result["stream_id"]
    assert isinstance(stream_id, str)

    stream = server.stream_registry.get(stream_id)

    assert stream is not None
    assert stream.thread_id == thread_id
    assert stream.user_input == "hello"


@pytest.mark.asyncio
async def test_rpc_stream_handshake_then_open_sse_stream():
    harness = MiniHarness(agent_loop=FakeAgentLoop())
    server = AppServer(harness=harness)
    dispatcher = RPCDispatcher(server=server)

    # 1. 创建 Thread
    thread_id = server.create_thread()

    # 2. 客户端通过 RPC 请求一个流式 Turn
    request = RPCRequest(
        id="req-stream-1",
        method="turn.stream",
        params={
            "thread_id": thread_id,
            "user_input": "hello",
        },
    )

    response = await dispatcher.handle(request)

    assert response.error is None
    assert response.result is not None

    stream_id = response.result["stream_id"]

    assert isinstance(stream_id, str)

    # 3. 客户端拿 stream_id 打开 SSE 流
    chunks: list[str] = []

    async for chunk in server.open_stream(stream_id):
        chunks.append(chunk)

    # 4. 验证收到了完整生命周期
    assert chunks[0].startswith("event: turn_started\n")

    assert chunks[-1].startswith("event: turn_completed\n")

    # 5. 流消费结束后自动清理
    assert server.stream_registry.get(stream_id) is None
