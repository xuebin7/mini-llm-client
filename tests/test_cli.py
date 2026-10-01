import pytest

from mini_llm.harness.cli import parse_command, run_cli
from mini_llm.harness.runtime import MiniHarness
from tests.fakes import FakeAgentLoop


def test_parse_send_command():
    name, argument = parse_command("send hello")

    assert name == "send"
    assert argument == "hello"


def test_parse_new_command():
    name, argument = parse_command("new")

    assert name == "new"
    assert argument is None


@pytest.mark.asyncio
async def test_run_cli_exit(monkeypatch):
    harness = MiniHarness(agent_loop=FakeAgentLoop())

    inputs = iter(["exit"])

    monkeypatch.setattr(
        "builtins.input",
        lambda _: next(inputs),
    )

    await run_cli(harness)

    assert harness.list_threads() == []


@pytest.mark.asyncio
async def test_run_cli_new_creates_thread(
    monkeypatch,
    capsys,
):
    harness = MiniHarness(agent_loop=FakeAgentLoop())

    inputs = iter(
        [
            "new",
            "exit",
        ]
    )

    monkeypatch.setattr(
        "builtins.input",
        lambda _: next(inputs),
    )

    await run_cli(harness)

    threads = harness.list_threads()

    assert len(threads) == 1

    captured = capsys.readouterr()

    assert "created thread:" in captured.out
    assert threads[0].id in captured.out


@pytest.mark.asyncio
async def test_run_cli_send_message(
    monkeypatch,
    capsys,
):
    agent_loop = FakeAgentLoop()
    harness = MiniHarness(agent_loop=agent_loop)

    inputs = iter(
        [
            "new",
            "send hello",
            "exit",
        ]
    )

    monkeypatch.setattr(
        "builtins.input",
        lambda _: next(inputs),
    )

    await run_cli(harness)

    captured = capsys.readouterr()

    assert "created thread:" in captured.out
    assert "assistant: ok" in captured.out

    threads = harness.list_threads()

    assert len(threads) == 1
    assert len(threads[0].turns) == 1

    assert len(agent_loop.calls) == 1
    assert agent_loop.calls[0]["user_input"] == "hello"


@pytest.mark.asyncio
async def test_run_cli_list_threads(
    monkeypatch,
    capsys,
):
    harness = MiniHarness(agent_loop=FakeAgentLoop())

    inputs = iter(
        [
            "new",
            "threads",
            "exit",
        ]
    )

    monkeypatch.setattr(
        "builtins.input",
        lambda _: next(inputs),
    )

    await run_cli(harness)

    threads = harness.list_threads()
    captured = capsys.readouterr()

    assert len(threads) == 1
    assert threads[0].id in captured.out


@pytest.mark.asyncio
async def test_run_cli_list_turns(
    monkeypatch,
    capsys,
):
    agent_loop = FakeAgentLoop()
    harness = MiniHarness(agent_loop=agent_loop)

    inputs = iter(
        [
            "new",
            "send hello",
            "turns",
            "exit",
        ]
    )

    monkeypatch.setattr(
        "builtins.input",
        lambda _: next(inputs),
    )

    await run_cli(harness)

    threads = harness.list_threads()
    thread = threads[0]
    turn = thread.turns[0]

    captured = capsys.readouterr()

    assert len(thread.turns) == 1
    assert turn.id in captured.out
    assert str(turn.status) in captured.out


@pytest.mark.asyncio
async def test_run_cli_send_without_active_thread(
    monkeypatch,
    capsys,
):
    harness = MiniHarness(agent_loop=FakeAgentLoop())

    inputs = iter(
        [
            "send hello",
            "exit",
        ]
    )

    monkeypatch.setattr(
        "builtins.input",
        lambda _: next(inputs),
    )

    await run_cli(harness)

    captured = capsys.readouterr()

    assert "no active thread" in captured.out


@pytest.mark.asyncio
async def test_run_cli_returns_without_active_thread(
    monkeypatch,
    capsys,
):
    harness = MiniHarness(agent_loop=FakeAgentLoop())

    inputs = iter(
        [
            "turns",
            "exit",
        ]
    )

    monkeypatch.setattr(
        "builtins.input",
        lambda _: next(inputs),
    )

    await run_cli(harness)

    captured = capsys.readouterr()

    assert "no active thread" in captured.out
