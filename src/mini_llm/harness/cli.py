from mini_llm.harness.runtime import MiniHarness


def parse_command(command: str) -> tuple[str, str | None]:
    command = command.strip()

    if command.startswith("send "):
        return "send", command.removeprefix("send ").strip()

    return command, None


async def run_cli(harness: MiniHarness) -> None:
    current_thread_id: str | None = None

    while True:
        raw_command = input("> ")
        command, argument = parse_command(raw_command)

        if command == "exit":
            break

        if command == "new":
            thread = harness.create_thread()
            current_thread_id = thread.id

            print(f"created thread: {thread.id}")
            continue

        if command == "threads":
            threads = harness.list_threads()

            for thread in threads:
                print(thread.id)

            continue

        if command == "turns":
            if current_thread_id is None:
                print("no active thread")
                continue

            thread = harness.get_thread(current_thread_id)

            if thread is None:
                print("thread not found")
                continue

            for turn in harness.list_turns(thread):
                print(f"{turn.id} {turn.status}")

            continue

        if command == "send":
            if current_thread_id is None:
                print("no active thread")
                continue

            if argument is None or not argument:
                print("message is empty")
                continue

            result = await harness.run_turn_by_id(
                thread_id=current_thread_id,
                user_input=argument,
            )

            print(f"assistant: {result.output}")
            continue

        print(f"unknown command: {command}")
