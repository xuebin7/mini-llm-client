from mini_llm.harness.models import Thread, Turn


def test_thread_can_gen_id():
    thread = Thread()

    assert thread.id is not None and thread.id.startswith("thread-")


def test_two_thread_id_different():
    thread1 = Thread()
    thread2 = Thread()

    assert thread1.id != thread2.id


def test_turn_can_gen_id():
    thread = Thread()

    turn = Turn(
        thread_id=thread.id,
    )

    assert turn.id is not None and turn.id.startswith("turn-")
    assert turn.thread_id == thread.id


def test_two_turn_has_same_thread_id_and_diff_id():
    thread = Thread()

    turn1 = Turn(thread_id=thread.id)

    turn2 = Turn(thread_id=thread.id)

    assert turn1.thread_id == thread.id
    assert turn2.thread_id == thread.id
    assert turn1.id != turn2.id
