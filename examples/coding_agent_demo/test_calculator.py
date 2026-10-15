from calculator import divide


def test_divide():
    assert divide(10, 2) == 5


def test_divide_fraction():
    assert divide(9, 3) == 3
