from string_utils import is_palindrome


def test_palindrome():
    assert is_palindrome("level") is True


def test_not_palindrome():
    assert is_palindrome("hello") is False
