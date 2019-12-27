import pytest
from plymi_mod6.basic_functions import count_vowels, merge_max_mappings


@pytest.mark.parametrize(
    ("input_string", "include_y", "expected"),
    [("aA bB yY", False, 2), ("aA bB yY", True, 4), ("", False, 0), ("", True, 0)],
)
def test_count_vowels_basic(input_string: str, include_y: bool, expected: int):
    assert count_vowels(input_string, include_y) == expected


@pytest.mark.parametrize(
    ("dict1", "dict2", "expected"),
    [
        (dict(a=1, b=2), dict(b=20, c=-1), dict(a=1, b=20, c=-1)),
        (dict(), dict(b=20, c=-1), dict(b=20, c=-1)),
        (dict(a=1, b=2), dict(), dict(a=1, b=2)),
        (dict(), dict(), dict()),
    ],
)
def test_merge_max_mappings(dict1: dict, dict2: dict, expected: dict):
    assert merge_max_mappings(dict1, dict2) == expected
