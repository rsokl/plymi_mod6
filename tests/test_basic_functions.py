from random import shuffle
from typing import Dict
from string import printable

import hypothesis.strategies as st
import pytest
from hypothesis import given, note

from plymi_mod6.basic_functions import count_vowels, merge_max_mappings


##################################
# Basic implementations of tests #
##################################


def test_count_vowels_basic():
    # test basic strings with uppercase and lowercase letters
    assert count_vowels("aA bB yY", include_y=False) == 2
    assert count_vowels("aA bB yY", include_y=True) == 4

    # test empty strings
    assert count_vowels("", include_y=False) == 0
    assert count_vowels("", include_y=True) == 0


def test_merge_max_mappings():
    # test documented behavior
    dict1 = {"a": 1, "b": 2}
    dict2 = {"b": 20, "c": -1}
    expected = {"a": 1, "b": 20, "c": -1}
    assert merge_max_mappings(dict1, dict2) == expected

    # test empty dict1
    dict1 = {}
    dict2 = {"a": 10.2, "f": -1.0}
    expected = dict2
    assert merge_max_mappings(dict1, dict2) == expected

    # test empty dict2
    dict1 = {"a": 10.2, "f": -1.0}
    dict2 = {}
    expected = dict1
    assert merge_max_mappings(dict1, dict2) == expected

    # test both empty
    dict1 = {}
    dict2 = {}
    expected = {}
    assert merge_max_mappings(dict1, dict2) == expected


###########################################
# Using pytest's parameterization feature #
###########################################


@pytest.mark.parametrize(
    "input_string, include_y, expected_count",
    [("aA bB yY", False, 2), ("aA bB yY", True, 4), ("", False, 0), ("", True, 0)],
)
def test_count_vowels_parameterized(
    input_string: str, include_y: bool, expected_count: int
):
    assert count_vowels(input_string, include_y) == expected_count


@pytest.mark.parametrize(
    "dict_a, dict_b, expected_merged",
    [
        (dict(a=1, b=2), dict(b=20, c=-1), dict(a=1, b=20, c=-1)),
        (dict(), dict(b=20, c=-1), dict(b=20, c=-1)),
        (dict(a=1, b=2), dict(), dict(a=1, b=2)),
        (dict(), dict(), dict()),
    ],
)
def test_merge_max_mappings_parameterized(
    dict_a: dict, dict_b: dict, expected_merged: dict
):
    assert merge_max_mappings(dict_a, dict_b) == expected_merged


####################
# Using Hypothesis #
####################


_not_vowels = "".join([l for l in printable if l.lower() not in set("aeiouy")])


@given(
    not_vowels=st.text(alphabet=_not_vowels),
    not_ys=st.text(alphabet="aeiouAEIOU"),
    ys=st.text(alphabet="yY"),
)
def test_count_vowels_hypothesis(not_vowels: str, not_ys: str, ys: str):
    # shuffle ys into string
    letters = list(not_vowels) + list(not_ys) + list(ys)
    shuffle(letters)
    in_string = "".join(letters)
    note(f"in_string: {in_string}")
    assert count_vowels(in_string, include_y=False) == len(not_ys)
    assert count_vowels(in_string, include_y=True) == len(not_ys) + len(ys)


@given(
    dict1=st.dictionaries(
        keys=st.integers(-10, 10) | st.text(), values=st.integers(-10, 10)
    ),
    dict2=st.dictionaries(
        keys=st.integers(-10, 10) | st.text(), values=st.integers(-10, 10)
    ),
)
def test_merge_max_mappings_hypothesis(dict1: Dict[int, int], dict2: Dict[int, int]):
    merged_dict = merge_max_mappings(dict1, dict2)
    assert set(merged_dict) == set(dict1).union(
        dict2
    ), "novel keys were introduced or lost"

    assert set(merged_dict.values()) <= set(dict1.values()).union(
        dict2.values()
    ), "novel values were introduced"

    assert all(
        dict1[k] <= merged_dict[k] for k in dict1
    ), "`merged_dict` contains a non-max value"

    assert all(
        dict2[k] <= merged_dict[k] for k in dict2
    ), "`merged_dict` contains a non-max value"

    for k, v in merged_dict.items():
        assert (k, v) in dict1.items() or (
            k,
            v,
        ) in dict2.items(), "`merged_dict` did not preserve the key-value pairings"
