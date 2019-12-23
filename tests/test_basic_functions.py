from plymi_mod6.basic_functions import count_vowels, merge_max_mappings


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
    expected = {'a': 1, 'b': 20, 'c': -1}
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
