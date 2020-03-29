import pytest


@pytest.mark.parametrize("size", [0, 1, 2, 3])
def test_range_length(size):
    assert len(range(size)) == size


@pytest.mark.parametrize("a, b, c", [(1, 2, 3), (4, 5, 6), (7, 8, 9), (10, 11, 12)])
def test_inequality(a, b, c):
    assert a < b < c


@pytest.mark.parametrize("x", [0, 1, 2])
@pytest.mark.parametrize("y", [10, 20])
def test_all_combinations(x, y):
    # will run:
    # x=0 y=10
    # x=0 y=20
    # x=1 y=10
    # x=1 y=20
    # x=2 y=10
    # x=2 y=20
    pass
