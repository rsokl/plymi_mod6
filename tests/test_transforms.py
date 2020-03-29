from typing import Callable

import hypothesis.extra.numpy as hnp
import hypothesis.strategies as st
import numpy as np
import pytest
from hypothesis import given
from numpy.testing import assert_allclose

from plymi_mod6.transforms import rotate, translate, shear, scale


@pytest.mark.parametrize(
    ("deg_rot, expected_transform"),
    [
        (0, lambda x: x),
        (90, lambda x: x[:, ::-1] * (-1, 1)),
        (180, lambda x: -x),
        (270, lambda x: x[:, ::-1] * (1, -1)),
        (360, lambda x: x),
    ],
)
@given(
    points=hnp.arrays(
        shape=st.integers(0, 10).map(lambda x: (x, 2)),
        dtype=np.float64,
        elements=st.floats(-1e10, 1e10),
    )
)
def test_known_rotations(
    points: np.ndarray,
    deg_rot: float,
    expected_transform: Callable[[np.ndarray], np.ndarray],
):
    actual = rotate(points, deg_rot)
    desired = expected_transform(points)
    assert_allclose(actual=actual, desired=desired, atol=1e-9, rtol=1e-9)


@given(
    x_shift=st.floats(-1e10, 1e10),
    y_shift=st.floats(-1e10, 1e10),
    num_points=st.integers(0, 10),
)
def test_translation_identity(x_shift: float, y_shift: float, num_points: int):
    points = np.zeros((num_points, 2), dtype=np.float64)
    points -= (x_shift, y_shift)
    points = translate(points, x_shift=x_shift, y_shift=y_shift)
    assert_allclose(actual=points, desired=np.zeros_like(points), atol=1e-10)


@given(
    points=hnp.arrays(
        shape=st.integers(0, 10).map(lambda x: (x, 2)),
        dtype=np.float64,
        elements=st.floats(-1e10, 1e10),
    ),
    shear_factor=st.floats(-1e10, 1e10),
)
def test_x_shear(points: np.ndarray, shear_factor: float):
    actual = shear(points, x_shear=shear_factor, y_shear=0.)
    desired = np.copy(points)
    desired[:, 0] += shear_factor * points[:, 1]
    assert_allclose(actual=actual, desired=desired, atol=1e-9, rtol=1e-9)


@given(
    points=hnp.arrays(
        shape=st.integers(0, 10).map(lambda x: (x, 2)),
        dtype=np.float64,
        elements=st.floats(-1e10, 1e10),
    ),
    shear_factor=st.floats(-1e10, 1e10),
)
def test_y_shear(points: np.ndarray, shear_factor: float):
    actual = shear(points, y_shear=shear_factor, x_shear=0.)
    desired = np.copy(points)
    desired[:, 1] += shear_factor * points[:, 0]
    assert_allclose(actual=actual, desired=desired, atol=1e-9, rtol=1e-9)


@given(
    x_scale=st.floats(1e-9, 1e10),
    y_scale=st.floats(1e-9, 1e10),
    num_points=st.integers(0, 10),
)
def test_scaling_identity(x_scale: float, y_scale: float, num_points: int):
    points = np.ones((num_points, 2), dtype=np.float64)
    points /= (x_scale, y_scale)
    points = scale(points, x_scale=x_scale, y_scale=y_scale)
    assert_allclose(actual=points, desired=np.ones_like(points), rtol=1e-10)
