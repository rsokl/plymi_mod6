from typing import Callable

import hypothesis.extra.numpy as hnp
import hypothesis.strategies as st
import numpy as np
from hypothesis import given
from numpy.testing import assert_allclose

from plymi_mod6.homography import transform_corners
from plymi_mod6.transforms import rotate, scale, shear, translate

from .custom_strategies import quad_corners


@given(
    ref_corners=quad_corners(),
    points=hnp.arrays(
        shape=st.tuples(st.integers(1, 10), st.just(2)),
        dtype=np.float64,
        elements=st.floats(-1e6, 1e6),
    ),
)
def test_identity_homography(ref_corners: np.ndarray, points: np.ndarray):
    transformed_points = transform_corners(
        points, source_corners=ref_corners, dest_corners=ref_corners
    )
    assert_allclose(actual=transformed_points, desired=points, atol=1e-7, rtol=1e-7)


def assert_matches_mapping(
    *,
    source_corners: np.ndarray,
    points: np.ndarray,
    linear_transform: Callable[[np.ndarray], np.ndarray],
    atol: float = 1e-8,
    rtol: float = 1e-8
):
    """
    Tests that the projective transformation can reproduce the
    linear transform
    """
    dest_corners = linear_transform(source_corners)

    actual_points = transform_corners(
        points, source_corners=source_corners, dest_corners=dest_corners
    )

    desired_points = linear_transform(points)

    assert_allclose(actual=actual_points, desired=desired_points, atol=atol, rtol=rtol)


def assert_metamorphic_mapping(
    *,
    source_corners: np.ndarray,
    dest_corners: np.ndarray,
    points: np.ndarray,
    linear_transform: Callable[[np.ndarray], np.ndarray],
    atol: float = 1e-8,
    rtol: float = 1e-8
):
    """
    Tests that projecting points via:
                src_corners -> LinMap(dst_corners)

    is equivalent to performing to projecting points via:
                src_corners -> dst_corners

    and then applying the linear mapping to the projected points.
    """
    dest_points = transform_corners(
        points, source_corners=source_corners, dest_corners=dest_corners
    )
    desired_points = linear_transform(dest_points)

    lin_map_dest_corners = linear_transform(dest_corners)

    actual_points = transform_corners(
        points, source_corners=source_corners, dest_corners=lin_map_dest_corners
    )

    assert_allclose(actual=actual_points, desired=desired_points, atol=atol, rtol=rtol)


@given(
    source_corners=quad_corners(),
    points=hnp.arrays(
        shape=st.tuples(st.integers(1, 10), st.just(2)),
        dtype=np.float64,
        elements=st.floats(-1e2, 1e2),
    ),
    deg_rot=st.floats(-360, 360),
)
def test_projective_rotation(
    source_corners: np.ndarray, points: np.ndarray, deg_rot: float,
):
    lin_transform = lambda x: rotate(x, deg_rot)

    assert_matches_mapping(
        source_corners=source_corners, points=points, linear_transform=lin_transform,
        atol=1e-6, rtol=1e-6
    )


@given(
    source_corners=quad_corners(),
    dest_corners=quad_corners(),
    points=hnp.arrays(
        shape=st.tuples(st.integers(1, 10), st.just(2)),
        dtype=np.float64,
        elements=st.floats(-1e2, 1e2),
    ),
    deg_rot=st.floats(-360, 360),
)
def test_metamorphic_rotation(
    source_corners: np.ndarray,
    dest_corners: np.ndarray,
    points: np.ndarray,
    deg_rot: float,
):
    lin_transform = lambda x: rotate(x, deg_rot)

    assert_metamorphic_mapping(
        source_corners=source_corners,
        dest_corners=dest_corners,
        points=points,
        linear_transform=lin_transform,
    )


@given(
    source_corners=quad_corners(),
    points=hnp.arrays(
        shape=st.tuples(st.integers(1, 10), st.just(2)),
        dtype=np.float64,
        elements=st.floats(-1e2, 1e2),
    ),
    x_scale=st.floats(-1e2, 1e2).filter(lambda x: x),  # exclude singular transforms
    y_scale=st.floats(-1e2, 1e2).filter(lambda x: x),  # exclude singular transforms
)
def test_projective_scaling(
    source_corners: np.ndarray, points: np.ndarray, x_scale: float, y_scale: float,
):
    lin_transform = lambda x: scale(x, x_scale=x_scale, y_scale=y_scale)

    assert_matches_mapping(
        source_corners=source_corners,
        points=points,
        linear_transform=lin_transform,
        atol=1e-6,
        rtol=1e-6,
    )


@given(
    source_corners=quad_corners(),
    dest_corners=quad_corners(),
    points=hnp.arrays(
        shape=st.tuples(st.integers(1, 10), st.just(2)),
        dtype=np.float64,
        elements=st.floats(-1e2, 1e2),
    ),
    x_scale=st.floats(-1e2, 1e2).filter(lambda x: x),  # exclude singular transforms
    y_scale=st.floats(-1e2, 1e2).filter(lambda x: x),  # exclude singular transforms
)
def test_metamorphic_scaling(
    source_corners: np.ndarray,
    dest_corners: np.ndarray,
    points: np.ndarray,
    x_scale: float,
    y_scale: float,
):
    lin_transform = lambda x: scale(x, x_scale=x_scale, y_scale=y_scale)

    assert_metamorphic_mapping(
        source_corners=source_corners,
        dest_corners=dest_corners,
        points=points,
        linear_transform=lin_transform,
        atol=1e-6,
        rtol=1e-6,
    )


@given(
    source_corners=quad_corners(),
    points=hnp.arrays(
        shape=st.tuples(st.integers(1, 10), st.just(2)),
        dtype=np.float64,
        elements=st.floats(-1e2, 1e2),
    ),
    x_shear=st.floats(-1e2, 1e2),
    y_shear=st.floats(-1e2, 1e2),
)
def test_projective_shearing(
    source_corners: np.ndarray, points: np.ndarray, x_shear: float, y_shear: float,
):
    lin_transform = lambda x: shear(x, x_shear=x_shear, y_shear=y_shear)

    assert_matches_mapping(
        source_corners=source_corners,
        points=points,
        linear_transform=lin_transform,
        atol=1e-5,
        rtol=1e-5,
    )


@given(
    source_corners=quad_corners(),
    dest_corners=quad_corners(),
    points=hnp.arrays(
        shape=st.tuples(st.integers(1, 10), st.just(2)),
        dtype=np.float64,
        elements=st.floats(-1e3, 1e3),
    ),
    x_shear=st.floats(-1e2, 1e2),
    y_shear=st.floats(-1e2, 1e2),
)
def test_metamorphic_shearing(
    source_corners: np.ndarray,
    dest_corners: np.ndarray,
    points: np.ndarray,
    x_shear: float,
    y_shear: float,
):
    lin_transform = lambda x: shear(x, x_shear=x_shear, y_shear=y_shear)

    assert_metamorphic_mapping(
        source_corners=source_corners,
        dest_corners=dest_corners,
        points=points,
        linear_transform=lin_transform,
        atol=1e-5,
        rtol=1e-5,
    )


@given(
    source_corners=quad_corners(),
    points=hnp.arrays(
        shape=st.tuples(st.integers(1, 10), st.just(2)),
        dtype=np.float64,
        elements=st.floats(-1e3, 1e3),
    ),
    x_shift=st.floats(-1e3, 1e3),
    y_shift=st.floats(-1e3, 1e3),
)
def test_projective_translation(
    source_corners: np.ndarray, points: np.ndarray, x_shift: float, y_shift: float,
):
    lin_transform = lambda x: translate(x, x_shift=x_shift, y_shift=y_shift)

    assert_matches_mapping(
        source_corners=source_corners,
        points=points,
        linear_transform=lin_transform,
        atol=1e-6,
        rtol=1e-6,
    )


@given(
    source_corners=quad_corners(),
    dest_corners=quad_corners(),
    points=hnp.arrays(
        shape=st.tuples(st.integers(1, 10), st.just(2)),
        dtype=np.float64,
        elements=st.floats(-1e3, 1e3),
    ),
    x_shift=st.floats(-1e3, 1e3),
    y_shift=st.floats(-1e3, 1e3),
)
def test_metamorphic_translation(
    source_corners: np.ndarray,
    dest_corners: np.ndarray,
    points: np.ndarray,
    x_shift: float,
    y_shift: float,
):
    lin_transform = lambda x: translate(x, x_shift=x_shift, y_shift=y_shift)

    assert_metamorphic_mapping(
        source_corners=source_corners,
        dest_corners=dest_corners,
        points=points,
        linear_transform=lin_transform,
        atol=1e-6,
        rtol=1e-6,
    )
