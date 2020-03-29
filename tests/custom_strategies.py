import hypothesis.strategies as st
import numpy as np


@st.composite
def quad_corners(
    draw, *, corner_magnitude=1e2, min_separation=0.01
) -> st.SearchStrategy[np.ndarray]:
    """
    A Hypothesis strategy for four corners of a quadrilateral.

    The corners are guaranteed to have counter-clockwise ("right-handed") ordering.

    Parameters
    ----------
    corner_magnitude : float, optional (default=1e6)
        The maximum size - in magnitude - that any of the coordinates
        can take on.

    min_separation : float, optional (default=1e-2)
        The smallest guaranteed margin between two corners along a
        single dimension.

    Returns
    -------
    SearchStrategy[np.ndarray]
       shape-(4, 2) array of four ordered pairs (float-64)
    """

    min_x = draw(st.floats(-corner_magnitude, corner_magnitude))
    max_x = draw(st.floats(min_separation, corner_magnitude)) + min_x

    min_y = draw(st.floats(-corner_magnitude, corner_magnitude))
    max_y = draw(st.floats(min_separation, corner_magnitude)) + min_y

    shift = draw(st.integers(min_value=0, max_value=3))

    array_corners = np.array(
        [(min_x, min_y), (max_x, min_y), (max_x, max_y), (min_x, max_y)]
    )

    # "roll" the ordering of the points such that the "upper-left" point
    # does not always come first, however the right-handed ordering of
    # the corners is preserved
    array_corners = np.roll(array_corners, shift=shift, axis=0)

    return array_corners
