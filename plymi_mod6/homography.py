from typing import Tuple
import numpy as np


__all__ = ["transform_corners"]


def _get_cartesian_to_homogeneous_transform(
    p1: Tuple[float, float],
    p2: Tuple[float, float],
    p3: Tuple[float, float],
    p4: Tuple[float, float],
) -> np.ndarray:
    """
    Given a set of four 2D cartesian coordinates, produces
    the transformation matrix that maps the following basis
    to homogeneous coordinates:
           (1, 0, 0) -> Z(x1, y1, 1)
           (0, 1, 0) -> Z(x2, y2, 1)
           (0, 0, 1) -> Z(x3, y3, 1)
           (1, 1, 1) -> Z(x4, y4, 1)

    Where Z is a real-valued constant

    Parameters
    ----------
    p1 : Tuple[float, float]
        An ordered pair (x1, y1)
    p2 : Tuple[float, float]
        An ordered pair (x2, y2)
    p3 : Tuple[float, float]
        An ordered pair (x3, y3)
    p4 : Tuple[float, float]
        An ordered pair (x4, y4)

    Returns
    -------
    np.ndarray, shape=(3, 3)
        The transformation matrix
    """
    (x1, y1) = p1
    (x2, y2) = p2
    (x3, y3) = p3
    (x4, y4) = p4
    system = np.array(
        [[x1, x2, x3, x4], [y1, y2, y3, y4], [1, 1, 1, 1]], dtype=np.float64
    )
    A = system[:, :3]
    b = system[:, 3]
    return A * np.linalg.solve(A, b)


def transform_corners(
    points: np.ndarray, *, source_corners: np.ndarray, dest_corners: np.ndarray
) -> np.ndarray:
    """ Perform a projective transform on a sequence of 2D points,
    given four corners of a plane in the source coordinate system,
    and the corresponding corners in the coordinate system.

    Parameters
    ----------
    points : array_like, shape=(N, 2)
        A sequence of ordered pairs to undergo the projective transform.

    source_corners : array_like, shape=(4, 2)
        The ordered pairs for the four corners of the original coordinate system.

    dest_corners : array_like, shape=(4, 2)
        The corresponding ordered pairs for the four corners of the destination
        coordinate system.

    Returns
    -------
    numpy.ndarray, shape=(N, 2)
        The array of N projected points.

    Examples
    --------
    >>> import numpy as np
    >>> original_coords = np.array([[0., 0.], [1., 0.], [1., 1.], [0., 1.]])
    >>> # all distances dilated by 2x and all x-coordinates shifted by +1
    >>> new_coords = 2. * original_coords + np.array([1., 0])
    >>> points = np.array([[0.5, 0.5]])  # center of original corners
    >>> transform_corners(points, source_corners=original_coords, dest_corners=new_coords)
    array([[2., 1.]])
    """
    points = np.asarray(points, dtype=np.float64)
    if not (points.ndim == 2 and points.shape[1] == 2):
        raise ValueError(
            "`points` must be array-like with shape-(N, 2), got shape {}".format(
                points.shape
            )
        )

    source_corners = np.asarray(source_corners)

    # convert points to latitude/longitude
    A = _get_cartesian_to_homogeneous_transform(
        source_corners[0], source_corners[1], source_corners[2], source_corners[3]
    )
    A_inv = np.linalg.inv(A)

    B = _get_cartesian_to_homogeneous_transform(
        dest_corners[0], dest_corners[1], dest_corners[2], dest_corners[3]
    )

    # maps: new-corners (homogeneous-basis) <- cartesian-basis <- old-corners
    C = np.matmul(B, A_inv)

    # source_pts:
    #          [[px1, px2, ...]
    #           [py1, py2, ...],
    #           [  1,   1, ...]]
    num_pts = points.shape[0]
    source_pts = np.hstack([points, np.ones((num_pts, 1))]).T

    # homogeneous_pts:
    #          [[x1', y1', z1'],
    #           [x2', y2', z2'],
    #            ...]
    homogeneous_pts = np.matmul(C, source_pts).T

    # destination: (x'', y''), where
    #            x'' = x'/z'
    #            y'' = y'/z'
    z = homogeneous_pts[:, (2,)]  # shape-(N, 1)
    return homogeneous_pts[:, :2] / z
