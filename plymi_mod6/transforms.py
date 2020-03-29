"""
Contains implementations of various linear transforms applied to shape-(N, 2) arrays
of (x, y) coordinates.
"""

import numpy as np
from numpy import ndarray

__all__ = ["translate", "rotate", "shear", "scale"]


def translate(points: ndarray, *, x_shift: float, y_shift: float) -> ndarray:
    """
       [(x1, y1), (x2, y2), ...] -> [(x1 + dx, y1 + dy), (x2 + dx, y2 + dy), ...]

    Parameters
    ----------
    points : ndarray, shape=(N, 2)
        Length-N array of (x, y) coordinates

    x_shift : float

    y_shift : float

    Returns
    -------
    translated_points : ndarray, shape=(N, 2)
        Length-N array of (x, y) coordinates, each translated by `(x_shift, y_shift)`
    """
    return points + np.array([x_shift, y_shift])


def rotate(points: ndarray, deg_rot: float) -> ndarray:
    """
    Rotates each (x, y) point CCW relative to +x by the specified number of degrees.

       [(x1, y1), (x2, y2), ...] -> [(Rx1, Ry1), (Rx2, Ry2 ), ...]

    Parameters
    ----------
    points : ndarray, shape=(N, 2)
        Length-N array of (x, y) coordinates

    deg_rot : float
        The degrees to rotate a point (x, y) CCW relative to +x

    Returns
    -------
    translated_points : ndarray, shape=(N, 2)
        Length-N array of (x, y) coordinates, each having been rotated by
        the amount `deg_rot`.
    """
    angle_rad = deg_rot * np.pi / 180

    rot_matrix = np.array(
        [
            [np.cos(angle_rad), -np.sin(angle_rad)],
            [np.sin(angle_rad), np.cos(angle_rad)],
        ]
    )

    return np.matmul(points, rot_matrix.T)


def shear(points: ndarray, *, x_shear: float, y_shear: float) -> ndarray:
    """
    Applies a shear along the x and y dimensions to each (x, y) point

    Parameters
    ----------
    points : ndarray, shape=(N, 2)
        Length-N array of (x, y) coordinates

    x_shear : float
        The shear factor applied along the x-axis

    y_shear : float
        The shear factor applied along the y-axis

    Returns
    -------
    translated_points : ndarray, shape=(N, 2)
        Length-N array of (x, y) that have undergone shearing
    """
    shear_matrix = np.array(
        [
            [1.0, x_shear],
            [y_shear, 1.0],
        ]
    )
    return np.matmul(points, shear_matrix.T)


def scale(points: ndarray, *, x_scale: float, y_scale: float) -> ndarray:
    """
    Applies a scale along the x and y dimensions to each (x, y) point

    Parameters
    ----------
    points : ndarray, shape=(N, 2)
        Length-N array of (x, y) coordinates

    x_scale : float
        The scale factor applied along the x-axis

    y_scale : float
        The scale factor applied along the y-axis

    Returns
    -------
    translated_points : ndarray, shape=(N, 2)
        Length-N array of (x, y) that have undergone scale
    """
    scale_matrix = np.array(
        [
            [x_scale, 0.0],
            [0.0, y_scale],
        ]
    )
    return np.matmul(points, scale_matrix.T)
