import numpy as np

__all__ = ["pairwise_dists"]


def pairwise_dists(x, y):
    """ Computing pairwise distances using memory-efficient
    vectorization.

    Parameters
    ----------
    x : numpy.ndarray, shape=(M, D)
    y : numpy.ndarray, shape=(N, D)

    Returns
    -------
    numpy.ndarray, shape=(M, N)
        The Euclidean distance between each pair of
        rows between `x` and `y`."""
    dists = -2 * np.matmul(x, y.T)
    dists += np.sum(x**2, axis=1)[:, np.newaxis]
    dists += np.sum(y**2, axis=1)
    return np.sqrt(dists)
