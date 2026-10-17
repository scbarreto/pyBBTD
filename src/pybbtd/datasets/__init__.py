"""Example data for the tutorials."""

from pathlib import Path

import numpy as np

_DATA_DIR = Path(__file__).parent


def load_bbtd_board():
    """Load the synthetic "BBTD" polarization image.

    A 160 x 160 image where each pixel holds a Stokes vector
    ``[S0, S1, S2, S3]``. The letters B, B, T, D and the background each
    have one polarization state, given by unit values of S1, S2 or S3,
    while S0 is a random intensity between 0.5 and 1.

    Returns
    -------
    np.ndarray
        Tensor of shape ``(160, 160, 4)``.
    """
    return np.load(_DATA_DIR / "bbtd_board.npy")


def make_blob_factors(n_rows, n_cols, R, L, width=None, seed=None):
    """Draw non-negative spatial factors whose maps look like bright spots.

    The map of term ``r``, ``A_r @ B_r.T``, is a sum of ``L`` Gaussian
    blobs at random positions. Each blob is separable (a column profile
    times a row profile), so the map has rank ``L``.

    Parameters
    ----------
    n_rows, n_cols : int
        Image size.
    R : int
        Number of terms.
    L : int
        Number of blobs per term (the rank of each map).
    width : float, optional
        Standard deviation of the blobs, in pixels. Defaults to a tenth of
        the smaller image side.
    seed : int, optional
        Seed of the random positions.

    Returns
    -------
    A : np.ndarray
        Shape ``(n_rows, R * L)``.
    B : np.ndarray
        Shape ``(n_cols, R * L)``.
    """
    rng = np.random.default_rng(seed)
    if width is None:
        width = min(n_rows, n_cols) / 10

    row_centers = rng.uniform(0, n_rows, R * L)
    col_centers = rng.uniform(0, n_cols, R * L)
    rows = np.arange(n_rows)[:, None]
    cols = np.arange(n_cols)[:, None]

    A = np.exp(-0.5 * ((rows - row_centers) / width) ** 2)
    B = np.exp(-0.5 * ((cols - col_centers) / width) ** 2)
    return A, B
