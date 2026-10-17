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
