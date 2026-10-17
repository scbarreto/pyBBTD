import numpy as np
import numpy.testing as npt

from pybbtd.btd import spatial_maps
from pybbtd.datasets import load_bbtd_board, make_blob_factors


def test_load_bbtd_board():
    T = load_bbtd_board()
    assert T.shape == (160, 160, 4)
    assert 0.5 <= T[..., 0].min() and T[..., 0].max() <= 1
    # one unit polarization state per region
    npt.assert_allclose(np.linalg.norm(T[..., 1:], axis=-1), 1)


def test_make_blob_factors():
    A, B = make_blob_factors(30, 40, R=3, L=2, seed=1)
    assert A.shape == (30, 6)
    assert B.shape == (40, 6)
    assert A.min() >= 0
    assert B.min() >= 0
    for spatial_map in spatial_maps(A, B, 2):
        assert np.linalg.matrix_rank(spatial_map) == 2


def test_make_blob_factors_is_reproducible():
    first = make_blob_factors(10, 12, R=2, L=1, seed=7)
    second = make_blob_factors(10, 12, R=2, L=1, seed=7)
    for X, Y in zip(first, second, strict=True):
        npt.assert_array_equal(X, Y)
