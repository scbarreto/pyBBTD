import numpy as np
import numpy.testing as npt

from pybbtd.datasets import load_bbtd_board


def test_load_bbtd_board():
    T = load_bbtd_board()
    assert T.shape == (160, 160, 4)
    assert 0.5 <= T[..., 0].min() and T[..., 0].max() <= 1
    # one unit polarization state per region
    npt.assert_allclose(np.linalg.norm(T[..., 1:], axis=-1), 1)
