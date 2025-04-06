import pytest
import numpy as np
import tensorly as tly
from tensorly.tenalg import outer, khatri_rao

from pybbtd.btd import BTD, validate_R_L, constraint_matrix


def test_valid_initialization_with_int_L():
    btd = BTD(dims=(15, 16, 10), R=3, L=2, block_mode="LL1")
    assert btd.rank == 3
    np.testing.assert_array_equal(btd.L, np.array([2, 2, 2]))
    assert btd.block_mode == "LL1"


def test_valid_initialization_with_list_L():
    btd = BTD(dims=(20, 19, 18), R=3, L=[2, 3, 4], block_mode="L1L")
    assert btd.rank == 3
    np.testing.assert_array_equal(btd.L, np.array([2, 3, 4]))
    assert btd.block_mode == "L1L"


def test_invalid_R_negative():
    with pytest.raises(ValueError, match="R should be a positive integer."):
        BTD(dims=(10, 10, 10), R=-1, L=2, block_mode="LL1")


def test_invalid_R_non_integer():
    with pytest.raises(ValueError, match="R should be a positive integer."):
        BTD(dims=(10, 10, 10), R=2.5, L=2, block_mode="LL1")


def test_invalid_L_length_mismatch():
    with pytest.raises(
        ValueError,
        match="L should be either a single integer or a list/array of length R.",
    ):
        BTD(dims=(10, 10, 10), R=3, L=[2, 3], block_mode="LL1")


def test_invalid_L_type():
    with pytest.raises(
        ValueError,
        match="L should be either a single integer or a list/array of length R.",
    ):
        BTD(dims=(10, 10, 10), R=3, L="invalid", block_mode="LL1")


def test_invalid_block_mode():
    with pytest.raises(ValueError):
        BTD(dims=(10, 10, 10), R=3, L=2, block_mode="invalid")


def test_unfoldings():
    R, L = validate_R_L(3, 2)
    Lsum = np.array(L).sum()
    dims = (13, 12, 11)
    theta = constraint_matrix(R, L)
    # random factors for LL1
    A = np.random.rand(dims[0], Lsum)
    B = np.random.rand(dims[1], Lsum)
    C = np.random.rand(dims[2], R)

    # compute tensor from definition
    T = np.zeros(dims)
    ind = 0
    for r in range(R):
        Ar = A[:, ind : ind + L[r]]
        Br = B[:, ind : ind + L[r]]
        ind += L[r]
        T += outer([Ar @ Br.T, C[:, r]])

    # compare with unfoldings
    T0 = A @ (khatri_rao([B, C @ theta])).T
    assert np.allclose(tly.unfold(T, 0), T0)

    T1 = B @ (khatri_rao([A, C @ theta])).T
    assert np.allclose(tly.unfold(T, 1), T1)

    T2 = C @ theta @ (khatri_rao([A, B])).T
    assert np.allclose(tly.unfold(T, 2), T2)
