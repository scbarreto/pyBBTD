import pytest
import numpy as np
from pybbtd.btd import BTD


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
