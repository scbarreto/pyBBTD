import pytest
import numpy as np
from pybbtd.btd import BTD


def test_valid_initialization_with_int_L():
    btd = BTD(R=3, L=2, mode="LL1")
    assert btd.rank == 3
    np.testing.assert_array_equal(btd.L, np.array([2, 2, 2]))
    assert btd.mode == "LL1"


def test_valid_initialization_with_list_L():
    btd = BTD(R=3, L=[2, 3, 4], mode="L1L")
    assert btd.rank == 3
    np.testing.assert_array_equal(btd.L, np.array([2, 3, 4]))
    assert btd.mode == "L1L"


def test_invalid_R_negative():
    with pytest.raises(ValueError, match="R should be a positive integer."):
        BTD(R=-1, L=2, mode="LL1")


def test_invalid_R_non_integer():
    with pytest.raises(ValueError, match="R should be a positive integer."):
        BTD(R=2.5, L=2, mode="LL1")


def test_invalid_L_length_mismatch():
    with pytest.raises(
        ValueError,
        match="L should be either a single integer or a list/array of length R.",
    ):
        BTD(R=3, L=[2, 3], mode="LL1")


def test_invalid_L_type():
    with pytest.raises(
        ValueError,
        match="L should be either a single integer or a list/array of length R.",
    ):
        BTD(R=3, L="invalid", mode="LL1")


def test_invalid_mode():
    with pytest.raises(ValueError):
        BTD(R=3, L=2, mode="invalid")
