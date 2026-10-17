import numpy as np
import pytest

from pybbtd.uniqueness import (
    _ll1_conditions,
    check_uniqueness_bbtd,
    check_uniqueness_ll1,
)

# ──────────────────────────────────────────────────────────────────────────────
# Rank-(L, L, 1) BTD
# ──────────────────────────────────────────────────────────────────────────────


@pytest.mark.parametrize(
    ("dims_and_ranks", "expected"),
    [
        # min(N1, N2) = 6 < L*R = 8; N3 = 4 >= R and 3 + 3 >= R + 2;
        # floor(36 / 4) = 9 >= R and 3 + 3 + 4 >= 2R + 2
        ((6, 6, 4, 4, 2), [False, True, False, False, True]),
        # N1 = 8 >= L*R and min(3, 4) + min(3, 4) >= R + 2; N3 = 3 < R
        ((8, 6, 3, 4, 2), [False, False, True, False, True]),
        # same as above with the first two modes swapped
        ((6, 8, 3, 4, 2), [False, False, False, True, True]),
        # only the last condition: floor(64 / 4) = 16 >= R and 4 + 4 + 4 >= 12
        ((8, 8, 4, 5, 2), [False, False, False, False, True]),
        # only the first condition (a single rank-one term)
        ((1, 1, 1, 1, 1), [True, False, False, False, False]),
        # every condition
        ((10, 10, 5, 2, 3), [True, True, True, True, True]),
        # no condition
        ((5, 5, 5, 3, 3), [False, False, False, False, False]),
    ],
)
def test_ll1_conditions_one_by_one(dims_and_ranks, expected):
    assert _ll1_conditions(*dims_and_ranks) == expected
    assert check_uniqueness_ll1(*dims_and_ranks) is any(expected)


@pytest.mark.parametrize(
    ("N1", "N2", "N3", "R", "L"),
    [(5, 5, 5, 3, 3), (3, 3, 2, 5, 2), (3, 8, 8, 4, 2)],
)
def test_ll1_no_condition_satisfied(N1, N2, N3, R, L):
    assert check_uniqueness_ll1(N1, N2, N3, R, L) is False


def test_ll1_accepts_numpy_integer_L():
    assert check_uniqueness_ll1(N1=10, N2=10, N3=5, R=2, L=np.int64(3)) is True


def test_ll1_rejects_varying_L():
    with pytest.raises(NotImplementedError, match="L should be an integer"):
        check_uniqueness_ll1(N1=10, N2=10, N3=10, R=2, L=[2, 3])


# ──────────────────────────────────────────────────────────────────────────────
# Rank-(L1, L1, L2, L2) BBTD: unique if either reshaped BTD is unique
# ──────────────────────────────────────────────────────────────────────────────


def test_bbtd_both_reshapings_unique():
    assert check_uniqueness_bbtd(N1=10, N2=10, N3=10, N4=10, R=2, L1=2, L2=2) is True


def test_bbtd_only_first_reshaping_unique():
    # merge modes 1, 2: BTD(10, 10, 9, R=2, L=2) is unique
    # merge modes 3, 4: BTD(3, 3, 100, R=2, L=3) is not
    assert check_uniqueness_ll1(10, 10, 9, 2, 2) is True
    assert check_uniqueness_ll1(3, 3, 100, 2, 3) is False
    assert check_uniqueness_bbtd(N1=3, N2=3, N3=10, N4=10, R=2, L1=3, L2=2) is True


def test_bbtd_only_second_reshaping_unique():
    assert check_uniqueness_bbtd(N1=10, N2=10, N3=3, N4=3, R=2, L1=2, L2=3) is True


def test_bbtd_neither_reshaping_unique():
    assert check_uniqueness_bbtd(N1=3, N2=3, N3=3, N4=3, R=5, L1=3, L2=3) is False


@pytest.mark.parametrize(
    ("dims", "R", "L1", "L2"),
    [((20, 25, 10, 10), 2, 5, 2), ((20, 25, 4, 4), 3, 3, 2)],
)
def test_bbtd_tutorial_settings_are_unique(dims, R, L1, L2):
    assert check_uniqueness_bbtd(*dims, R, L1, L2) is True


def test_bbtd_accepts_numpy_integers():
    assert (
        check_uniqueness_bbtd(20, 25, 10, 10, R=2, L1=np.int64(5), L2=np.int64(2))
        is True
    )


@pytest.mark.parametrize(("L1", "L2", "name"), [([2], 2, "L1"), (2, [2], "L2")])
def test_bbtd_rejects_non_integer_ranks(L1, L2, name):
    with pytest.raises(NotImplementedError, match=f"{name} should be an integer"):
        check_uniqueness_bbtd(N1=10, N2=10, N3=10, N4=10, R=2, L1=L1, L2=L2)
