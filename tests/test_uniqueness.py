import numpy as np
import pytest

from pybbtd.uniqueness import check_uniqueness_BBTD, check_uniqueness_LL1


# --- Tests for check_uniqueness_LL1 ---


def test_LL1_condition_41():
    # Satisfied via cond 41: min(N1, N2) >= L*R and R <= N3
    assert check_uniqueness_LL1(N1=10, N2=10, N3=5, R=2, L=3) is True


def test_LL1_condition_42():
    # Satisfied via cond 42: N3 >= R and kruskal-like sum on N1, N2
    assert check_uniqueness_LL1(N1=10, N2=10, N3=10, R=2, L=3) is True


def test_LL1_condition_43():
    # Satisfied via cond 43: N1 >= L*R and sum on N2, N3 (conds 41, 42 fail since R > N3)
    assert check_uniqueness_LL1(N1=8, N2=8, N3=3, R=4, L=2) is True


def test_LL1_condition_44():
    # Satisfied via cond 44: N2 >= L*R and sum on N1, N3 (conds 41, 42, 43 fail)
    assert check_uniqueness_LL1(N1=6, N2=8, N3=3, R=4, L=2) is True


def test_LL1_condition_45():
    # Satisfied via cond 45: floor(N1*N2/L^2) >= R and kruskal-like sum >= 2R+2
    assert check_uniqueness_LL1(N1=10, N2=10, N3=5, R=2, L=2) is True


def test_LL1_no_condition_satisfied():
    assert check_uniqueness_LL1(N1=5, N2=5, N3=5, R=3, L=3) is False


def test_LL1_small_dims_large_rank():
    assert check_uniqueness_LL1(N1=3, N2=3, N3=2, R=5, L=2) is False


def test_LL1_numpy_integer_L():
    assert check_uniqueness_LL1(N1=10, N2=10, N3=5, R=2, L=np.int64(3)) is True


def test_LL1_invalid_L_type():
    with pytest.raises(NotImplementedError, match="L should be an integer"):
        check_uniqueness_LL1(N1=10, N2=10, N3=10, R=2, L=[2, 3])


# --- Tests for check_uniqueness_BBTD ---


def test_BBTD_both_models_unique():
    assert check_uniqueness_BBTD(N1=10, N2=10, N3=10, N4=10, R=2, L1=2, L2=2) is True


def test_BBTD_only_model1_unique():
    # Model 1 (vectorize first block): BTD(10, 10, 9, R=2, L=2) -> True
    # Model 2 (vectorize second block): BTD(3, 3, 100, R=2, L=3) -> False
    assert check_uniqueness_BBTD(N1=3, N2=3, N3=10, N4=10, R=2, L1=3, L2=2) is True


def test_BBTD_only_model2_unique():
    # Model 1 (vectorize first block): BTD(3, 3, 100, R=2, L=3) -> False
    # Model 2 (vectorize second block): BTD(10, 10, 9, R=2, L=2) -> True
    assert check_uniqueness_BBTD(N1=10, N2=10, N3=3, N4=3, R=2, L1=2, L2=3) is True


def test_BBTD_neither_model_unique():
    assert check_uniqueness_BBTD(N1=3, N2=3, N3=3, N4=3, R=5, L1=3, L2=3) is False


def test_BBTD_vanilla_notebook_example():
    # Dims from example_vanilla_bbtd.ipynb: (20, 25, 10, 10), R=2, L1=5, L2=2
    assert check_uniqueness_BBTD(N1=20, N2=25, N3=10, N4=10, R=2, L1=5, L2=2) is True


def test_BBTD_cov_notebook_example():
    # Dims from example_cov_bbtd.ipynb: (20, 25, 4, 4), R=3, L1=3, L2=2
    assert check_uniqueness_BBTD(N1=20, N2=25, N3=4, N4=4, R=3, L1=3, L2=2) is True


def test_BBTD_numpy_integer_L1_L2():
    assert (
        check_uniqueness_BBTD(
            N1=20, N2=25, N3=10, N4=10, R=2, L1=np.int64(5), L2=np.int64(2)
        )
        is True
    )


def test_BBTD_invalid_L1_type():
    with pytest.raises(NotImplementedError, match="L1 should be an integer"):
        check_uniqueness_BBTD(N1=10, N2=10, N3=10, N4=10, R=2, L1=[2], L2=2)


def test_BBTD_invalid_L2_type():
    with pytest.raises(NotImplementedError, match="L2 should be an integer"):
        check_uniqueness_BBTD(N1=10, N2=10, N3=10, N4=10, R=2, L1=2, L2=[2])
