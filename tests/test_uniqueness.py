import pytest
from pybbtd.uniqueness import check_uniqueness_LL1


def test_valid_uniqueness_condition_41():
    # Test where cond_BTD_41 is true
    assert check_uniqueness_LL1(N1=10, N2=10, N3=5, R=2, L=3) is True


def test_valid_uniqueness_condition_42():
    # Test where cond_BTD_42 is true
    assert check_uniqueness_LL1(N1=10, N2=10, N3=10, R=2, L=3) is True


def test_valid_uniqueness_condition_43():
    # Test where cond_BTD_43 is true
    assert check_uniqueness_LL1(N1=10, N2=5, N3=5, R=2, L=3) is False


def test_valid_uniqueness_condition_44():
    # Test where cond_BTD_44 is true
    assert check_uniqueness_LL1(N1=5, N2=10, N3=5, R=2, L=3) is False


def test_valid_uniqueness_condition_45():
    # Test where cond_BTD_45 is true
    assert check_uniqueness_LL1(N1=10, N2=10, N3=5, R=2, L=2) is True


def test_no_condition_is_true():
    # Test where none of the conditions are true
    assert check_uniqueness_LL1(N1=5, N2=5, N3=5, R=3, L=3) is False


def test_invalid_L_type():
    # Test where L is not an integer
    with pytest.raises(NotImplementedError, match="L should be an integer"):
        check_uniqueness_LL1(N1=10, N2=10, N3=10, R=2, L=[2, 3])
