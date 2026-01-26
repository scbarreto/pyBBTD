import pytest
import numpy as np
import numpy.testing as npt
import tensorly as tly
from tensorly.tenalg import outer, khatri_rao
from pybbtd.btd import BTD, _validate_R_L, _constraint_matrix, factors_to_tensor
from pybbtd.solvers.btd_als import BTD_ALS, init_BTD_factors


# ──────────────────────────────────────────────────────────────────────────────
# Section 1: BTD class
# ──────────────────────────────────────────────────────────────────────────────


def test_valid_initialization_with_int_L():
    model = BTD(dims=(15, 16, 10), R=3, L=2, block_mode="LL1")
    assert model.rank == 3
    npt.assert_array_equal(model.L, np.array([2, 2, 2]))
    assert model.block_mode == "LL1"
    assert model.dims == (15, 16, 10)


def test_valid_initialization_with_list_L():
    model = BTD(dims=(20, 19, 18), R=3, L=[2, 3, 4], block_mode="L1L")
    assert model.rank == 3
    npt.assert_array_equal(model.L, np.array([2, 3, 4]))
    assert model.block_mode == "L1L"


def test_valid_initialization_stores_none_state():
    model = BTD(dims=(10, 10, 10), R=2, L=2, block_mode="LL1")
    assert model.factors is None
    assert model.tensor is None
    assert model.fit_error is None


def test_invalid_dims_not_list():
    with pytest.raises(ValueError, match="list or tuple of three positive integers"):
        BTD(dims="bad", R=2, L=2, block_mode="LL1")


def test_invalid_dims_wrong_length():
    with pytest.raises(ValueError, match="list or tuple of three positive integers"):
        BTD(dims=(10, 10), R=2, L=2, block_mode="LL1")


def test_invalid_dims_non_positive():
    with pytest.raises(ValueError, match="list or tuple of three positive integers"):
        BTD(dims=(10, 0, 5), R=2, L=2, block_mode="LL1")


def test_invalid_R_negative():
    with pytest.raises(ValueError, match="R should be a positive integer"):
        BTD(dims=(10, 10, 10), R=-1, L=2, block_mode="LL1")


def test_invalid_R_non_integer():
    with pytest.raises(ValueError, match="R should be a positive integer"):
        BTD(dims=(10, 10, 10), R=2.5, L=2, block_mode="LL1")


def test_invalid_L_length_mismatch():
    with pytest.raises(
        ValueError,
        match="L should be either a single integer or a list/array of length R",
    ):
        BTD(dims=(10, 10, 10), R=3, L=[2, 3], block_mode="LL1")


def test_invalid_L_type():
    with pytest.raises(
        ValueError,
        match="L should be either a single integer or a list/array of length R",
    ):
        BTD(dims=(10, 10, 10), R=3, L="invalid", block_mode="LL1")


def test_invalid_L_non_positive_in_list():
    with pytest.raises(
        ValueError, match="Each element in L should be greater than zero"
    ):
        BTD(dims=(10, 10, 10), R=3, L=[2, 0, 1], block_mode="LL1")


def test_invalid_block_mode():
    with pytest.raises(ValueError, match="Invalid mode"):
        BTD(dims=(10, 10, 10), R=3, L=2, block_mode="invalid")


def test_constraint_matrix_shape():
    R, L = 3, 2
    model = BTD(dims=(10, 10, 10), R=R, L=L, block_mode="LL1")
    theta = model.get_constraint_matrix()
    Lsum = np.array(model.L).sum()
    assert theta.shape == (R, Lsum)


def test_factors_to_tensor_shape():
    np.random.seed(42)
    dims = (13, 12, 11)
    R, L = 3, 2
    model = BTD(dims=dims, R=R, L=L, block_mode="LL1")
    Lsum = np.array(model.L).sum()
    A = np.random.randn(dims[0], Lsum)
    B = np.random.randn(dims[1], Lsum)
    C = np.random.randn(dims[2], R)
    theta = model.get_constraint_matrix()
    T = factors_to_tensor(A, B, C, theta)
    assert T.shape == dims


def test_fit_invalid_algorithm():
    model = BTD(dims=(10, 10, 10), R=2, L=2, block_mode="LL1")
    T = np.random.randn(10, 10, 10)
    with pytest.raises(UserWarning, match="Algorithm not implemented yet"):
        model.fit(T, algorithm="INVALID")


def test_unfoldings():
    np.random.seed(42)
    R, L = _validate_R_L(3, 2)
    Lsum = np.array(L).sum()
    dims = (13, 12, 11)
    theta = _constraint_matrix(R, L)

    A = np.random.rand(dims[0], Lsum)
    B = np.random.rand(dims[1], Lsum)
    C = np.random.rand(dims[2], R)

    # Compute tensor from definition
    T = np.zeros(dims)
    ind = 0
    for r in range(R):
        Ar = A[:, ind : ind + L[r]]
        Br = B[:, ind : ind + L[r]]
        ind += L[r]
        T += outer([Ar @ Br.T, C[:, r]])

    # Compare with unfoldings
    T0 = A @ (khatri_rao([B, C @ theta])).T
    npt.assert_allclose(tly.unfold(T, 0), T0)

    T1 = B @ (khatri_rao([A, C @ theta])).T
    npt.assert_allclose(tly.unfold(T, 1), T1)

    T2 = C @ theta @ (khatri_rao([A, B])).T
    npt.assert_allclose(tly.unfold(T, 2), T2)


# ──────────────────────────────────────────────────────────────────────────────
# Section 2: ALS solver
# ──────────────────────────────────────────────────────────────────────────────


def test_als_init_random_shapes():
    np.random.seed(0)
    dims = (15, 14, 10)
    R, L = 3, 2
    model = BTD(dims=dims, R=R, L=L, block_mode="LL1")
    Lsum = np.array(model.L).sum()
    A, B, C = init_BTD_factors(model, strat="random")
    assert A.shape == (dims[0], Lsum)
    assert B.shape == (dims[1], Lsum)
    assert C.shape == (dims[2], R)


def test_als_init_svd_shapes():
    np.random.seed(0)
    dims = (15, 14, 10)
    R, L = 3, 2
    model = BTD(dims=dims, R=R, L=L, block_mode="LL1")
    Lsum = np.array(model.L).sum()
    T = np.random.randn(*dims)
    A, B, C = init_BTD_factors(model, strat="svd", T=T)
    assert A.shape == (dims[0], Lsum)
    assert B.shape == (dims[1], Lsum)
    assert C.shape == (dims[2], R)


def test_als_init_svd_requires_T():
    model = BTD(dims=(15, 14, 10), R=3, L=2, block_mode="LL1")
    with pytest.raises(ValueError, match="SVD init requires input data T"):
        init_BTD_factors(model, strat="svd", T=None)


def test_als_init_invalid_strategy():
    model = BTD(dims=(15, 14, 10), R=3, L=2, block_mode="LL1")
    with pytest.raises(ValueError, match="not implemented"):
        init_BTD_factors(model, strat="bad")


def test_als_invalid_model_type():
    T = np.random.randn(10, 10, 10)
    with pytest.raises(TypeError, match="must be an instance of the BTD class"):
        BTD_ALS("not_a_model", T)


def test_als_invalid_tensor_type():
    model = BTD(dims=(10, 10, 10), R=2, L=2, block_mode="LL1")
    with pytest.raises(TypeError, match="must be a numpy array"):
        BTD_ALS(model, None)


def test_als_mismatched_dims():
    model = BTD(dims=(10, 10, 10), R=2, L=2, block_mode="LL1")
    T = np.random.randn(5, 5, 5)
    with pytest.raises(ValueError, match="do not match"):
        BTD_ALS(model, T)


# ──────────────────────────────────────────────────────────────────────────────
# Section 3: ALS fit convergence (all block modes)
# ──────────────────────────────────────────────────────────────────────────────


def test_fit_LL1():
    np.random.seed(202505)
    X = BTD([20, 18, 16], 3, 2, block_mode="LL1")

    A0, B0, C0 = init_BTD_factors(X)
    theta = X.get_constraint_matrix()
    T_gt = factors_to_tensor(A0, B0, C0, theta)

    X.fit(T_gt, abs_tol=1e-14)
    npt.assert_allclose(X.tensor, T_gt, rtol=1e-5, atol=1e-8)


def test_fit_1LL():
    np.random.seed(202505)
    X = BTD([16, 20, 21], 2, 2, block_mode="1LL")

    dims = X.dims
    R = X.rank
    L = X.L
    Lsum = np.array(L).sum()
    A0 = np.random.randn(dims[0], R)
    B0 = np.random.randn(dims[1], Lsum)
    C0 = np.random.randn(dims[2], Lsum)
    theta = X.get_constraint_matrix()
    T_gt = factors_to_tensor(A0, B0, C0, theta, block_mode="1LL")

    X.fit(T_gt, abs_tol=1e-14)
    npt.assert_allclose(X.tensor, T_gt, rtol=1e-5, atol=1e-8)


def test_fit_L1L():
    np.random.seed(202505)
    X = BTD([22, 10, 21], 2, 2, block_mode="L1L")

    dims = X.dims
    R = X.rank
    L = X.L
    Lsum = np.array(L).sum()
    A0 = np.random.randn(dims[0], Lsum)
    B0 = np.random.randn(dims[1], R)
    C0 = np.random.randn(dims[2], Lsum)
    theta = X.get_constraint_matrix()
    T_gt = factors_to_tensor(A0, B0, C0, theta, block_mode="L1L")

    X.fit(T_gt, abs_tol=1e-14)
    npt.assert_allclose(X.tensor, T_gt, rtol=1e-5, atol=1e-8)


if __name__ == "__main__":
    pytest.main()
