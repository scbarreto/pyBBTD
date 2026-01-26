import pytest
import numpy as np
import numpy.testing as npt
from pybbtd.bbtd import BBTD, factors_to_tensor
from pybbtd.solvers.bbtd_vanilla_als import BBTD_ALS, init_BBTD_factors
from pybbtd.solvers.bbtd_cov_admm import BBTD_COV_ADMM, init_BBTD_cov_factors


# ──────────────────────────────────────────────────────────────────────────────
# Section 1: BBTD class
# ──────────────────────────────────────────────────────────────────────────────


def test_valid_initialization():
    model = BBTD(dims=(10, 12, 8, 6), R=2, L1=3, L2=2)
    npt.assert_equal(model.dims, (10, 12, 8, 6))
    npt.assert_equal(model.rank, 2)
    npt.assert_equal(model.L1, 3)
    npt.assert_equal(model.L2, 2)
    assert model.factors is None
    assert model.tensor is None
    assert model.fit_error is None


def test_invalid_dims_not_list():
    with pytest.raises(ValueError, match="list or tuple of four positive integers"):
        BBTD(dims="bad", R=2, L1=2, L2=2)


def test_invalid_dims_wrong_length():
    with pytest.raises(ValueError, match="list or tuple of four positive integers"):
        BBTD(dims=(10, 10, 10), R=2, L1=2, L2=2)


def test_invalid_dims_non_positive():
    with pytest.raises(ValueError, match="list or tuple of four positive integers"):
        BBTD(dims=(10, 10, 0, 5), R=2, L1=2, L2=2)


def test_invalid_R():
    with pytest.raises(ValueError, match="R should be a positive integer"):
        BBTD(dims=(10, 10, 8, 6), R=-1, L1=2, L2=2)


def test_invalid_L1():
    with pytest.raises(ValueError, match="L1 should be a positive integer"):
        BBTD(dims=(10, 10, 8, 6), R=2, L1=0, L2=2)


def test_invalid_L2():
    with pytest.raises(ValueError, match="L2 should be a positive integer"):
        BBTD(dims=(10, 10, 8, 6), R=2, L1=2, L2=2.5)


def test_constraint_matrices_shapes():
    R, L1, L2 = 3, 2, 4
    model = BBTD(dims=(10, 10, 8, 8), R=R, L1=L1, L2=L2)
    phi, psi = model.get_constraint_matrices()
    assert phi.shape == (L1 * R, L1 * L2 * R)
    assert psi.shape == (L2 * R, L1 * L2 * R)


def test_factors_to_tensor():
    np.random.seed(42)
    dims = (10, 12, 8, 6)
    R, L1, L2 = 2, 3, 2
    model = BBTD(dims=dims, R=R, L1=L1, L2=L2)
    A = np.random.randn(dims[0], L1 * R)
    B = np.random.randn(dims[1], L1 * R)
    C = np.random.randn(dims[2], L2 * R)
    D = np.random.randn(dims[3], L2 * R)
    phi, psi = model.get_constraint_matrices()
    T = factors_to_tensor(A, B, C, D, phi, psi)
    assert T.shape == dims


def test_fit_invalid_algorithm():
    model = BBTD(dims=(10, 10, 8, 6), R=2, L1=2, L2=2)
    T = np.random.randn(10, 10, 8, 6)
    with pytest.raises(NotImplementedError, match="INVALID"):
        model.fit(T, algorithm="INVALID")


# ──────────────────────────────────────────────────────────────────────────────
# Section 2: Vanilla ALS solver
# ──────────────────────────────────────────────────────────────────────────────


def test_als_init_random_shapes():
    np.random.seed(0)
    dims = (10, 12, 8, 6)
    R, L1, L2 = 2, 3, 2
    model = BBTD(dims=dims, R=R, L1=L1, L2=L2)
    A, B, C, D = init_BBTD_factors(model, strat="random")
    assert A.shape == (dims[0], L1 * R)
    assert B.shape == (dims[1], L1 * R)
    assert C.shape == (dims[2], L2 * R)
    assert D.shape == (dims[3], L2 * R)


def test_als_init_svd_shapes():
    np.random.seed(0)
    dims = (10, 12, 8, 6)
    R, L1, L2 = 2, 3, 2
    model = BBTD(dims=dims, R=R, L1=L1, L2=L2)
    T = np.random.randn(*dims)
    A, B, C, D = init_BBTD_factors(model, strat="svd", T=T)
    assert A.shape == (dims[0], L1 * R)
    assert B.shape == (dims[1], L1 * R)
    assert C.shape == (dims[2], L2 * R)
    assert D.shape == (dims[3], L2 * R)


def test_als_init_svd_requires_T():
    model = BBTD(dims=(10, 12, 8, 6), R=2, L1=3, L2=2)
    with pytest.raises(ValueError, match="SVD init requires input data T"):
        init_BBTD_factors(model, strat="svd", T=None)


def test_als_init_invalid_strategy():
    model = BBTD(dims=(10, 12, 8, 6), R=2, L1=3, L2=2)
    with pytest.raises(ValueError, match="not implemented"):
        init_BBTD_factors(model, strat="bad")


def test_als_invalid_model_type():
    T = np.random.randn(10, 10, 8, 6)
    with pytest.raises(TypeError, match="must be an instance of the BBTD class"):
        BBTD_ALS("not_a_model", T)


def test_als_invalid_tensor_type():
    model = BBTD(dims=(10, 10, 8, 6), R=2, L1=2, L2=2)
    with pytest.raises(TypeError, match="must be a numpy array"):
        BBTD_ALS(model, None)


def test_als_mismatched_dims():
    model = BBTD(dims=(10, 10, 8, 6), R=2, L1=2, L2=2)
    T = np.random.randn(5, 5, 4, 3)
    with pytest.raises(ValueError, match="do not match"):
        BBTD_ALS(model, T)


def test_als_fit_random_init():
    np.random.seed(202505)
    dims = (15, 14, 10, 8)
    R, L1, L2 = 2, 2, 2
    model = BBTD(dims=dims, R=R, L1=L1, L2=L2)

    # Generate ground-truth tensor from random factors
    A0, B0, C0, D0 = init_BBTD_factors(model, strat="random")
    phi, psi = model.get_constraint_matrices()
    T_gt = factors_to_tensor(A0, B0, C0, D0, phi, psi)

    # Fit with random init
    factors, fit_error = BBTD_ALS(
        model, T_gt, init="random", max_iter=2000, abs_tol=1e-14, rel_tol=1e-6
    )
    assert fit_error[-1] < 1e-3


def test_als_fit_svd_init():
    np.random.seed(202505)
    dims = (15, 14, 10, 8)
    R, L1, L2 = 2, 2, 2
    model = BBTD(dims=dims, R=R, L1=L1, L2=L2)

    # Generate ground-truth tensor from random factors
    A0, B0, C0, D0 = init_BBTD_factors(model, strat="random")
    phi, psi = model.get_constraint_matrices()
    T_gt = factors_to_tensor(A0, B0, C0, D0, phi, psi)

    # Fit with SVD init
    factors, fit_error = BBTD_ALS(
        model, T_gt, init="svd", max_iter=2000, abs_tol=1e-14, rel_tol=1e-6
    )
    assert fit_error[-1] < 1e-3


# ──────────────────────────────────────────────────────────────────────────────
# Section 3: Constrained ADMM solver
# ──────────────────────────────────────────────────────────────────────────────


def _make_cov_ground_truth(dims, R, L1, L2, seed=42):
    """Helper to build a noiseless covariance-structured ground-truth tensor."""
    np.random.seed(seed)
    model = BBTD(dims=dims, R=R, L1=L1, L2=L2)

    # Non-negative spatial factors
    A0 = np.random.rand(dims[0], L1 * R)
    B0 = np.random.rand(dims[1], L1 * R)
    # Complex spectral factor with D = C*
    C0 = np.random.randn(dims[2], L2 * R) + 1j * np.random.randn(dims[2], L2 * R)
    D0 = C0.conj()

    phi, psi = model.get_constraint_matrices()
    T_gt = factors_to_tensor(A0, B0, C0, D0, phi, psi)
    return model, T_gt, (A0, B0, C0, D0)


def test_admm_init_random_shapes():
    np.random.seed(0)
    dims = (10, 12, 6, 6)
    R, L1, L2 = 2, 2, 3
    model = BBTD(dims=dims, R=R, L1=L1, L2=L2)
    A, B, C = init_BBTD_cov_factors(model, strat="random")
    assert A.shape == (dims[0], L1 * R)
    assert B.shape == (dims[1], L1 * R)
    assert C.shape == (dims[2], L2 * R)
    # A and B should be non-negative (random uses rand, not randn)
    assert np.all(A >= 0)
    assert np.all(B >= 0)


def test_admm_init_svd_shapes():
    np.random.seed(0)
    dims = (10, 12, 6, 6)
    R, L1, L2 = 2, 2, 3
    model = BBTD(dims=dims, R=R, L1=L1, L2=L2)
    T = np.random.randn(*dims)
    A, B, C = init_BBTD_cov_factors(model, strat="svd", T=T)
    assert A.shape == (dims[0], L1 * R)
    assert B.shape == (dims[1], L1 * R)
    assert C.shape == (dims[2], L2 * R)
    # A and B should be non-negative (SVD init uses np.abs)
    assert np.all(A >= 0)
    assert np.all(B >= 0)


def test_admm_init_kmeans_shapes():
    np.random.seed(0)
    dims = (10, 12, 6, 6)
    R, L1, L2 = 2, 2, 3
    model = BBTD(dims=dims, R=R, L1=L1, L2=L2)
    # kmeans init needs a tensor with covariance-like structure
    _, T_gt, _ = _make_cov_ground_truth(dims, R, L1, L2, seed=0)
    A, B, C = init_BBTD_cov_factors(model, strat="kmeans", T=T_gt)
    assert A.shape == (dims[0], L1 * R)
    assert B.shape == (dims[1], L1 * R)
    assert C.shape == (dims[2], L2 * R)
    # A and B should be non-negative (NMF produces non-negative factors)
    assert np.all(A >= 0)
    assert np.all(B >= 0)


def test_admm_init_invalid_strategy():
    model = BBTD(dims=(10, 12, 6, 6), R=2, L1=2, L2=3)
    with pytest.raises(ValueError, match="not implemented"):
        init_BBTD_cov_factors(model, strat="bad")


def test_admm_invalid_model_type():
    T = np.random.randn(10, 10, 6, 6)
    with pytest.raises(TypeError, match="must be an instance of the BBTD class"):
        BBTD_COV_ADMM("not_a_model", T)


def test_admm_fit_random_init():
    dims = (10, 12, 6, 6)
    R, L1, L2 = 2, 2, 2
    model, T_gt, (A0, B0, C0, D0) = _make_cov_ground_truth(
        dims, R, L1, L2, seed=202505
    )

    factors, fit_error = BBTD_COV_ADMM(
        model,
        T_gt,
        init="random",
        max_iter=500,
        gamma=1.0,
        rho=1.0,
        inner_admm=50,
        rel_tol=1e-8,
        abs_tol=1e-12,
    )
    assert fit_error[-1] < 1e-1

    # Check D ≈ C* constraint
    C_est, D_est = factors[2], factors[3]
    npt.assert_allclose(D_est, C_est.conj(), atol=1e-3)


def test_admm_fit_kmeans_init():
    dims = (10, 12, 6, 6)
    R, L1, L2 = 2, 2, 2
    model, T_gt, (A0, B0, C0, D0) = _make_cov_ground_truth(
        dims, R, L1, L2, seed=202505
    )

    factors, fit_error = BBTD_COV_ADMM(
        model,
        T_gt,
        init="kmeans",
        max_iter=500,
        gamma=1.0,
        rho=1.0,
        inner_admm=50,
        rel_tol=1e-8,
        abs_tol=1e-12,
    )
    assert fit_error[-1] < 1e-1

    # Check D ≈ C* constraint
    C_est, D_est = factors[2], factors[3]
    npt.assert_allclose(D_est, C_est.conj(), atol=1e-3)


def test_admm_nonneg_spatial_maps():
    dims = (10, 12, 6, 6)
    R, L1, L2 = 2, 2, 2
    model, T_gt, _ = _make_cov_ground_truth(dims, R, L1, L2, seed=202505)

    factors, _ = BBTD_COV_ADMM(
        model,
        T_gt,
        init="random",
        max_iter=500,
        gamma=1.0,
        rho=1.0,
        inner_admm=50,
        rel_tol=1e-8,
        abs_tol=1e-12,
    )

    A_est, B_est = factors[0], factors[1]
    for r in range(R):
        S_r = A_est[:, r * L1 : (r + 1) * L1] @ B_est[:, r * L1 : (r + 1) * L1].T
        # Spatial maps should be approximately non-negative after projection
        assert np.all(S_r >= -1e-6), f"Spatial map S_{r} has negative entries"


if __name__ == "__main__":
    pytest.main()
