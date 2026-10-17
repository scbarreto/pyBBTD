import numpy as np
import numpy.testing as npt
import pytest
from scipy.optimize import nnls

from pybbtd import BTD
from pybbtd._linalg import hermitian_to_real, project_psd, real_to_hermitian
from pybbtd._utils import check_dims, check_positive_int, check_solver_inputs
from pybbtd.btd import _constraint_matrix, factors_to_tensor
from pybbtd.solvers._admm import admm_update, project_nonnegative
from pybbtd.solvers._kmeans import cluster_maps, match_norm, nmf_factors


def random_hermitian(*shape):
    Z = np.random.randn(*shape) + 1j * np.random.randn(*shape)
    return Z + np.swapaxes(Z, -1, -2).conj()


# ──────────────────────────────────────────────────────────────────────────────
# Input checks
# ──────────────────────────────────────────────────────────────────────────────


@pytest.mark.parametrize("value", [1, 7, np.int64(3)])
def test_check_positive_int_accepts(value):
    assert check_positive_int(value, "R") == value
    assert type(check_positive_int(value, "R")) is int


@pytest.mark.parametrize("value", [0, -1, 2.5, True, "3", None])
def test_check_positive_int_rejects(value):
    with pytest.raises(ValueError, match="R should be a positive integer"):
        check_positive_int(value, "R")


def test_check_dims():
    assert check_dims([3, np.int64(4)], 2) == (3, 4)
    for dims in [(3,), (3, 0), (3, 4.0), (3, False), "34", 34]:
        with pytest.raises(ValueError, match="two positive integers"):
            check_dims(dims, 2)


def test_check_solver_inputs():
    model = BTD((4, 3, 2), R=1, L=1)
    check_solver_inputs(model, BTD, np.zeros((4, 3, 2)))
    with pytest.raises(TypeError, match="instance of the BTD class"):
        check_solver_inputs(object(), BTD, np.zeros((4, 3, 2)))
    with pytest.raises(TypeError, match="numpy array"):
        check_solver_inputs(model, BTD, [[0.0]])
    with pytest.raises(ValueError, match=r"\(4, 3, 3\)\) do not match"):
        check_solver_inputs(model, BTD, np.zeros((4, 3, 3)))


# ──────────────────────────────────────────────────────────────────────────────
# Linear algebra
# ──────────────────────────────────────────────────────────────────────────────


def test_project_psd_is_nearest_psd_matrix():
    M = random_hermitian(4, 4)
    P = project_psd(M)
    npt.assert_allclose(P, P.conj().T, atol=1e-12)
    assert np.linalg.eigvalsh(P).min() >= -1e-12
    npt.assert_allclose(project_psd(P), P, atol=1e-12)
    # no PSD matrix is closer to M
    for _ in range(200):
        G = np.random.randn(4, 4) + 1j * np.random.randn(4, 4)
        Q = G @ G.conj().T
        assert np.linalg.norm(M - P) <= np.linalg.norm(M - Q) + 1e-12


def test_project_psd_uses_hermitian_part():
    M = np.array([[1.0, 2.0], [0.0, 1.0]])
    npt.assert_allclose(project_psd(M), project_psd(0.5 * (M + M.T)))


def test_hermitian_to_real_layout():
    K = 3
    X = random_hermitian(2, 5, K, K)
    packed = hermitian_to_real(X)
    assert packed.shape == (2, 5, K * K)
    for i in range(K):
        for j in range(K):
            expected = X[..., j, i].imag if i > j else X[..., i, j].real
            npt.assert_array_equal(packed[..., i * K + j], expected)
    assert hermitian_to_real(X, dtype="float32").dtype == np.float32


def test_real_to_hermitian_inverts_packing():
    X = random_hermitian(4, 6, 3, 3)
    npt.assert_allclose(real_to_hermitian(hermitian_to_real(X)), X, atol=1e-14)


# ──────────────────────────────────────────────────────────────────────────────
# ADMM factor update
# ──────────────────────────────────────────────────────────────────────────────


def test_admm_update_without_constraint_is_least_squares():
    Y = np.random.randn(5, 40)
    M = np.random.randn(3, 40)
    X0 = np.zeros((5, 3))
    Xt, _ = admm_update(Y, M, X0, X0, rho=1.0, project=lambda X: X, max_iter=200, tol=0)
    X_ls = np.linalg.lstsq(M.T, Y.T, rcond=None)[0].T
    npt.assert_allclose(Xt, X_ls, atol=1e-8)


def test_admm_update_with_non_negativity_matches_nnls():
    Y = np.random.randn(6, 40)
    M = np.random.randn(4, 40)
    X0 = np.zeros((6, 4))
    Xt, _ = admm_update(Y, M, X0, X0, 1.0, project_nonnegative, max_iter=5000, tol=0)
    X_nnls = np.array([nnls(M.T, y)[0] for y in Y])
    npt.assert_allclose(Xt, X_nnls, atol=1e-6)


def test_admm_update_real_factor_of_complex_data_matches_nnls():
    # with complex Y and M, a real non-negative X solves NNLS on the stacked
    # real and imaginary parts
    Y = np.random.randn(6, 40) + 1j * np.random.randn(6, 40)
    M = np.random.randn(4, 40) + 1j * np.random.randn(4, 40)
    X0 = np.zeros((6, 4))
    Xt, _ = admm_update(Y, M, X0, X0, 1.0, project_nonnegative, 5000, 0, real=True)
    M_stacked = np.hstack([M.real, M.imag]).T
    X_nnls = np.array([nnls(M_stacked, np.concatenate([y.real, y.imag]))[0] for y in Y])
    assert np.isrealobj(Xt)
    npt.assert_allclose(Xt, X_nnls, atol=1e-6)


# ──────────────────────────────────────────────────────────────────────────────
# K-means initialization helpers
# ──────────────────────────────────────────────────────────────────────────────


def test_cluster_maps():
    labels = np.array([0, 1, 1, 0, 2, 2])
    maps = cluster_maps(labels, 3, (2, 3), random_weights=False)
    npt.assert_array_equal(
        maps, [[[1, 0, 0], [1, 0, 0]], [[0, 1, 1], [0, 0, 0]], [[0, 0, 0], [0, 1, 1]]]
    )

    weighted = cluster_maps(labels, 3, (2, 3), random_weights=True)
    npt.assert_array_equal(weighted > 0, maps > 0)
    assert weighted.max() < 1


def test_nmf_factors_shapes_and_non_negativity():
    from sklearn.decomposition import NMF

    maps = np.random.rand(3, 8, 6)
    A, B = nmf_factors(
        maps, NMF(n_components=2, init="nndsvda", random_state=0, max_iter=500)
    )
    assert A.shape == (8, 6)
    assert B.shape == (6, 6)
    assert A.min() >= 0
    assert B.min() >= 0


def test_match_norm_rescales_tensor_to_data_norm():
    A, B, C = np.random.rand(6, 4), np.random.rand(5, 4), np.random.rand(3, 2)
    theta = _constraint_matrix(2, 2)
    T = 7.0 * np.random.rand(6, 5, 3)
    A, B = match_norm(A, B, T, factors_to_tensor(A, B, C, theta))
    npt.assert_allclose(
        np.linalg.norm(factors_to_tensor(A, B, C, theta)), np.linalg.norm(T)
    )
