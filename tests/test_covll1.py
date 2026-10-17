import itertools

import numpy as np
import numpy.testing as npt
import pytest
from _helpers import add_noise, btd_terms, relative_error, term_recovery_error

from pybbtd.btd import BTD, factors_to_tensor
from pybbtd.covll1 import (
    CovLL1,
    generate_covll1_factors,
    is_valid_covariance,
    validate_covariance_tensor,
)
from pybbtd.solvers.covll1_admm import fit_covll1_admm, init_covll1_factors
from pybbtd.stokes import Stokes


def disjoint_spatial_factors(n_rows, n_cols, R, L):
    """Spatial factors whose maps do not overlap: term r lives in its own columns."""
    A = 0.5 + 0.5 * np.random.rand(n_rows, L * R)
    B = np.zeros((n_cols, L * R))
    width = n_cols // R
    for r in range(R):
        B[r * width : (r + 1) * width, r * L : (r + 1) * L] = (
            0.5 + 0.5 * np.random.rand(width, L)
        )
    return A, B


def assert_columns_match_up_to_scale(estimate, truth, atol):
    """Columns agree in direction for the best column order (cosine similarity 1)."""
    best = min(
        max(
            1
            - abs(np.vdot(estimate[:, i], truth[:, j]))
            / (np.linalg.norm(estimate[:, i]) * np.linalg.norm(truth[:, j]))
            for i, j in enumerate(order)
        )
        for order in itertools.permutations(range(truth.shape[1]))
    )
    assert best < atol


# ──────────────────────────────────────────────────────────────────────────────
# Cov-LL1 model
# ──────────────────────────────────────────────────────────────────────────────


def test_init():
    model = CovLL1((12, 11, 9), R=2, L1=3, L2=2)
    assert isinstance(model, BTD)
    assert model.dims == (12, 11, 9)
    assert (model.rank, model.L1, model.L2) == (2, 3, 2)
    npt.assert_array_equal(model.L, [3, 3])
    assert repr(model) == "CovLL1(dims=(12, 11, 9), R=2, L1=3, L2=2, block_mode='LL1')"


def test_get_constraint_matrix():
    npt.assert_array_equal(
        CovLL1((12, 11, 9), R=2, L1=3, L2=2).get_constraint_matrix(),
        [[1, 1, 1, 0, 0, 0], [0, 0, 0, 1, 1, 1]],
    )


@pytest.mark.filterwarnings("ignore:Cannot guarantee uniqueness")
@pytest.mark.parametrize(
    ("dims", "L1", "L2", "message"),
    [
        ((10, 10, 8), 2, 2, "perfect square"),
        ((10, 10, 9), 11, 2, "L1 cannot exceed"),
        ((10, 10, 9), 2, 4, "L2 cannot exceed"),
        ((10, 10, 9), 2, 0, "L2 must be a positive integer"),
    ],
)
def test_invalid_parameters(dims, L1, L2, message):
    with pytest.raises(ValueError, match=message):
        CovLL1(dims, R=2, L1=L1, L2=L2)


def test_generate_factors():
    K, L2 = 3, 2
    A, B, C = generate_covll1_factors((8, 7, K**2), R=4, L1=2, L2=L2)
    assert A.shape == (8, 8)
    assert B.shape == (7, 8)
    assert C.shape == (K**2, 4)
    assert A.min() >= 0
    assert B.min() >= 0
    for column in C.T:
        Sigma = column.reshape(K, K)
        assert is_valid_covariance(Sigma)
        assert np.linalg.matrix_rank(Sigma) == L2


def test_generate_tensor_is_valid():
    model = CovLL1((6, 5, 9), R=2, L1=2, L2=2)
    factors, T = model.generate_covll1_tensor()
    assert model.factors is factors
    npt.assert_allclose(T, factors_to_tensor(*factors, model.get_constraint_matrix()))
    assert validate_covariance_tensor(T) is True


def test_validate_covariance_tensor_warns_on_invalid_pixels():
    with pytest.warns(UserWarning, match="100.00% of pixels"):
        assert validate_covariance_tensor(np.random.randn(4, 5, 9)) is False


@pytest.mark.parametrize(
    ("Sigma", "valid"),
    [
        (np.eye(3), True),
        (np.array([[2, 1j], [-1j, 2]]), True),
        (np.zeros((2, 2)), True),
        (np.ones((2, 3)), False),
        (np.array([[1, 2], [0, 1]]), False),
        (np.array([[1, 2], [2, 1]]), False),
        (np.diag([1, -1e-12]), True),
    ],
)
def test_is_valid_covariance(Sigma, valid):
    assert is_valid_covariance(Sigma) is valid


def test_fit_invalid_algorithm():
    with pytest.raises(ValueError, match="Unknown algorithm 'ALS'"):
        CovLL1((6, 5, 9), R=2, L1=2, L2=2).fit(np.zeros((6, 5, 9)), algorithm="ALS")


def test_solver_rejects_invalid_inputs():
    model = CovLL1((6, 5, 9), R=2, L1=2, L2=2)
    with pytest.raises(TypeError, match="must be an instance of the CovLL1 class"):
        fit_covll1_admm(Stokes((6, 5), R=2, L=2), np.zeros((6, 5, 4)))
    with pytest.raises(TypeError, match="must be a numpy array"):
        fit_covll1_admm(model, None)
    with pytest.raises(ValueError, match="do not match"):
        fit_covll1_admm(model, np.zeros((6, 5, 4)))
    with pytest.raises(ValueError, match="Unknown initialization 'svd'"):
        fit_covll1_admm(model, np.zeros((6, 5, 9)), init="svd")


# ──────────────────────────────────────────────────────────────────────────────
# Initialization
# ──────────────────────────────────────────────────────────────────────────────


def test_init_unknown_strategy_and_missing_data():
    model = CovLL1((6, 5, 9), R=2, L1=2, L2=2)
    with pytest.raises(ValueError, match="Unknown initialization 'bad'"):
        init_covll1_factors(model, strategy="bad")
    with pytest.raises(ValueError, match="K-means init requires input data T"):
        init_covll1_factors(model, strategy="kmeans")


def test_kmeans_init_finds_covariances_of_separate_regions():
    # each pixel belongs to a single term, so each cluster center is a
    # multiple of one true covariance matrix
    model = CovLL1((10, 12, 9), R=3, L1=2, L2=2)
    A, B = disjoint_spatial_factors(10, 12, 3, 2)
    _, _, C = generate_covll1_factors(model.dims, 3, 2, 2)
    T = factors_to_tensor(A, B, C, model.get_constraint_matrix())

    A0, B0, C0 = init_covll1_factors(model, strategy="kmeans", T=T)

    assert A0.min() >= 0
    assert B0.min() >= 0
    assert_columns_match_up_to_scale(C0, C, atol=1e-6)


# ──────────────────────────────────────────────────────────────────────────────
# AO-ADMM solver
# ──────────────────────────────────────────────────────────────────────────────


@pytest.mark.filterwarnings("ignore:Reached max_iter")
@pytest.mark.parametrize("init", ["random", "kmeans"])
@pytest.mark.parametrize("max_admm", [1, 5])
def test_fit_keeps_constraints(init, max_admm):
    model = CovLL1((8, 9, 9), R=2, L1=2, L2=2)
    _, T = model.generate_covll1_tensor()

    model.fit(add_noise(T, 0.1), init=init, max_iter=50, max_admm=max_admm)
    A, B, C = model.factors

    assert np.isrealobj(A)
    assert np.isrealobj(B)
    assert A.min() >= 0
    assert B.min() >= 0
    for column in C.T:
        assert is_valid_covariance(column.reshape(3, 3), tol=1e-8)


@pytest.mark.filterwarnings("ignore:Reached max_iter")
@pytest.mark.parametrize("seed", range(6))
def test_fit_recovers_terms(seed):
    np.random.seed(seed)
    model = CovLL1((12, 12, 4), R=2, L1=2, L2=2)
    (A0, B0, C0), T = model.generate_covll1_tensor()

    model.fit(T, init="kmeans", max_iter=1500, rel_tol=0, abs_tol=0)

    assert relative_error(model.tensor, T) < 1e-4
    estimated = btd_terms(*model.factors, model.L)
    assert term_recovery_error(estimated, btd_terms(A0, B0, C0, model.L)) < 1e-3


def test_fit_noisy_data_is_closer_to_truth_than_data():
    np.random.seed(0)
    model = CovLL1((12, 12, 4), R=2, L1=2, L2=2)
    (A0, B0, C0), T0 = model.generate_covll1_tensor()
    noise_level = 1e-2

    model.fit(
        add_noise(T0, noise_level),
        init="kmeans",
        max_iter=1500,
        rel_tol=1e-12,
        abs_tol=0,
    )

    assert relative_error(model.tensor, T0) < noise_level
    estimated = btd_terms(*model.factors, model.L)
    assert (
        term_recovery_error(estimated, btd_terms(A0, B0, C0, model.L)) < 5 * noise_level
    )


def test_fit_error_is_normalized_squared_error():
    model = CovLL1((8, 9, 4), R=2, L1=2, L2=2)
    _, T = model.generate_covll1_tensor()
    with pytest.warns(RuntimeWarning, match="max_iter=3"):
        model.fit(T, max_iter=3, rel_tol=0, abs_tol=0)
    assert len(model.fit_error) == 4
    npt.assert_allclose(
        model.fit_error[-1],
        np.linalg.norm(T - model.tensor) ** 2 / np.linalg.norm(T) ** 2,
    )
