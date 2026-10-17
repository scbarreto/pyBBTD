import warnings

import numpy as np
import numpy.testing as npt
import pytest
from _helpers import add_noise, bbtd_terms, relative_error, term_recovery_error

from pybbtd.bbtd import BBTD, _constraint_matrices, factors_to_tensor
from pybbtd.btd import spatial_maps
from pybbtd.solvers.bbtd_als import fit_bbtd_als, init_bbtd_factors
from pybbtd.solvers.bbtd_cov_admm import fit_bbtd_cov_admm, init_bbtd_cov_factors


def covariance_ground_truth(dims=(10, 12, 4, 4), R=2, L1=2, L2=2):
    """Non-negative spatial factors and D = C*, as in covariance imaging."""
    model = BBTD(dims=dims, R=R, L1=L1, L2=L2)
    A0 = np.random.rand(dims[0], L1 * R)
    B0 = np.random.rand(dims[1], L1 * R)
    C0 = np.random.randn(dims[2], L2 * R) + 1j * np.random.randn(dims[2], L2 * R)
    phi, psi = model.get_constraint_matrices()
    T0 = factors_to_tensor(A0, B0, C0, C0.conj(), phi, psi)
    return model, T0, (A0, B0, C0, C0.conj())


# ──────────────────────────────────────────────────────────────────────────────
# BBTD model
# ──────────────────────────────────────────────────────────────────────────────


def test_init():
    model = BBTD(dims=(10, 12, 8, 6), R=2, L1=3, L2=2)
    assert model.dims == (10, 12, 8, 6)
    assert (model.rank, model.L1, model.L2) == (2, 3, 2)
    assert model.factors is None
    assert model.tensor is None
    assert model.fit_error is None
    assert repr(model) == "BBTD(dims=(10, 12, 8, 6), R=2, L1=3, L2=2)"


@pytest.mark.parametrize(
    "dims", ["bad", (10, 10, 10), (10, 10, 0, 5), (10, 10, 2.0, 5)]
)
def test_invalid_dims(dims):
    with pytest.raises(ValueError, match="list or tuple of four positive integers"):
        BBTD(dims=dims, R=2, L1=2, L2=2)


@pytest.mark.parametrize(
    ("R", "L1", "L2", "name"),
    [(-1, 2, 2, "R"), (2, 0, 2, "L1"), (2, 2, 2.5, "L2"), (2, [2], 2, "L1")],
)
def test_invalid_ranks(R, L1, L2, name):
    with pytest.raises(ValueError, match=f"{name} should be a positive integer"):
        BBTD(dims=(10, 10, 8, 6), R=R, L1=L1, L2=L2)


def test_check_uniqueness():
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        assert BBTD((20, 25, 4, 4), R=3, L1=3, L2=2).check_uniqueness() is True
    with pytest.warns(UserWarning, match="Cannot guarantee uniqueness"):
        BBTD((3, 3, 3, 3), R=5, L1=3, L2=3)


def test_constraint_matrices_structure():
    R, L1, L2 = 3, 2, 4
    phi, psi = BBTD((10, 10, 8, 8), R, L1, L2).get_constraint_matrices()
    # phi repeats each column of A and B L2 times
    npt.assert_array_equal(phi, np.kron(np.eye(L1 * R), np.ones((1, L2))))
    # psi tiles each L2-block of C and D L1 times
    npt.assert_array_equal(psi, np.kron(np.eye(R), np.tile(np.eye(L2), (1, L1))))
    npt.assert_array_equal(phi, _constraint_matrices(R, L1, L2)[0])


def test_factors_to_tensor_matches_definition():
    dims, R, L1, L2 = (6, 5, 4, 3), 2, 3, 2
    A = np.random.randn(dims[0], L1 * R)
    B = np.random.randn(dims[1], L1 * R)
    C = np.random.randn(dims[2], L2 * R)
    D = np.random.randn(dims[3], L2 * R)
    phi, psi = _constraint_matrices(R, L1, L2)

    T = factors_to_tensor(A, B, C, D, phi, psi)

    npt.assert_allclose(T, sum(bbtd_terms(A, B, C, D, L1, L2)), atol=1e-12)


def test_fit_invalid_algorithm():
    model = BBTD(dims=(10, 10, 8, 6), R=2, L1=2, L2=2)
    with pytest.raises(ValueError, match="Unknown algorithm 'INVALID'"):
        model.fit(np.random.randn(10, 10, 8, 6), algorithm="INVALID")


@pytest.mark.filterwarnings("ignore:Reached max_iter")
@pytest.mark.parametrize("algorithm", ["ALS", "ADMM"])
def test_fit_stores_consistent_results(algorithm):
    model, T, _ = covariance_ground_truth()
    assert model.fit(T, algorithm=algorithm, max_iter=3) is model
    assert len(model.factors) == 4
    phi, psi = model.get_constraint_matrices()
    npt.assert_allclose(model.tensor, factors_to_tensor(*model.factors, phi, psi))


# ──────────────────────────────────────────────────────────────────────────────
# Unconstrained ALS solver
# ──────────────────────────────────────────────────────────────────────────────


@pytest.mark.parametrize("strategy", ["random", "svd"])
def test_als_init_shapes(strategy):
    dims, R, L1, L2 = (10, 12, 8, 6), 2, 3, 2
    model = BBTD(dims=dims, R=R, L1=L1, L2=L2)
    A, B, C, D = init_bbtd_factors(model, strategy=strategy, T=np.random.randn(*dims))
    assert A.shape == (10, 6)
    assert B.shape == (12, 6)
    assert C.shape == (8, 4)
    assert D.shape == (6, 4)


def test_als_init_errors():
    model = BBTD(dims=(10, 12, 8, 6), R=2, L1=3, L2=2)
    with pytest.raises(ValueError, match="SVD init requires input data T"):
        init_bbtd_factors(model, strategy="svd")
    with pytest.raises(ValueError, match="Unknown initialization 'bad'"):
        init_bbtd_factors(model, strategy="bad")


@pytest.mark.parametrize("solver", [fit_bbtd_als, fit_bbtd_cov_admm])
def test_solvers_reject_invalid_inputs(solver):
    model = BBTD(dims=(10, 10, 6, 6), R=2, L1=2, L2=2)
    with pytest.raises(TypeError, match="must be an instance of the BBTD class"):
        solver("not_a_model", np.random.randn(10, 10, 6, 6))
    with pytest.raises(TypeError, match="must be a numpy array"):
        solver(model, None)
    with pytest.raises(ValueError, match="do not match"):
        solver(model, np.random.randn(5, 5, 4, 3))


def test_als_fit_error_never_increases():
    # every ALS step solves its least-squares problem exactly
    model = BBTD(dims=(8, 7, 6, 5), R=2, L1=2, L2=2)
    with pytest.warns(RuntimeWarning, match="max_iter=30"):
        _, fit_error = fit_bbtd_als(
            model, np.random.randn(8, 7, 6, 5), max_iter=30, rel_tol=0
        )
    assert len(fit_error) == 31
    assert np.all(np.diff(fit_error) <= 1e-10 * fit_error[:-1])


@pytest.mark.parametrize("seed", [0, 1, 2])
@pytest.mark.parametrize("init", ["random", "svd"])
def test_als_recovers_terms(init, seed):
    np.random.seed(seed)
    model = BBTD((10, 9, 6, 5), R=2, L1=2, L2=2)
    A0, B0, C0, D0 = init_bbtd_factors(model)
    phi, psi = model.get_constraint_matrices()
    T = factors_to_tensor(A0, B0, C0, D0, phi, psi)

    model.fit(T, init=init, max_iter=3000, rel_tol=1e-12, abs_tol=1e-12)

    assert model.fit_error[-1] < 1e-10
    estimated = bbtd_terms(*model.factors, 2, 2)
    assert term_recovery_error(estimated, bbtd_terms(A0, B0, C0, D0, 2, 2)) < 1e-9


def test_als_noisy_data_fit_is_closer_to_truth_than_data():
    model = BBTD((10, 9, 6, 5), R=2, L1=2, L2=2)
    A0, B0, C0, D0 = init_bbtd_factors(model)
    phi, psi = model.get_constraint_matrices()
    T0 = factors_to_tensor(A0, B0, C0, D0, phi, psi)
    noise_level = 1e-2

    model.fit(
        add_noise(T0, noise_level), init="svd", max_iter=2000, rel_tol=1e-12, abs_tol=0
    )

    assert relative_error(model.tensor, T0) < noise_level
    estimated = bbtd_terms(*model.factors, 2, 2)
    assert (
        term_recovery_error(estimated, bbtd_terms(A0, B0, C0, D0, 2, 2))
        < 5 * noise_level
    )


# ──────────────────────────────────────────────────────────────────────────────
# Covariance ADMM solver
# ──────────────────────────────────────────────────────────────────────────────


@pytest.mark.parametrize("strategy", ["random", "svd", "kmeans"])
def test_admm_init_shapes_and_non_negativity(strategy):
    model, T, _ = covariance_ground_truth(dims=(10, 12, 6, 6), R=2, L1=2, L2=3)
    A, B, C = init_bbtd_cov_factors(model, strategy=strategy, T=T)
    assert A.shape == (10, 4)
    assert B.shape == (12, 4)
    assert C.shape == (6, 6)
    assert np.isrealobj(A)
    assert np.isrealobj(B)
    assert A.min() >= 0
    assert B.min() >= 0


@pytest.mark.parametrize("strategy", ["svd", "kmeans"])
def test_admm_init_requires_T(strategy):
    model = BBTD(dims=(10, 12, 6, 6), R=2, L1=2, L2=3)
    with pytest.raises(ValueError, match="requires input data T"):
        init_bbtd_cov_factors(model, strategy=strategy)


def test_admm_init_unknown_strategy():
    model = BBTD(dims=(10, 12, 6, 6), R=2, L1=2, L2=3)
    with pytest.raises(ValueError, match="Unknown initialization 'bad'"):
        init_bbtd_cov_factors(model, strategy="bad")


@pytest.mark.parametrize("seed", [0, 1, 2])
@pytest.mark.parametrize("init", ["random", "svd", "kmeans"])
def test_admm_recovers_terms_and_keeps_constraints(init, seed):
    np.random.seed(seed)
    model, T, truth = covariance_ground_truth()

    model.fit(
        T, algorithm="ADMM", init=init, max_iter=500, rel_tol=1e-10, abs_tol=1e-12
    )
    A, B, C, D = model.factors

    assert relative_error(model.tensor, T) < 1e-10
    assert (
        term_recovery_error(bbtd_terms(A, B, C, D, 2, 2), bbtd_terms(*truth, 2, 2))
        < 1e-9
    )
    npt.assert_array_equal(D, C.conj())
    assert spatial_maps(A, B, 2).min() >= 0


def test_admm_noisy_data_fit_is_closer_to_truth_than_data():
    model, T0, truth = covariance_ground_truth()
    noise_level = 1e-2

    model.fit(
        add_noise(T0, noise_level),
        algorithm="ADMM",
        init="kmeans",
        max_iter=500,
        rel_tol=1e-10,
        abs_tol=0,
    )

    assert relative_error(model.tensor, T0) < noise_level
    estimated = bbtd_terms(*model.factors, 2, 2)
    assert term_recovery_error(estimated, bbtd_terms(*truth, 2, 2)) < 5 * noise_level


def test_admm_warns_when_max_iter_is_reached():
    model, T, _ = covariance_ground_truth()
    with pytest.warns(RuntimeWarning, match="max_iter=2"):
        _, fit_error = fit_bbtd_cov_admm(model, T, max_iter=2, rel_tol=0, abs_tol=0)
    assert len(fit_error) == 2
