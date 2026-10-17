import logging
import warnings

import numpy as np
import numpy.testing as npt
import pytest
import tensorly as tly
from _helpers import add_noise, btd_terms, perturb, relative_error, term_recovery_error
from tensorly.tenalg import khatri_rao

import pybbtd.solvers.btd_als as btd_als
from pybbtd.btd import (
    BTD,
    _constraint_matrix,
    _validate_ranks,
    factors_to_tensor,
    spatial_maps,
)
from pybbtd.solvers.btd_als import fit_btd_als, init_btd_factors

# Shapes of the factors [A, B, C] in each block mode, for R terms with
# Lsum = sum(L) columns in the block factors.
FACTOR_COLUMNS = {
    "LL1": ("L", "L", "R"),
    "L1L": ("L", "R", "L"),
    "1LL": ("R", "L", "L"),
}


def random_factors(dims, R, Lsum, block_mode):
    columns = {"L": Lsum, "R": R}
    return [
        np.random.randn(n, columns[kind])
        for n, kind in zip(dims, FACTOR_COLUMNS[block_mode], strict=True)
    ]


def to_ll1(factors, block_mode):
    """Reorder model-layout factors into the LL1 layout used by the solver."""
    return [factors[axis] for axis in btd_als._LL1_AXES[block_mode]]


# ──────────────────────────────────────────────────────────────────────────────
# BTD model
# ──────────────────────────────────────────────────────────────────────────────


def test_init_with_int_L():
    model = BTD(dims=(15, 16, 10), R=3, L=2)
    assert model.dims == (15, 16, 10)
    assert model.rank == 3
    npt.assert_array_equal(model.L, [2, 2, 2])
    assert model.block_mode == "LL1"
    assert model.factors is None
    assert model.tensor is None
    assert model.fit_error is None


@pytest.mark.parametrize("L", [[2, 3, 4], (2, 3, 4), np.array([2, 3, 4])])
def test_init_with_sequence_L(L):
    model = BTD(dims=(20, 19, 18), R=3, L=L, block_mode="L1L")
    npt.assert_array_equal(model.L, [2, 3, 4])
    assert model.block_mode == "L1L"


def test_repr():
    assert repr(BTD((15, 16, 10), 3, [1, 2, 3])) == (
        "BTD(dims=(15, 16, 10), R=3, L=[1, 2, 3], block_mode='LL1')"
    )


@pytest.mark.parametrize(
    "dims", ["bad", (10, 10), (10, 0, 5), (10, 10.0, 5), (10, True, 5)]
)
def test_invalid_dims(dims):
    with pytest.raises(ValueError, match="list or tuple of three positive integers"):
        BTD(dims=dims, R=2, L=2)


@pytest.mark.parametrize("R", [-1, 0, 2.5, True])
def test_invalid_R(R):
    with pytest.raises(ValueError, match="R should be a positive integer"):
        BTD(dims=(10, 10, 10), R=R, L=2)


@pytest.mark.parametrize("L", [[2, 3], "invalid", 2.5])
def test_invalid_L_type_or_length(L):
    with pytest.raises(ValueError, match="single integer or a list/array of length R"):
        BTD(dims=(10, 10, 10), R=3, L=L)


@pytest.mark.parametrize("L", [0, -2, [2, 0, 1]])
def test_invalid_L_non_positive(L):
    with pytest.raises(
        ValueError, match="Each element in L should be greater than zero"
    ):
        BTD(dims=(10, 10, 10), R=3, L=L)


def test_invalid_block_mode():
    with pytest.raises(ValueError, match="Invalid mode"):
        BTD(dims=(10, 10, 10), R=3, L=2, block_mode="invalid")


def test_validate_ranks_broadcasts_int_L():
    R, L = _validate_ranks(3, 2)
    assert R == 3
    npt.assert_array_equal(L, [2, 2, 2])


def test_check_uniqueness_true_without_warning():
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        model = BTD(dims=(12, 11, 6), R=3, L=2)
    assert model.check_uniqueness() is True


def test_check_uniqueness_false_warns_at_user_line():
    with pytest.warns(UserWarning, match="Cannot guarantee uniqueness") as record:
        model = BTD(dims=(5, 5, 5), R=3, L=3)
    assert record[0].filename == __file__
    with pytest.warns(UserWarning):
        assert model.check_uniqueness() is False


@pytest.mark.parametrize(
    ("block_mode", "dims"),
    [("LL1", (8, 8, 3)), ("L1L", (8, 3, 8)), ("1LL", (3, 8, 8))],
)
def test_check_uniqueness_uses_rank_one_mode(block_mode, dims):
    # (8, 8, 3) with R=4, L=2 is unique in LL1 form; each mode permutes it
    assert BTD(dims, R=4, L=2, block_mode=block_mode).check_uniqueness() is True


def test_check_uniqueness_depends_on_block_mode():
    # the same shape is not covered once the rank-one mode moves
    with pytest.warns(UserWarning, match="Cannot guarantee uniqueness"):
        assert BTD((3, 8, 8), R=4, L=2, block_mode="LL1").check_uniqueness() is False


def test_check_uniqueness_skipped_for_varying_L():
    assert BTD((10, 10, 10), R=2, L=[1, 2]).check_uniqueness() is None


def test_constraint_matrix_structure():
    theta = _constraint_matrix(3, [1, 2, 3])
    expected = np.array(
        [
            [1, 0, 0, 0, 0, 0],
            [0, 1, 1, 0, 0, 0],
            [0, 0, 0, 1, 1, 1],
        ]
    )
    npt.assert_array_equal(theta, expected)
    npt.assert_array_equal(
        BTD((10, 10, 10), 3, [1, 2, 3]).get_constraint_matrix(), expected
    )


@pytest.mark.parametrize("block_mode", ["LL1", "L1L", "1LL"])
def test_factors_to_tensor_matches_definition(block_mode):
    dims, R, L = (7, 6, 5), 3, [1, 2, 3]
    factors = random_factors(dims, R, sum(L), block_mode)
    theta = _constraint_matrix(R, L)

    T = factors_to_tensor(*factors, theta, block_mode=block_mode)

    # build the LL1 tensor from the definition, then move the axes back
    A, B, C = to_ll1(factors, block_mode)
    T_ll1 = sum(btd_terms(A, B, C, L))
    axes = btd_als._LL1_AXES[block_mode]
    npt.assert_allclose(np.transpose(T, axes), T_ll1, atol=1e-12)


def test_factors_to_tensor_invalid_block_mode():
    with pytest.raises(ValueError, match="Unknown block mode"):
        factors_to_tensor(
            np.ones((2, 2)), np.ones((2, 2)), np.ones((2, 1)), np.ones((1, 2)), "LLL"
        )


def test_unfoldings():
    R, L = 3, [2, 2, 2]
    dims = (13, 12, 11)
    theta = _constraint_matrix(R, L)
    A = np.random.rand(dims[0], 6)
    B = np.random.rand(dims[1], 6)
    C = np.random.rand(dims[2], R)
    T = sum(btd_terms(A, B, C, L))

    npt.assert_allclose(tly.unfold(T, 0), A @ khatri_rao([B, C @ theta]).T)
    npt.assert_allclose(tly.unfold(T, 1), B @ khatri_rao([A, C @ theta]).T)
    npt.assert_allclose(tly.unfold(T, 2), C @ theta @ khatri_rao([A, B]).T)


def test_spatial_maps():
    A = np.random.rand(6, 6)
    B = np.random.rand(5, 6)
    maps = spatial_maps(A, B, 2)
    assert maps.shape == (3, 6, 5)
    for r in range(3):
        npt.assert_allclose(
            maps[r], A[:, 2 * r : 2 * r + 2] @ B[:, 2 * r : 2 * r + 2].T
        )


def test_fit_invalid_algorithm():
    model = BTD(dims=(10, 10, 10), R=2, L=2)
    with pytest.raises(ValueError, match="Unknown algorithm 'INVALID'"):
        model.fit(np.random.randn(10, 10, 10), algorithm="INVALID")


def test_fit_stores_consistent_results():
    model = BTD(dims=(10, 9, 8), R=2, L=2)
    T = np.random.randn(10, 9, 8)
    with pytest.warns(RuntimeWarning, match="max_iter=5"):
        assert model.fit(T, max_iter=5, rel_tol=0) is model
    assert len(model.factors) == 3
    assert len(model.fit_error) == 6
    rebuilt = factors_to_tensor(*model.factors, model.get_constraint_matrix())
    npt.assert_allclose(model.tensor, rebuilt)
    npt.assert_allclose(np.linalg.norm(T - model.tensor) ** 2, model.fit_error[-1])


# ──────────────────────────────────────────────────────────────────────────────
# ALS solver
# ──────────────────────────────────────────────────────────────────────────────


@pytest.mark.parametrize("strategy", ["random", "svd"])
def test_init_shapes(strategy):
    dims, R = (15, 14, 10), 3
    model = BTD(dims=dims, R=R, L=2)
    A, B, C = init_btd_factors(model, strategy=strategy, T=np.random.randn(*dims))
    assert A.shape == (15, 6)
    assert B.shape == (14, 6)
    assert C.shape == (10, 3)


def test_init_svd_spans_leading_singular_vectors():
    T = np.random.randn(15, 14, 10)
    A, _, _ = init_btd_factors(BTD(T.shape, 3, 2), strategy="svd", T=T)
    U = np.linalg.svd(tly.unfold(T, 0))[0][:, :6]
    # the columns of A lie in the span of the 6 leading left singular vectors
    npt.assert_allclose(U @ (U.T @ A), A, atol=1e-10)


def test_init_svd_requires_T():
    with pytest.raises(ValueError, match="SVD init requires input data T"):
        init_btd_factors(BTD((15, 14, 10), 3, 2), strategy="svd")


def test_init_unknown_strategy():
    with pytest.raises(ValueError, match="Unknown initialization 'bad'"):
        init_btd_factors(BTD((15, 14, 10), 3, 2), strategy="bad")


def test_solver_rejects_invalid_inputs():
    model = BTD(dims=(10, 10, 10), R=2, L=2)
    with pytest.raises(TypeError, match="must be an instance of the BTD class"):
        fit_btd_als("not_a_model", np.random.randn(10, 10, 10))
    with pytest.raises(TypeError, match="must be a numpy array"):
        fit_btd_als(model, None)
    with pytest.raises(ValueError, match="do not match"):
        fit_btd_als(model, np.random.randn(5, 5, 5))


@pytest.mark.parametrize("block_mode", ["LL1", "L1L", "1LL"])
def test_fit_error_never_increases(block_mode):
    # every ALS step solves its least-squares problem exactly
    dims = {"LL1": (9, 8, 7), "L1L": (9, 7, 8), "1LL": (7, 9, 8)}[block_mode]
    model = BTD(dims, R=3, L=2, block_mode=block_mode)
    with pytest.warns(RuntimeWarning, match="max_iter=30"):
        _, fit_error = fit_btd_als(
            model, np.random.randn(*dims), max_iter=30, rel_tol=0
        )
    assert len(fit_error) == 31
    assert np.all(np.diff(fit_error) <= 1e-10 * fit_error[:-1])


@pytest.mark.filterwarnings("ignore:Reached max_iter")
@pytest.mark.parametrize(
    ("block_mode", "dims"),
    [("LL1", (12, 11, 6)), ("L1L", (12, 6, 11)), ("1LL", (6, 12, 11))],
)
def test_converges_to_truth_from_nearby_start(block_mode, dims, monkeypatch):
    R, L = 3, [2, 2, 2]
    model = BTD(dims, R, L, block_mode=block_mode)
    true_factors = random_factors(dims, R, 6, block_mode)
    T = factors_to_tensor(*true_factors, model.get_constraint_matrix(), block_mode)

    # the solver starts from the truth plus a 1e-3 relative perturbation
    start = perturb(to_ll1(true_factors, block_mode), 1e-3)
    monkeypatch.setattr(btd_als, "_init_factors", lambda *args: start)
    model.fit(T, max_iter=500, rel_tol=1e-14, abs_tol=0)

    assert relative_error(model.tensor, T) < 1e-10
    estimated = btd_terms(*to_ll1(model.factors, block_mode), L)
    assert (
        term_recovery_error(estimated, btd_terms(*to_ll1(true_factors, block_mode), L))
        < 1e-8
    )


@pytest.mark.filterwarnings("ignore:Reached max_iter")
def test_best_of_restarts_recovers_terms():
    # ALS is not convex, so the best of a few random restarts is kept
    model = BTD((12, 11, 6), R=3, L=2)
    A0, B0, C0 = init_btd_factors(model)
    T = factors_to_tensor(A0, B0, C0, model.get_constraint_matrix())

    fits = []
    for _ in range(3):
        model.fit(T, init="random", max_iter=2000, rel_tol=1e-12, abs_tol=1e-14)
        fits.append((model.fit_error[-1], model.factors))
    _, best = min(fits, key=lambda fit: fit[0])

    assert (
        term_recovery_error(btd_terms(*best, model.L), btd_terms(A0, B0, C0, model.L))
        < 1e-6
    )


def test_noisy_data_fit_is_closer_to_truth_than_data():
    model = BTD((12, 11, 6), R=3, L=2)
    A0, B0, C0 = init_btd_factors(model)
    T0 = factors_to_tensor(A0, B0, C0, model.get_constraint_matrix())
    noise_level = 1e-2

    model.fit(
        add_noise(T0, noise_level), init="svd", max_iter=2000, rel_tol=1e-12, abs_tol=0
    )

    assert relative_error(model.tensor, T0) < noise_level
    assert (
        term_recovery_error(
            btd_terms(*model.factors, model.L), btd_terms(A0, B0, C0, model.L)
        )
        < 5 * noise_level
    )


def test_rel_tol_stops_early_and_logs(caplog):
    model = BTD((10, 9, 8), R=2, L=2)
    with caplog.at_level(logging.INFO, logger="pybbtd"):
        model.fit(np.random.randn(10, 9, 8), max_iter=1000, rel_tol=1e-2)
    n_iter = len(model.fit_error) - 1
    assert n_iter < 1000
    e = model.fit_error
    assert abs(e[-1] - e[-2]) / e[-1] < 1e-2
    assert f"Iteration {n_iter}: fit error stopped decreasing" in caplog.text


def test_abs_tol_stops_on_exact_data():
    model = BTD((10, 9, 8), R=2, L=2)
    A0, B0, C0 = init_btd_factors(model)
    T = factors_to_tensor(A0, B0, C0, model.get_constraint_matrix())
    model.fit(T, init="svd", max_iter=2000, rel_tol=0, abs_tol=1e-12)
    assert model.fit_error[-1] / np.linalg.norm(T) < 1e-12
    assert model.fit_error[-2] / np.linalg.norm(T) >= 1e-12
