import numpy as np
import numpy.testing as npt
import pytest
from _helpers import add_noise, btd_terms, relative_error, term_recovery_error

from pybbtd.btd import BTD, factors_to_tensor
from pybbtd.solvers.stokes_admm import fit_stokes_admm, init_stokes_factors
from pybbtd.stokes import (
    Stokes,
    coherency_to_stokes,
    generate_stokes_factors,
    is_valid_stokes,
    project_stokes,
    stokes_to_coherency,
    validate_stokes_tensor,
)


def project_onto_cone(S):
    """Closed-form Euclidean projection onto {S : S0 >= ||(S1, S2, S3)||}."""
    s0, s = S[0], S[1:]
    norm = np.linalg.norm(s)
    if norm <= s0:
        return S.copy()
    if norm <= -s0:
        return np.zeros(4)
    scale = (s0 + norm) / 2
    return np.concatenate([[scale], scale * s / norm])


# ──────────────────────────────────────────────────────────────────────────────
# Stokes model
# ──────────────────────────────────────────────────────────────────────────────


def test_init():
    model = Stokes([10, 9], R=2, L=3)
    assert isinstance(model, BTD)
    assert model.dims == (10, 9, 4)
    assert model.rank == 2
    npt.assert_array_equal(model.L, [3, 3])
    assert model.block_mode == "LL1"
    assert model.factors is None
    assert model.tensor is None
    assert model.fit_error is None
    assert repr(model) == "Stokes(dims=(10, 9, 4), R=2, L=[3, 3], block_mode='LL1')"


@pytest.mark.parametrize("spatial_dims", [(10,), (10, 9, 4), (10, 0)])
def test_invalid_spatial_dims(spatial_dims):
    with pytest.raises(ValueError, match="list or tuple of two positive integers"):
        Stokes(spatial_dims, R=2, L=2)


def test_generate_factors():
    A, B, C = generate_stokes_factors((8, 7, 4), R=3, L=[2, 2, 2])
    assert A.shape == (8, 6)
    assert B.shape == (7, 6)
    assert C.shape == (4, 3)
    assert A.min() >= 0
    assert B.min() >= 0
    # fully polarized states with unit intensity
    npt.assert_allclose(C[0], 1)
    npt.assert_allclose(np.linalg.norm(C[1:], axis=0), C[0])


def test_generate_tensor_is_valid():
    model = Stokes([8, 7], R=2, L=1)
    factors, T = model.generate_stokes_tensor()
    assert model.factors is factors
    npt.assert_allclose(T, factors_to_tensor(*factors, model.get_constraint_matrix()))
    assert validate_stokes_tensor(T) is True


def test_validate_stokes_tensor_warns_on_invalid_pixels():
    T = np.zeros((2, 5, 4))
    T[..., 0] = 1.0
    T[0, :, 1] = 2.0  # half the pixels are more polarized than intense
    with pytest.warns(UserWarning, match="50.00% of pixels"):
        assert validate_stokes_tensor(T) is False


def test_fit_invalid_algorithm():
    with pytest.raises(ValueError, match="Unknown algorithm 'ALS'"):
        Stokes([10, 10], R=2, L=1).fit(np.zeros((10, 10, 4)), algorithm="ALS")


# ──────────────────────────────────────────────────────────────────────────────
# Stokes vectors
# ──────────────────────────────────────────────────────────────────────────────


@pytest.mark.parametrize(
    ("S", "valid"),
    [
        ([1.0, 0.5, 0.5, 0.5], True),
        ([1.0, 1.0, 0.0, 0.0], True),
        ([1.0, 0.0, 0.0, 0.0], True),
        ([0.0, 0.0, 0.0, 0.0], True),
        ([-1.0, 0.0, 0.0, 0.0], False),
        ([0.1, 0.5, 0.5, 0.5], False),
    ],
)
def test_is_valid_stokes(S, valid):
    assert is_valid_stokes(np.array(S)) is valid


def test_coherency_round_trips():
    S = np.array([1.0, 0.3, 0.4, 0.2])
    npt.assert_allclose(coherency_to_stokes(stokes_to_coherency(S)), S, atol=1e-12)
    coh = np.array([[0.6, 0.2 + 0.1j], [0.2 - 0.1j, 0.4]])
    npt.assert_allclose(stokes_to_coherency(coherency_to_stokes(coh)), coh, atol=1e-12)


def test_coherency_of_valid_stokes_is_psd_with_trace_s0():
    S = np.array([2.0, 0.5, -1.0, 1.2])
    coh = stokes_to_coherency(S)
    npt.assert_allclose(coh, coh.conj().T)
    assert np.linalg.eigvalsh(coh).min() >= -1e-12
    npt.assert_allclose(np.trace(coh).real, S[0])


def test_project_stokes_matches_closed_form_cone_projection():
    for _ in range(200):
        S = 2 * np.random.randn(4)
        npt.assert_allclose(project_stokes(S), project_onto_cone(S), atol=1e-12)


def test_project_stokes_is_idempotent_and_valid():
    for _ in range(50):
        P = project_stokes(np.random.randn(4))
        assert is_valid_stokes(P)
        npt.assert_allclose(project_stokes(P), P, atol=1e-12)


# ──────────────────────────────────────────────────────────────────────────────
# Initialization
# ──────────────────────────────────────────────────────────────────────────────


def test_init_unknown_strategy_and_missing_data():
    model = Stokes([6, 5], R=2, L=2)
    with pytest.raises(ValueError, match="Unknown initialization 'bad'"):
        init_stokes_factors(model, strategy="bad")
    with pytest.raises(ValueError, match="K-means init requires input data T"):
        init_stokes_factors(model, strategy="kmeans")


def test_kmeans_init_finds_stokes_vectors_of_separate_regions():
    # each pixel belongs to a single term, so each cluster center is a
    # multiple of one true Stokes vector
    R, L = 3, 2
    model = Stokes([10, 12], R=R, L=L)
    A = 0.5 + 0.5 * np.random.rand(10, L * R)
    B = np.zeros((12, L * R))
    for r in range(R):
        B[4 * r : 4 * r + 4, L * r : L * r + L] = 0.5 + 0.5 * np.random.rand(4, L)
    _, _, C = generate_stokes_factors(model.dims, R, model.L)
    T = factors_to_tensor(A, B, C, model.get_constraint_matrix())

    A0, B0, C0 = init_stokes_factors(model, strategy="kmeans", T=T)

    assert A0.min() >= 0
    assert B0.min() >= 0
    # unit-intensity centers equal the true (unit-intensity) Stokes vectors
    order = [int(np.argmin(np.linalg.norm(C - c[:, None], axis=0))) for c in C0.T]
    assert sorted(order) == list(range(R))
    npt.assert_allclose(C0, C[:, order], atol=1e-10)


# ──────────────────────────────────────────────────────────────────────────────
# AO-ADMM solver
# ──────────────────────────────────────────────────────────────────────────────


def test_solver_rejects_invalid_inputs():
    model = Stokes([2, 3], R=1, L=1)
    T = np.random.rand(2, 3, 4)
    with pytest.raises(TypeError, match="must be an instance of the Stokes class"):
        fit_stokes_admm(model=0, T=T)
    with pytest.raises(TypeError, match="must be a numpy array"):
        fit_stokes_admm(model, T=None)
    with pytest.raises(ValueError, match="do not match"):
        fit_stokes_admm(model, T=np.random.rand(3, 2, 5))
    with pytest.raises(ValueError, match="Unknown initialization 'wrong_one'"):
        fit_stokes_admm(model, T=T, init="wrong_one")


@pytest.mark.filterwarnings("ignore:Reached max_iter")
@pytest.mark.parametrize("init", ["random", "kmeans"])
def test_fit_keeps_constraints(init):
    model = Stokes([8, 9], R=2, L=2)
    _, T = model.generate_stokes_tensor()

    model.fit(add_noise(T, 0.2), init=init, max_iter=50)
    A, B, C = model.factors

    assert A.min() >= 0
    assert B.min() >= 0
    assert all(is_valid_stokes(c) for c in C.T)


@pytest.mark.parametrize("seed", [0, 1, 2, 3])
@pytest.mark.parametrize("init", ["random", "kmeans"])
def test_fit_recovers_terms(init, seed):
    np.random.seed(seed)
    model = Stokes([12, 12], R=2, L=2)
    (A0, B0, C0), T = model.generate_stokes_tensor()

    model.fit(T, init=init, max_iter=3000, rel_tol=1e-12, abs_tol=1e-14, admm_tol=1e-10)

    assert relative_error(model.tensor, T) < 1e-6
    estimated = btd_terms(*model.factors, model.L)
    assert term_recovery_error(estimated, btd_terms(A0, B0, C0, model.L)) < 1e-5


@pytest.mark.filterwarnings("ignore:Reached max_iter")
def test_fit_noisy_data_is_closer_to_truth_than_data():
    model = Stokes([12, 12], R=2, L=2)
    (A0, B0, C0), T0 = model.generate_stokes_tensor()
    noise_level = 1e-2

    model.fit(
        add_noise(T0, noise_level),
        init="kmeans",
        max_iter=3000,
        rel_tol=1e-12,
        abs_tol=0,
    )

    assert relative_error(model.tensor, T0) < noise_level
    estimated = btd_terms(*model.factors, model.L)
    assert (
        term_recovery_error(estimated, btd_terms(A0, B0, C0, model.L)) < 5 * noise_level
    )


def test_fit_error_is_squared_error_and_max_iter_warns():
    model = Stokes([10, 10], R=1, L=1)
    _, T = model.generate_stokes_tensor()
    with pytest.warns(RuntimeWarning, match="max_iter=2"):
        model.fit(T, init="kmeans", max_iter=2, rel_tol=-1, abs_tol=-1)
    assert len(model.fit_error) == 3
    npt.assert_allclose(model.fit_error[-1], np.linalg.norm(T - model.tensor) ** 2)
