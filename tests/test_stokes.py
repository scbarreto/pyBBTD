import pytest
import numpy as np
import numpy.testing as npt
from pybbtd.stokes import (
    Stokes,
    validate_stokes_tensor,
    generate_stokes_factors,
    check_stokes_constraints,
    stokes2coh,
    coh2stokes,
    stokes_projection,
)
from pybbtd.btd import BTD, factors_to_tensor
from pybbtd.solvers.stokes_admm import STOKES_ADMM


# ──────────────────────────────────────────────────────────────────────────────
# Section 1: Stokes class (stokes.py)
# ──────────────────────────────────────────────────────────────────────────────


def test_stokes_constructor():
    S = Stokes([10, 10], R=1, L=2)
    npt.assert_equal(S.dims, (10, 10, 4))
    npt.assert_equal(S.rank, 1)
    npt.assert_equal(S.L, 2)


def test_stokes_inherits_btd():
    S = Stokes([10, 10], R=2, L=1)
    assert isinstance(S, BTD)


def test_stokes_block_mode_is_LL1():
    S = Stokes([10, 10], R=2, L=2)
    assert S.block_mode == "LL1"


def test_stokes_stores_none_state():
    S = Stokes([10, 10], R=2, L=1)
    assert S.factors is None
    assert S.tensor is None
    assert S.fit_error is None


def test_generate_stokes_factors_shapes():
    np.random.seed(42)
    R = 2
    L = [1, 1]
    factors = generate_stokes_factors((8, 7, 4), R, L)
    npt.assert_equal(factors[0].shape, (8, 2))
    npt.assert_equal(factors[1].shape, (7, 2))
    npt.assert_equal(factors[2].shape, (4, 2))


def test_generate_stokes_factors_nonneg():
    np.random.seed(42)
    R = 2
    L = [2, 2]
    A, B, C = generate_stokes_factors((10, 8, 4), R, L)
    assert np.all(A >= 0)
    assert np.all(B >= 0)


def test_generate_stokes_tensor_shape():
    np.random.seed(42)
    S = Stokes([8, 7], R=2, L=[1, 1])
    _, tensor = S.generate_stokes_tensor()
    npt.assert_equal(tensor.shape, (8, 7, 4))


def test_generate_stokes_tensor_valid():
    np.random.seed(42)
    S = Stokes([4, 4], R=2, L=[1, 1])
    _, tensor = S.generate_stokes_tensor()
    validate_stokes_tensor(tensor)


def test_validate_stokes_tensor_warning():
    np.random.seed(42)
    T0 = np.random.rand(10, 9, 4)
    with pytest.warns(UserWarning, match="Stokes constraints"):
        validate_stokes_tensor(T0)


def test_fit_invalid_algorithm():
    model = Stokes([10, 10], R=2, L=[1, 1])
    with pytest.raises(UserWarning, match="Algorithm not implemented yet"):
        model.fit("dummy_data", algorithm="INVALID")


# ──────────────────────────────────────────────────────────────────────────────
# Section 2: Stokes utilities (stokes2coh, coh2stokes, constraints, projection)
# ──────────────────────────────────────────────────────────────────────────────


def test_stokes_constraints_valid_vector():
    S = np.array([1.0, 0.5, 0.5, 0.5])
    assert check_stokes_constraints(S) == 1


def test_stokes_constraints_invalid_negative_S0():
    S = np.array([-1.0, 0.0, 0.0, 0.0])
    assert check_stokes_constraints(S) == 0


def test_stokes_constraints_invalid_norm():
    S = np.array([0.1, 0.5, 0.5, 0.5])
    assert check_stokes_constraints(S) == 0


def test_stokes2coh_coh2stokes_roundtrip():
    S_orig = np.array([1.0, 0.3, 0.4, 0.2])
    coh = stokes2coh(S_orig)
    S_rec = coh2stokes(coh)
    npt.assert_allclose(S_rec, S_orig, atol=1e-12)


def test_coh2stokes_stokes2coh_roundtrip():
    coh_orig = np.array([[0.6, 0.2 + 0.1j], [0.2 - 0.1j, 0.4]])
    S = coh2stokes(coh_orig)
    coh_rec = stokes2coh(S)
    npt.assert_allclose(coh_rec, coh_orig, atol=1e-12)


def test_stokes_projection_valid_vector_unchanged():
    S = np.array([1.0, 0.5, 0.5, 0.5])
    S_proj = stokes_projection(S)
    npt.assert_allclose(S_proj, S, atol=1e-10)


def test_stokes_projection_produces_valid_vector():
    S_invalid = np.array([0.1, 0.5, 0.5, 0.5])
    S_proj = stokes_projection(S_invalid)
    assert check_stokes_constraints(S_proj) == 1


# ──────────────────────────────────────────────────────────────────────────────
# Section 3: ADMM solver (stokes_admm.py)
# ──────────────────────────────────────────────────────────────────────────────


def test_admm_invalid_model_type():
    T = np.random.rand(10, 10, 4)
    with pytest.raises(TypeError, match="must be an instance of the Stokes class"):
        STOKES_ADMM(Stokes_model=0, T=T)


def test_admm_invalid_tensor_type():
    model = Stokes([10, 10], R=2, L=2)
    with pytest.raises(TypeError, match="must be a numpy"):
        STOKES_ADMM(model, T=None)


def test_admm_mismatched_dims():
    model = Stokes([2, 3], R=2, L=2)
    T_wrong = np.random.rand(3, 2, 5)
    with pytest.raises(ValueError, match="do not match"):
        STOKES_ADMM(model, T=T_wrong)


def test_admm_invalid_init_strategy():
    model = Stokes([2, 3], R=2, L=2)
    T = np.random.rand(2, 3, 4)
    with pytest.raises(ValueError, match="not implemented"):
        STOKES_ADMM(model, T=T, init="wrong_one")


def test_admm_fit_random_init():
    np.random.seed(0)
    X = Stokes([7, 8], R=2, L=1)

    [A0, B0, C0], T0 = X.generate_stokes_tensor()
    theta = X.get_constraint_matrix()
    T_gt = factors_to_tensor(
        A0, B0, C0, theta, block_mode="LL1"
    ) + 0 * 1e-8 * np.random.randn(*X.dims)

    X.fit(
        data=T_gt,
        algorithm="ADMM",
        init="random",
        max_iter=2000,
        rho=1,
        max_admm=1,
        rel_tol=1e-3,
        abs_tol=1e-15,
    )
    assert X.fit_error[-1] < 1e-1


def test_admm_fit_kmeans_init():
    np.random.seed(10)
    X = Stokes([15, 15], R=1, L=2)
    [A0, B0, C0], T0 = X.generate_stokes_tensor()

    X.fit(
        data=T0,
        algorithm="ADMM",
        init="kmeans",
        max_iter=1000,
        rho=1,
        max_admm=1,
        rel_tol=1e-8,
        abs_tol=1e-15,
        admm_tol=1e-10,
    )
    assert X.fit_error[-1] < 1e-8


def test_admm_convergence():
    np.random.seed(10)
    X = Stokes([15, 15], R=3, L=1)

    [A0, B0, C0], T0 = X.generate_stokes_tensor()
    theta = X.get_constraint_matrix()
    T_gt = factors_to_tensor(
        A0, B0, C0, theta, block_mode="LL1"
    ) + 0 * 1e-2 * np.random.randn(*X.dims)

    X.fit(
        data=T_gt,
        algorithm="ADMM",
        init="kmeans",
        max_iter=1000,
        rho=1,
        max_admm=1,
        rel_tol=1e-15,
        abs_tol=1e-15,
        admm_tol=1e-5,
    )
    assert X.fit_error[-1] < 1e-5


def test_admm_stokes_constraints_after_fit():
    np.random.seed(10)
    X = Stokes([15, 15], R=2, L=1)
    [A0, B0, C0], T0 = X.generate_stokes_tensor()

    X.fit(
        data=T0,
        algorithm="ADMM",
        init="kmeans",
        max_iter=500,
        rho=1,
        max_admm=1,
        rel_tol=1e-8,
        abs_tol=1e-15,
        admm_tol=1e-10,
    )
    C_est = X.factors[2]
    for r in range(X.rank):
        assert check_stokes_constraints(C_est[:, r]) == 1, (
            f"Column {r} of C does not satisfy Stokes constraints"
        )


def test_admm_max_iter_warning():
    np.random.seed(42)
    X = Stokes([10, 10], R=1, L=1)
    [A0, B0, C0], T0 = X.generate_stokes_tensor()
    theta = X.get_constraint_matrix()
    T_gt = factors_to_tensor(
        A0, B0, C0, theta, block_mode="LL1"
    ) + 0 * 1e-5 * np.random.randn(*X.dims)

    with pytest.warns(RuntimeWarning, match="max number of iteration"):
        X.fit(
            data=T_gt,
            algorithm="ADMM",
            init="kmeans",
            max_iter=1,
            rho=1,
            max_admm=1,
            rel_tol=-1,
            abs_tol=-1,
            admm_tol=1e-10,
        )


if __name__ == "__main__":
    pytest.main()
