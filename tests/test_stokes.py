import pytest
from pybbtd.stokes import Stokes
import numpy as np
import pybbtd.stokes as stokes
import numpy.testing as npt
import pybbtd.btd as btd
import pybbtd.solvers.stokes_admm as stokes_admm


def test_stokes_constructor():
    spatial_dims = [10, 10]
    R = 1
    L = 2
    stokes = Stokes(spatial_dims, R, L)
    npt.assert_equal(stokes.dims, tuple(spatial_dims + [4]))
    npt.assert_equal(stokes.rank, R)
    npt.assert_equal(stokes.L, L)


def test_validate_dims():
    spatial_dims = [10, 10]
    R = 1
    L = 2
    stokes = Stokes(spatial_dims, R, L)
    assert stokes.validate_dims()


def test_generate_stokes_factors():
    spatial_dims = [8, 7]
    R = 2
    L = [1, 1]
    factors = stokes.generate_stokes_factors(tuple(spatial_dims + [4]), R, L)
    npt.assert_equal(factors[0].shape, (8, 2))
    npt.assert_equal(factors[1].shape, (7, 2))
    npt.assert_equal(factors[2].shape, (4, 2))


def test_generate_stokes_tensor():
    spatial_dims = [8, 7]
    R = 2
    L = [1, 1]
    stokes = Stokes(spatial_dims, R, L)
    _, tensor = stokes.generate_stokes_tensor()
    npt.assert_equal(tensor.shape, (8, 7, 4))


def test_validate_stokes_tensor_warning():
    T0 = np.random.rand(10, 9, 4)
    with pytest.warns(UserWarning, match="Stokes constraints"):
        stokes.validate_stokes_tensor(T0)


def test_stokes_kmeans():
    R = 2
    L = 2
    X = stokes.Stokes([15, 15], R, L)
    _, T0 = X.generate_stokes_tensor()

    features, initialC = stokes_admm.stokes_kmeans(R, T0)

    assert features.shape == (R, T0.shape[0], T0.shape[1])
    assert initialC.shape == (4, R)


def test_fit_admm_algorithm_random_init():
    np.random.seed(10)

    R = 2
    L = 2
    btd.validate_R_L(R, L)
    X = stokes.Stokes([15, 15], R, L)

    [A0, B0, C0], T0 = X.generate_stokes_tensor()
    theta = X.get_constraint_matrix()
    Tnoisy = btd.factors_to_tensor(
        A0, B0, C0, theta, block_mode="LL1"
    ) + 0 * 1e-5 * np.random.randn(*X.dims)
    X.fit(
        data=Tnoisy,
        algorithm="ADMM",
        init="random",
        max_iter=1000,
        rho=1,
        max_admm=1,
        rel_tol=10**-8,
        abs_tol=10**-15,
        admm_tol=10**-10,
    )
    # Assert

    assert X.fit_error[-1] < 10**-8


def test_fit_admm_algorithm_kmeans_init():
    np.random.seed(10)

    R = 2
    L = 2
    btd.validate_R_L(R, L)
    X = stokes.Stokes([15, 15], R, L)

    [A0, B0, C0], T0 = X.generate_stokes_tensor()
    theta = X.get_constraint_matrix()
    Tnoisy = btd.factors_to_tensor(
        A0, B0, C0, theta, block_mode="LL1"
    ) + 0 * 1e-5 * np.random.randn(*X.dims)
    X.fit(
        data=Tnoisy,
        algorithm="ADMM",
        init="kmeans",
        max_iter=1000,
        rho=1,
        max_admm=1,
        rel_tol=10**-8,
        abs_tol=10**-15,
        admm_tol=10**-10,
    )
    # Assert

    assert X.fit_error[-1] < 10**-8


def test_fit_invalid_algorithm_raises_warning():
    model = Stokes([10, 10], 2, [1, 1])

    with pytest.raises(UserWarning, match="Algorithm not implemented yet"):
        model.fit("dummy_data", algorithm="NotImplemented_ForceError")


if __name__ == "__main__":
    pytest.main()
