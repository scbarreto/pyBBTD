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
    S = Stokes(spatial_dims, R, L)
    npt.assert_equal(S.dims, tuple(spatial_dims + [4]))
    npt.assert_equal(S.rank, R)
    npt.assert_equal(S.L, L)


def test_validate_stokes_tensor():
    spatial_dims = [4, 4]
    R = 2
    L = [1, 1]
    S = Stokes(spatial_dims, R, L)
    _, tensor = S.generate_stokes_tensor()
    stokes.validate_stokes_tensor(tensor)


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


def test_fit_admm_random_init():
    np.random.seed(0)
    R = 2
    L = 1
    btd._validate_R_L(R, L)
    X = stokes.Stokes([7, 8], R, L)

    [A0, B0, C0], T0 = X.generate_stokes_tensor()
    theta = X.get_constraint_matrix()
    Tnoisy = btd.factors_to_tensor(
        A0, B0, C0, theta, block_mode="LL1"
    ) + 0 * 1e-8 * np.random.randn(*X.dims)
    X.fit(
        data=Tnoisy,
        algorithm="ADMM",
        init="random",
        max_iter=2000,
        rho=1,
        max_admm=1,
        rel_tol=10**-3,
        abs_tol=10**-15,
    )

    print(X.fit_error[-1])

    assert X.fit_error[-1] < 10**-1


def test_fit_admm_kmeans_init():
    np.random.seed(10)

    R = 1
    L = 2
    btd._validate_R_L(R, L)
    X = stokes.Stokes([15, 15], R, L)

    [A0, B0, C0], T0 = X.generate_stokes_tensor()

    X.fit(
        data=T0,
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


def test_wrong_tensor_init_kmeans():
    # Create Stokes model
    R = 2
    L = 2
    btd._validate_R_L(R, L)
    X = stokes.Stokes([2, 3], R, L)
    Twrong = np.random.rand(3, 2, 5)
    Tright = np.random.rand(2, 3, 4)
    with pytest.raises(ValueError, match="do not match"):
        X.fit(init="kmeans", data=Twrong)
    with pytest.raises(TypeError, match="must be a numpy"):
        X.fit(init="kmeans", data=None)
    with pytest.raises(ValueError, match="not implemented"):
        X.fit(init="wrong_one", data=Tright)


def test_admm_convergence():
    np.random.seed(10)

    R = 3
    L = 1
    btd._validate_R_L(R, L)
    X = stokes.Stokes([15, 15], R, L)

    [A0, B0, C0], T0 = X.generate_stokes_tensor()
    theta = X.get_constraint_matrix()
    Tnoisy = btd.factors_to_tensor(
        A0, B0, C0, theta, block_mode="LL1"
    ) + 0 * 1e-2 * np.random.randn(*X.dims)
    X.fit(
        data=Tnoisy,
        algorithm="ADMM",
        init="kmeans",
        max_iter=1000,
        rho=1,
        max_admm=1,
        rel_tol=10**-15,
        abs_tol=10**-15,
        admm_tol=10**-5,
    )
    # Assert

    assert X.fit_error[-1] < 10**-5


def test_stokes_instance():
    T = np.random.rand(10, 10, 4)
    Stokes_model = 0  # Replace with the desired value for testing

    with pytest.raises(TypeError, match="must be an instance of the Stokes class."):
        stokes_admm.Stokes_ADMM(Stokes_model=Stokes_model, T=T)


def test_max_iter_criteria():
    R = 1
    L = 1
    btd._validate_R_L(R, L)
    X = stokes.Stokes([10, 10], R, L)

    [A0, B0, C0], T0 = X.generate_stokes_tensor()
    theta = X.get_constraint_matrix()
    Tnoisy = btd.factors_to_tensor(
        A0, B0, C0, theta, block_mode="LL1"
    ) + 0 * 1e-5 * np.random.randn(*X.dims)
    # Assert
    with pytest.warns(RuntimeWarning, match="max number of iteration"):
        X.fit(
            data=Tnoisy,
            algorithm="ADMM",
            init="kmeans",
            max_iter=1,
            rho=1,
            max_admm=1,
            rel_tol=-1,
            abs_tol=-1,
            admm_tol=10**-10,
        )


if __name__ == "__main__":
    pytest.main()
