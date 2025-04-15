import pytest
from pybbtd.stokes import Stokes
import numpy as np
import pybbtd.stokes as stokes
import numpy.testing as npt


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


# Run the tests
if __name__ == "__main__":
    pytest.main()
