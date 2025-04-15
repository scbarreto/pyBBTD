import pytest
from pybbtd.stokes import Stokes
import numpy as np
import pybbtd.stokes as stokes
# Test Stokes class


def test_stokes_constructor():
    spatial_dims = [10, 10]
    R = 1
    L = 2
    stokes = Stokes(spatial_dims, R, L)
    assert stokes.dims == tuple(spatial_dims + [4])
    assert stokes.rank == R
    assert stokes.L == L


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
    stokes = Stokes(spatial_dims, R, L)
    factors = stokes.generate_stokes_factors(tuple(spatial_dims + [4]), R, L)
    assert factors[0].shape == (8, 2)
    assert factors[1].shape == (7, 2)
    assert factors[2].shape == (4, 2)


def test_load_stokes_tensor():
    T0 = np.ones((3, 3, 4))
    for i in range(3):
        for j in range(3):
            T0[i, j, :] = [1, 0.3, 0.5, 0.2]
    R = 1
    L = 2
    loaded_object = stokes.load_stokes_tensor(T0, R, L)
    expected_tensor = np.array(
        [
            [[1.0, 0.3, 0.5, 0.2], [1.0, 0.3, 0.5, 0.2], [1.0, 0.3, 0.5, 0.2]],
            [[1.0, 0.3, 0.5, 0.2], [1.0, 0.3, 0.5, 0.2], [1.0, 0.3, 0.5, 0.2]],
            [[1.0, 0.3, 0.5, 0.2], [1.0, 0.3, 0.5, 0.2], [1.0, 0.3, 0.5, 0.2]],
        ]
    )

    assert np.array_equal(loaded_object.tensor, expected_tensor)


def test_generate_stokes_tensor():
    spatial_dims = [8, 7]
    R = 2
    L = [1, 1]
    stokes = Stokes(spatial_dims, R, L)
    _, tensor = stokes.generate_stokes_tensor()
    assert tensor.shape == (8, 7, 4)


def test_validate_stokes_tensor():
    # Generate a random tensor gthat does not satisfy the Stokes constraints and check if the warning is caught
    T0 = np.random.rand(10, 9, 4)

    assert not stokes.validate_stokes_tensor(T0)


# Run the tests
if __name__ == "__main__":
    pytest.main()
