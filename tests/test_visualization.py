import matplotlib.pyplot as plt
import numpy as np
import numpy.testing as npt
import pytest

from pybbtd.stokes import generate_stokes_factors
from pybbtd.visualization import (
    plot_bbtd_cov_terms,
    plot_error,
    plot_stokes_terms,
    stokes_to_ellipse,
)


@pytest.mark.parametrize(
    ("S", "psi", "chi"),
    [
        ([1, 1, 0, 0], 0, 0),  # horizontal linear
        ([1, 0, 1, 0], np.pi / 4, 0),  # linear at +45 degrees
        ([1, -1, 0, 0], np.pi / 2, 0),  # vertical linear
        ([1, 0, 0, 1], 0, np.pi / 4),  # circular, positive S3
        ([1, 0, 0, -1], 0, -np.pi / 4),  # circular, negative S3
    ],
)
def test_stokes_to_ellipse_angles(S, psi, chi):
    x, y, psi_est, chi_est = stokes_to_ellipse(np.array(S, dtype=float))
    npt.assert_allclose([psi_est, chi_est], [psi, chi], atol=1e-12)
    npt.assert_allclose(np.sqrt(x**2 + y**2).max(), 1)


def test_stokes_to_ellipse_shapes():
    x, y, _, _ = stokes_to_ellipse(np.array([1.0, 0.0, 0.0, 1.0]), n_points=50)
    assert x.shape == y.shape == (50,)
    # a circular state traces a circle
    npt.assert_allclose(np.sqrt(x**2 + y**2), 1)


def test_plot_error_single_curve():
    plot_error(np.array([1.0, 0.1, 0.01]), title="fit")
    ax = plt.gcf().axes[0]
    assert len(ax.lines) == 1
    assert ax.get_yscale() == "log"
    assert ax.get_title() == "fit"


def test_plot_error_several_curves_with_labels():
    plot_error([np.ones(3), np.ones(4)], labels=["a", "b"], semilogy=False)
    ax = plt.gcf().axes[0]
    assert len(ax.lines) == 2
    assert ax.get_yscale() == "linear"
    assert [t.get_text() for t in ax.get_legend().get_texts()] == ["a", "b"]


def test_plot_stokes_terms():
    R, L = 3, 2
    A, B, C = generate_stokes_factors((8, 7, 4), R, [L] * R)
    plot_stokes_terms(A, B, C, R, L)
    titles = [ax.get_title() for ax in plt.gcf().axes]
    assert sum("A_" in t for t in titles) == R
    assert sum("psi" in t for t in titles) == R


@pytest.mark.parametrize(
    ("shapes", "message"),
    [
        (((8, 4), (7, 4), (3, 2)), "shape \\(4, R\\)"),
        (((8, 4), (7, 4), (4, 3)), "R=2"),
        (((8, 3), (7, 4), (4, 2)), "A has 3 columns"),
        (((8, 4), (7, 5), (4, 2)), "B has 5 columns"),
    ],
)
def test_plot_stokes_terms_rejects_bad_shapes(shapes, message):
    A, B, C = (np.ones(shape) for shape in shapes)
    with pytest.raises(ValueError, match=message):
        plot_stokes_terms(A, B, C, R=2, L=2)


def test_plot_bbtd_cov_terms():
    R, L1, L2 = 3, 2, 2
    A = np.random.rand(8, L1 * R)
    B = np.random.rand(7, L1 * R)
    C = np.random.randn(4, L2 * R) + 1j * np.random.randn(4, L2 * R)
    plot_bbtd_cov_terms(A, B, C, R, L1, L2)
    titles = [ax.get_title() for ax in plt.gcf().axes]
    assert sum(t.startswith("Spatial") for t in titles) == R
    assert sum(t.startswith("Covariance") for t in titles) == R
