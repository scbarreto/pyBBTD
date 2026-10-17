import numpy as np
import numpy.testing as npt
import pytest

import pybbtd
from pybbtd import BBTD, BTD, CovLL1, Stokes, solvers, visualization


def test_top_level_exports():
    assert set(pybbtd.__all__) >= {"BTD", "BBTD", "CovLL1", "Stokes", "__version__"}
    assert isinstance(pybbtd.__version__, str)
    assert issubclass(CovLL1, BTD)
    assert issubclass(Stokes, BTD)


def test_solver_and_plot_exports():
    for name in solvers.__all__:
        assert callable(getattr(solvers, name))
    for name in visualization.__all__:
        assert callable(getattr(visualization, name))


def test_bbtd_admm_dispatches_to_covariance_solver():
    model = BBTD((6, 5, 3, 3), R=2, L1=2, L2=1)
    T = np.random.rand(6, 5, 3, 3) + 0j
    with pytest.warns(RuntimeWarning, match="max_iter=2"):
        model.fit(T, algorithm="ADMM", max_iter=2, rel_tol=0, abs_tol=0)
    C, D = model.factors[2], model.factors[3]
    npt.assert_array_equal(D, C.conj())
