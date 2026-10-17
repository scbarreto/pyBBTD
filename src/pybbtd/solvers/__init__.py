"""Fitting algorithms for the pyBBTD models.

Each model's ``fit`` method calls one of these solvers; they can also be
called directly with a model and a data tensor.
"""

from pybbtd.solvers.bbtd_als import fit_bbtd_als, init_bbtd_factors
from pybbtd.solvers.bbtd_cov_admm import fit_bbtd_cov_admm, init_bbtd_cov_factors
from pybbtd.solvers.btd_als import fit_btd_als, init_btd_factors
from pybbtd.solvers.covll1_admm import fit_covll1_admm, init_covll1_factors
from pybbtd.solvers.stokes_admm import fit_stokes_admm, init_stokes_factors

__all__ = [
    "fit_bbtd_als",
    "fit_bbtd_cov_admm",
    "fit_btd_als",
    "fit_covll1_admm",
    "fit_stokes_admm",
    "init_bbtd_cov_factors",
    "init_bbtd_factors",
    "init_btd_factors",
    "init_covll1_factors",
    "init_stokes_factors",
]
