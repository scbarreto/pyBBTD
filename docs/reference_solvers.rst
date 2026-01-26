.. _reference.solvers:

Solvers
==============

BTD -- ALS
---------------

.. currentmodule:: pybbtd.solvers.btd_als

.. list-table::
   :header-rows: 0

   * - :func:`~pybbtd.solvers.btd_als.BTD_ALS`
     - Alternating Least Squares solver for the BTD decomposition.
   * - :func:`~pybbtd.solvers.btd_als.init_BTD_factors`
     - Initialize factor matrices for the BTD decomposition.

.. automodule:: pybbtd.solvers.btd_als
    :members:
    :show-inheritance:

Cov-LL1 -- ADMM
-------------------

.. currentmodule:: pybbtd.solvers.covll1_admm

.. list-table::
   :header-rows: 0

   * - :func:`~pybbtd.solvers.covll1_admm.CovLL1_ADMM`
     - AO-ADMM solver for the Cov-LL1 decomposition.
   * - :func:`~pybbtd.solvers.covll1_admm.ADMM_A`
     - Inner ADMM update for factor A with non-negativity constraint.
   * - :func:`~pybbtd.solvers.covll1_admm.ADMM_B`
     - Inner ADMM update for factor B with non-negativity constraint.
   * - :func:`~pybbtd.solvers.covll1_admm.ADMM_C`
     - Inner ADMM update for factor C with PSD covariance constraint.
   * - :func:`~pybbtd.solvers.covll1_admm.init_covll1_factors`
     - Initialize factor matrices for the Cov-LL1 decomposition.
   * - :func:`~pybbtd.solvers.covll1_admm.kmeans_init`
     - K-means based initialization for the Cov-LL1 decomposition.
   * - :func:`~pybbtd.solvers.covll1_admm.project_to_psd`
     - Project a covariance matrix onto the PSD cone.

.. automodule:: pybbtd.solvers.covll1_admm
    :members:
    :show-inheritance:

Stokes -- ADMM
-----------------

.. currentmodule:: pybbtd.solvers.stokes_admm

.. list-table::
   :header-rows: 0

   * - :func:`~pybbtd.solvers.stokes_admm.Stokes_ADMM`
     - AO-ADMM solver for the Stokes-constrained BTD-LL1 decomposition.
   * - :func:`~pybbtd.solvers.stokes_admm.ADMM_C`
     - Inner ADMM update for factor C with Stokes constraint.
   * - :func:`~pybbtd.solvers.stokes_admm.init_Stokes_factors`
     - Initialize factor matrices for the Stokes-BTD decomposition.
   * - :func:`~pybbtd.solvers.stokes_admm.kmeans_init`
     - K-means based initialization for the Stokes-BTD decomposition.

.. automodule:: pybbtd.solvers.stokes_admm
    :members:
    :show-inheritance:

BBTD -- Vanilla ALS
-----------------------

.. currentmodule:: pybbtd.solvers.bbtd_vanilla_als

.. list-table::
   :header-rows: 0

   * - :func:`~pybbtd.solvers.bbtd_vanilla_als.BBTD_ALS`
     - Vanilla ALS solver for the unconstrained BBTD decomposition.
   * - :func:`~pybbtd.solvers.bbtd_vanilla_als.init_BBTD_factors`
     - Initialize factor matrices for the BBTD decomposition.

.. automodule:: pybbtd.solvers.bbtd_vanilla_als
    :members:
    :show-inheritance:

BBTD -- Constrained ADMM
----------------------------

.. currentmodule:: pybbtd.solvers.bbtd_cov_admm

.. list-table::
   :header-rows: 0

   * - :func:`~pybbtd.solvers.bbtd_cov_admm.BBTD_COV_ADMM`
     - ADMM solver for the constrained BBTD decomposition.
   * - :func:`~pybbtd.solvers.bbtd_cov_admm.init_BBTD_cov_factors`
     - Initialize factor matrices for the constrained BBTD decomposition.

.. automodule:: pybbtd.solvers.bbtd_cov_admm
    :members:
    :show-inheritance:
