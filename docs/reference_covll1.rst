.. _reference.covll1:

.. currentmodule:: pybbtd.covll1

Cov-LL1 Class
=============

Class Methods
---------------

.. list-table::
   :header-rows: 0

   * - :meth:`~pybbtd.covll1.CovLL1.fit`
     - Fit a Cov-LL1 model to the given data using the specified algorithm (default: ADMM).
   * - :meth:`~pybbtd.covll1.CovLL1.generate_covll1_tensor`
     - Generate a random covariance imaging tensor and store the factors.
   * - :meth:`~pybbtd.covll1.CovLL1.get_constraint_matrix`
     - Return the constraint matrix for the CP-equivalent Cov-LL1 model.

Module Functions
------------------

.. list-table::
   :header-rows: 0

   * - :func:`~pybbtd.covll1.generate_covll1_factors`
     - Generate random non-negative spatial factors and vectorized covariance columns.
   * - :func:`~pybbtd.covll1.validate_cov_matrices`
     - Check if all covariance matrices in a tensor are valid (Hermitian, PSD).
   * - :func:`~pybbtd.covll1.is_valid_covariance`
     - Check if a single matrix is a valid covariance matrix.
   * - :func:`~pybbtd.covll1._validate_dimensions`
     - Validate dimensions and parameters for the Cov-LL1 model.

.. automodule:: pybbtd.covll1
   :members:
   :undoc-members:
   :show-inheritance:
