.. _reference.bbtd:

.. currentmodule:: pybbtd.bbtd

BBTD Class
=============

Class Methods
---------------

.. list-table::
   :header-rows: 0

   * - :meth:`~pybbtd.bbtd.BBTD.fit`
     - Fit a BBTD model to the given data using the specified algorithm (default: ALS).
   * - :meth:`~pybbtd.bbtd.BBTD.get_constraint_matrices`
     - Return the constraint matrices phi and psi for the CP-equivalent BBTD model.

Module Functions
------------------

.. list-table::
   :header-rows: 0

   * - :func:`~pybbtd.bbtd.factors_to_tensor`
     - Reconstruct a full 4-D tensor from BBTD factor matrices and constraint matrices.
   * - :func:`~pybbtd.bbtd._constraint_matrices`
     - Compute the block-diagonal constraint matrices phi and psi for the BBTD model.
   * - :func:`~pybbtd.bbtd._validate_R_L1_L2`
     - Validate that R, L1, and L2 are positive integers.

.. automodule:: pybbtd.bbtd
   :members:
   :undoc-members:
   :show-inheritance:
