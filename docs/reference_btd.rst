.. _reference.btd:

.. currentmodule:: pybbtd.btd

BTD Class
=============

Class Methods
---------------

.. list-table::
   :header-rows: 0

   * - :meth:`~pybbtd.btd.BTD.fit`
     - Fit a BTD model to the given data using the specified algorithm (default: ALS).
   * - :meth:`~pybbtd.btd.BTD.check_uniqueness`
     - Check if sufficient conditions for essential uniqueness are satisfied.
   * - :meth:`~pybbtd.btd.BTD.get_constraint_matrix`
     - Return the constraint matrix for the CP-equivalent BTD model.
   * - :meth:`~pybbtd.btd.BTD.to_cpd_format`
     - Convert the BTD to CPD format (not implemented yet).

Module Functions
------------------

.. list-table::
   :header-rows: 0

   * - :func:`~pybbtd.btd.factors_to_tensor`
     - Convert BTD factor matrices to a full tensor for a given block mode.
   * - :func:`~pybbtd.btd._constraint_matrix`
     - Compute the block-diagonal constraint matrix that maps BTD factors to CP-equivalent factors.
   * - :func:`~pybbtd.btd._validate_R_L`
     - Validate that R and L are positive integers (or a list of positive integers for L).

.. automodule:: pybbtd.btd
   :members:
   :undoc-members:
   :show-inheritance:
