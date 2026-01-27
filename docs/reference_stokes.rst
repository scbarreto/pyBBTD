.. _reference.stokes:

.. currentmodule:: pybbtd.stokes

Stokes Class
=============

Class Methods
---------------

.. list-table::
   :header-rows: 0

   * - :meth:`~pybbtd.stokes.Stokes.fit`
     - Fit a Stokes-BTD model to the given data using the specified algorithm (default: ADMM).
   * - :meth:`~pybbtd.stokes.Stokes.generate_stokes_tensor`
     - Generate a random Stokes tensor and store the factors.

Module Functions
------------------

.. list-table::
   :header-rows: 0

   * - :func:`~pybbtd.stokes.generate_stokes_factors`
     - Generate random non-negative spatial factors and valid Stokes columns.
   * - :func:`~pybbtd.stokes.validate_stokes_tensor`
     - Check if all Stokes vectors in a tensor satisfy the physical constraints.
   * - :func:`~pybbtd.stokes.check_stokes_constraints`
     - Check if a single Stokes vector satisfies the physical constraints.
   * - :func:`~pybbtd.stokes.stokes2coh`
     - Construct the 2x2 coherency matrix from a Stokes vector.
   * - :func:`~pybbtd.stokes.coh2stokes`
     - Compute the Stokes vector from a 2x2 coherency matrix.
   * - :func:`~pybbtd.stokes.proj_psd`
     - Project a matrix onto the set of positive semidefinite Hermitian matrices.
   * - :func:`~pybbtd.stokes.stokes_projection`
     - Project a Stokes vector onto the set of physically valid Stokes vectors.

.. automodule:: pybbtd.stokes
   :members:
   :undoc-members:
   :show-inheritance:
