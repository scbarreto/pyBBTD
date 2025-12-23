.. _reference.stokes:

 .. currentmodule:: pybbtd.stokes

Stokes Class
=============

Core Methods
---------------
.. list-table::
   :header-rows: 0

   * - :meth:`~pybbtd.stokes.Stokes.fit`
     - Computes Stokes-BTD factor matrices for provided data using the specified algorithm.
   * - :meth:`~pybbtd.stokes.Stokes.generate_stokes_tensor`
     - Generates a random Stokes tensor.
   * - :meth:`~pybbtd.stokes.check_stokes_constraints`
     -  Check if a 4-D tensor satisfies the Stokes constraints..
   * - :meth:`~pybbtd.stokes.coh2stokes`
     - Returns Stokes parameters from polarization (coherency) matrix
   * - :meth:`~pybbtd.stokes.stokes2coh`
     - Returns polarization (coherency) matrix from Stokes parameters
   * - :meth:`~pybbtd.stokes.generate_stokes_factors`
     - Generates random factors that follow Stokes-BTD decomposition.
   * - :meth:`~pybbtd.stokes.projPSD`
     - Projection of matrix onto the set of PSD hermitian matrices.
   * - :meth:`~pybbtd.stokes.stokesProjection`
     - Projection of Stokes parameters onto the set of valid Stokes vectors.
   * - :meth:`~pybbtd.stokes.validate_stokes_tensor`
     - Checks if all Stokes vectors in a tensor are valid.

.. automodule:: pybbtd.stokes
   :members:
   :undoc-members:
   :show-inheritance:
