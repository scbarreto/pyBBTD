"""Block-block term decomposition (BBTD) of fourth-order tensors."""

import numpy as np
from scipy.linalg import block_diag
from tensorly.cp_tensor import cp_to_tensor

from pybbtd._utils import check_dims, check_positive_int, unknown_option_error
from pybbtd.uniqueness import _report_uniqueness, check_uniqueness_bbtd

ALGORITHMS = ("ALS", "ADMM")


class BBTD:
    """Fourth-order tensor model made of R rank-(L1, L1, L2, L2) terms.

    A tensor of shape ``(I, J, K, M)`` is written as

    .. math::

        \\mathcal{T} = \\sum_{r=1}^{R} (\\mathbf{A}_r \\mathbf{B}_r^{\\top})
        \\circ (\\mathbf{C}_r \\mathbf{D}_r^{\\top}),

    where :math:`\\mathbf{A}_r \\mathbf{B}_r^{\\top}` is a rank-L1 matrix of
    shape ``(I, J)`` (for imaging data, a spatial map) and
    :math:`\\mathbf{C}_r \\mathbf{D}_r^{\\top}` is a rank-L2 matrix of shape
    ``(K, M)``. For covariance imaging, ``K = M`` and
    :math:`\\mathbf{D}_r = \\mathbf{C}_r^*`, so the second block is a
    covariance matrix :math:`\\mathbf{C}_r \\mathbf{C}_r^{H}`.

    Parameters
    ----------
    dims : tuple of int
        Shape ``(I, J, K, M)`` of the tensor.
    R : int
        Number of terms.
    L1 : int
        Rank of the first block (shared by A and B).
    L2 : int
        Rank of the second block (shared by C and D).

    Attributes
    ----------
    dims : tuple of int
        Shape of the tensor.
    rank : int
        Number of terms R.
    L1, L2 : int
        Block ranks.
    factors : list of np.ndarray or None
        Factor matrices ``[A, B, C, D]`` estimated by :meth:`fit`.
    tensor : np.ndarray or None
        Tensor rebuilt from ``factors`` by :meth:`fit`.
    fit_error : np.ndarray or None
        Fit error at each iteration of the last :meth:`fit`.
    """

    def __init__(self, dims, R, L1, L2):
        self.dims = check_dims(dims, 4)
        self.rank = check_positive_int(R, "R")
        self.L1 = check_positive_int(L1, "L1")
        self.L2 = check_positive_int(L2, "L2")

        self.check_uniqueness()

        self.factors = None
        self.tensor = None
        self.fit_error = None

    def __repr__(self):
        return f"BBTD(dims={self.dims}, R={self.rank}, L1={self.L1}, L2={self.L2})"

    def check_uniqueness(self):
        """Check a sufficient condition for the decomposition to be unique.

        A warning is issued when uniqueness cannot be guaranteed.

        Returns
        -------
        bool
            ``True`` if uniqueness is guaranteed.
        """
        unique = check_uniqueness_bbtd(*self.dims, self.rank, self.L1, self.L2)
        _report_uniqueness(unique)
        return unique

    def get_constraint_matrices(self):
        """Return the matrices :math:`\\Phi` and :math:`\\Psi` that write the BBTD as a CP model.

        With them, the BBTD equals a CP decomposition with ``L1 * L2 * R``
        rank-one terms and factors ``A @ phi``, ``B @ phi``, ``C @ psi`` and
        ``D @ psi``.

        Returns
        -------
        phi : np.ndarray
            Shape ``(L1 * R, L1 * L2 * R)``.
        psi : np.ndarray
            Shape ``(L2 * R, L1 * L2 * R)``.
        """
        return _constraint_matrices(self.rank, self.L1, self.L2)

    def fit(self, data, algorithm="ALS", **kwargs):
        """Estimate the factors that best explain ``data``.

        Updates :attr:`factors`, :attr:`tensor` and :attr:`fit_error`.

        Parameters
        ----------
        data : np.ndarray
            Tensor of shape :attr:`dims` to decompose.
        algorithm : {"ALS", "ADMM"}, default "ALS"
            ``"ALS"`` fits an unconstrained BBTD
            (:func:`~pybbtd.solvers.bbtd_als.fit_bbtd_als`). ``"ADMM"`` fits
            the covariance model, with non-negative spatial maps and
            ``D = C*`` (:func:`~pybbtd.solvers.bbtd_cov_admm.fit_bbtd_cov_admm`).
        **kwargs
            Passed to the solver, e.g. ``init``, ``max_iter``, ``rel_tol``,
            ``abs_tol``.

        Returns
        -------
        BBTD
            The fitted model.
        """
        if algorithm == "ALS":
            from pybbtd.solvers.bbtd_als import fit_bbtd_als

            self.factors, self.fit_error = fit_bbtd_als(self, data, **kwargs)
        elif algorithm == "ADMM":
            from pybbtd.solvers.bbtd_cov_admm import fit_bbtd_cov_admm

            self.factors, self.fit_error = fit_bbtd_cov_admm(self, data, **kwargs)
        else:
            raise unknown_option_error("algorithm", algorithm, ALGORITHMS)

        phi, psi = self.get_constraint_matrices()
        self.tensor = factors_to_tensor(*self.factors, phi, psi)
        return self


def _constraint_matrices(R, L1, L2):
    """Build the block-diagonal constraint matrices :math:`\\Phi` and :math:`\\Psi`.

    :math:`\\Phi` repeats each column of A and B ``L2`` times, and
    :math:`\\Psi` tiles each block of C and D ``L1`` times, so that every
    pair of columns from the two blocks forms one CP term.
    """
    # Phi: L1 * R rows of L2 ones
    phi = block_diag(*[np.ones(L2)] * (L1 * R))

    # Psi: for each term, L1 copies of the L2 x L2 identity side by side
    psi = block_diag(*[np.tile(np.eye(L2), L1)] * R)

    return phi, psi


def factors_to_tensor(A, B, C, D, phi, psi):
    """Rebuild the full fourth-order tensor from BBTD factors.

    Parameters
    ----------
    A : np.ndarray
        Factor of shape ``(I, L1 * R)``.
    B : np.ndarray
        Factor of shape ``(J, L1 * R)``.
    C : np.ndarray
        Factor of shape ``(K, L2 * R)``.
    D : np.ndarray
        Factor of shape ``(M, L2 * R)``.
    phi, psi : np.ndarray
        Constraint matrices from :meth:`BBTD.get_constraint_matrices`.

    Returns
    -------
    np.ndarray
        Tensor of shape ``(I, J, K, M)``.
    """
    weights = np.ones(phi.shape[1])
    return cp_to_tensor((weights, [A @ phi, B @ phi, C @ psi, D @ psi]))
