"""Block term decomposition (BTD) of third-order tensors into rank-(L, L, 1) terms."""

from numbers import Integral

import numpy as np
from scipy.linalg import block_diag
from tensorly.cp_tensor import cp_to_tensor

from pybbtd._utils import check_dims, check_positive_int, unknown_option_error
from pybbtd.uniqueness import _report_uniqueness, check_uniqueness_ll1

BLOCK_MODES = ("LL1", "L1L", "1LL")


class BTD:
    """Third-order tensor model made of R rank-(L, L, 1) block terms.

    Each term is the outer product of a rank-L matrix and a vector. With the
    default ``"LL1"`` block mode, a tensor of shape ``(I, J, K)`` is written as

    .. math::

        \\mathcal{T} = \\sum_{r=1}^{R} (\\mathbf{A}_r \\mathbf{B}_r^{\\top})
        \\circ \\mathbf{c}_r,

    with :math:`\\mathbf{A}_r` of shape ``(I, L)``, :math:`\\mathbf{B}_r` of
    shape ``(J, L)`` and :math:`\\mathbf{c}_r` of length ``K``. The modes
    ``"L1L"`` and ``"1LL"`` put the vector in the second or first mode instead.

    Parameters
    ----------
    dims : tuple of int
        Shape ``(I, J, K)`` of the tensor.
    R : int
        Number of block terms.
    L : int or sequence of int
        Rank of the matrix part of each term. A single integer is used for
        all R terms.
    block_mode : {"LL1", "L1L", "1LL"}, default "LL1"
        Which mode holds the vector part of each term.

    Attributes
    ----------
    dims : tuple of int
        Shape of the tensor.
    rank : int
        Number of block terms R.
    L : np.ndarray
        Rank of each term, shape ``(R,)``.
    block_mode : str
        Block mode of the decomposition.
    factors : list of np.ndarray or None
        Factor matrices ``[A, B, C]`` estimated by :meth:`fit`.
    tensor : np.ndarray or None
        Tensor rebuilt from ``factors`` by :meth:`fit`.
    fit_error : np.ndarray or None
        Fit error at each iteration of the last :meth:`fit`.
    """

    def __init__(self, dims, R, L, block_mode="LL1"):
        self.dims = check_dims(dims, 3)
        self.rank, self.L = _validate_ranks(R, L)
        if block_mode not in BLOCK_MODES:
            raise ValueError(
                f"Invalid mode '{block_mode}'. Block mode must be one of {BLOCK_MODES}."
            )
        self.block_mode = block_mode

        self.check_uniqueness()

        self.factors = None
        self.tensor = None
        self.fit_error = None

    def __repr__(self):
        return (
            f"{type(self).__name__}(dims={self.dims}, R={self.rank}, "
            f"L={self.L.tolist()}, block_mode='{self.block_mode}')"
        )

    def check_uniqueness(self):
        """Check a sufficient condition for the decomposition to be unique.

        A warning is issued when uniqueness cannot be guaranteed.

        Returns
        -------
        bool or None
            ``True`` if uniqueness is guaranteed, ``False`` if it cannot be
            guaranteed, and ``None`` if the check is not available (it is
            only implemented when all terms share the same L).
        """
        if not np.all(self.L == self.L[0]):
            return None

        N1, N2, N3 = self.dims
        # the conditions are stated with the rank-one mode last
        if self.block_mode == "LL1":
            unique = check_uniqueness_ll1(N1, N2, N3, self.rank, self.L[0])
        elif self.block_mode == "L1L":
            unique = check_uniqueness_ll1(N1, N3, N2, self.rank, self.L[0])
        else:
            unique = check_uniqueness_ll1(N2, N3, N1, self.rank, self.L[0])

        _report_uniqueness(unique)
        return unique

    def fit(self, data, algorithm="ALS", **kwargs):
        """Estimate the factors that best explain ``data``.

        Updates :attr:`factors`, :attr:`tensor` and :attr:`fit_error`.

        Parameters
        ----------
        data : np.ndarray
            Tensor of shape :attr:`dims` to decompose.
        algorithm : {"ALS"}, default "ALS"
            Fitting algorithm (see :func:`~pybbtd.solvers.btd_als.fit_btd_als`).
        **kwargs
            Passed to the solver, e.g. ``init``, ``max_iter``, ``rel_tol``,
            ``abs_tol``.

        Returns
        -------
        BTD
            The fitted model.
        """
        if algorithm != "ALS":
            raise unknown_option_error("algorithm", algorithm, ["ALS"])

        from pybbtd.solvers.btd_als import fit_btd_als

        self.factors, self.fit_error = fit_btd_als(self, data, **kwargs)
        self.tensor = factors_to_tensor(
            *self.factors, self.get_constraint_matrix(), block_mode=self.block_mode
        )
        return self

    def get_constraint_matrix(self):
        """Return the matrix :math:`\\Theta` that writes the BTD as a CP model.

        :math:`\\Theta` repeats each column of the vector factor ``L[r]``
        times, so the BTD can be computed as a CP decomposition with
        ``sum(L)`` rank-one terms.

        Returns
        -------
        np.ndarray
            Constraint matrix of shape ``(R, sum(L))``.
        """
        return _constraint_matrix(self.rank, self.L)


def _validate_ranks(R, L):
    """Validate R and L, and return L as an integer array of length R.

    Parameters
    ----------
    R : int
        Number of block terms.
    L : int or sequence of int
        Rank of each term. A single integer is repeated R times.

    Returns
    -------
    R : int
    L : np.ndarray
        Integer array of length R.
    """
    R = check_positive_int(R, "R")

    if isinstance(L, Integral) and not isinstance(L, bool):
        L_array = np.ones(R, dtype=int) * L
    elif isinstance(L, (list, tuple, np.ndarray)) and len(L) == R:
        L_array = np.array(L)
    else:
        raise ValueError(
            "L should be either a single integer or a list/array of length R."
        )
    if np.any(L_array <= 0):
        raise ValueError("Each element in L should be greater than zero.")

    return R, L_array


def _constraint_matrix(R, L):
    """Build the block-diagonal matrix :math:`\\Theta` of shape ``(R, sum(L))``.

    Row ``r`` holds ``L[r]`` ones, so ``C @ theta`` repeats column ``r`` of
    ``C`` ``L[r]`` times.
    """
    _, L_array = _validate_ranks(R, L)
    return block_diag(*[np.ones(Lr) for Lr in L_array])


def factors_to_tensor(A, B, C, theta, block_mode="LL1"):
    """Rebuild the full tensor from BTD factors.

    Parameters
    ----------
    A, B, C : np.ndarray
        Factor matrices. With ``block_mode="LL1"``, ``A`` and ``B`` have
        ``sum(L)`` columns and ``C`` has ``R`` columns; the other modes move
        the R-column factor accordingly.
    theta : np.ndarray
        Constraint matrix from :meth:`BTD.get_constraint_matrix`.
    block_mode : {"LL1", "L1L", "1LL"}, default "LL1"
        Block mode of the decomposition.

    Returns
    -------
    np.ndarray
        Tensor of shape ``(A.shape[0], B.shape[0], C.shape[0])``.
    """
    weights = np.ones(theta.shape[1])
    if block_mode == "LL1":
        return cp_to_tensor((weights, [A, B, C @ theta]))
    if block_mode == "L1L":
        return cp_to_tensor((weights, [A, B @ theta, C]))
    if block_mode == "1LL":
        return cp_to_tensor((weights, [A @ theta, B, C]))
    raise unknown_option_error("block mode", block_mode, BLOCK_MODES)


def spatial_maps(A, B, L):
    """Return the matrix part :math:`\\mathbf{A}_r \\mathbf{B}_r^{\\top}` of each term.

    For imaging data these are the R spatial maps: images showing where
    each term is present.

    Parameters
    ----------
    A : np.ndarray
        Factor of shape ``(I, R * L)``, with the columns of term ``r`` in
        the ``r``-th block of ``L`` columns.
    B : np.ndarray
        Factor of shape ``(J, R * L)`` with the same block structure.
    L : int
        Number of columns per term.

    Returns
    -------
    np.ndarray
        Array of shape ``(R, I, J)``.
    """
    R = A.shape[1] // L
    return np.stack(
        [A[:, r * L : (r + 1) * L] @ B[:, r * L : (r + 1) * L].T for r in range(R)]
    )
