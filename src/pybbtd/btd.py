import numpy as np
from pybbtd.uniqueness import check_uniqueness_LL1
from scipy.linalg import block_diag
from tensorly.cp_tensor import cp_to_tensor
from pybbtd.solvers.btd_als import BTD_ALS


class BTD:
    """
    Class for tensors admitting a Block Term Decomposition (BTD) into
    rank-(L, L, 1) terms.

    The decomposition writes a third-order tensor as a sum of R block terms,
    each formed by the outer product of a rank-L matrix and a vector. Three
    block modes are supported: ``"LL1"``, ``"L1L"``, and ``"1LL"``, indicating
    which mode carries the rank-one structure.

    :param dims: Dimensions ``(I, J, K)`` of the tensor.
    :type dims: Tuple[int, int, int]
    :param R: Number of block terms (rank of the decomposition).
    :type R: int
    :param L: Rank of the block in each term. Can be a single integer
        (same rank for all terms) or a list of R integers.
    :type L: int or list[int]
    :param block_mode: Block mode of the decomposition.
        One of ``"LL1"``, ``"L1L"``, ``"1LL"`` (default: ``"LL1"``).
    :type block_mode: str

    :ivar factors: List of factor matrices ``[A, B, C]`` after fitting,
        or ``None`` before fitting.
    :vartype factors: list[np.ndarray] or None
    :ivar tensor: Reconstructed tensor after fitting, or ``None`` before fitting.
    :vartype tensor: np.ndarray or None
    :ivar fit_error: Array of fitting errors at each iteration, or ``None``
        before fitting.
    :vartype fit_error: np.ndarray or None
    """

    def __init__(self, dims, R: int, L: int, block_mode="LL1"):
        # Validate dims
        if (
            not isinstance(dims, (list, tuple))
            or len(dims) != 3
            or not all(isinstance(d, int) and d > 0 for d in dims)
        ):
            raise ValueError(
                "dims should be a list or tuple of three positive integers."
            )
        self.dims = tuple(dims)

        # validate R, L values
        self.rank, self.L = _validate_R_L(R, L)

        # Validate mode
        valid_block_modes = {"LL1", "L1L", "1LL"}
        if block_mode not in valid_block_modes:
            raise ValueError(
                f"Invalid mode '{block_mode}'. Block mode must be one of {valid_block_modes}."
            )
        self.block_mode = block_mode

        # check uniqueness, raise warning if parameters cannot be guaranteed to lead to unique solution
        self.check_uniqueness()

        # Initialize additional variables
        self.factors = None
        self.tensor = None
        self.fit_error = None  # stores last fit error

    def check_uniqueness(self):
        """
        Check if sufficient conditions for essential uniqueness are satisfied.

        Uses the algebraic conditions from the literature (constant-L case only).
        Prints whether uniqueness can be guaranteed or not. When L varies across
        terms, the check is skipped with a message.
        """
        if not np.all(self.L == self.L[0]):
            return print(
                "Uniqueness checks are currently implemented for the constant L case.\nSkipping uniqueness tests."
            )

        N1, N2, N3 = self.dims

        if self.block_mode == "LL1":
            unique = check_uniqueness_LL1(
                N1, N2, N3, self.rank, self.L[0]
            )  # should fix value of L
        elif self.block_mode == "L1L":
            unique = check_uniqueness_LL1(
                N1, N3, N2, self.rank, self.L[0]
            )  # should fix value of L
        elif self.block_mode == "1LL":
            unique = check_uniqueness_LL1(
                N2, N3, N1, self.rank, self.L[0]
            )  # should fix value of L
        if unique is True:
            print("Sufficient condition for uniqueness satisfied")
        else:
            print("Cannot guarantee uniqueness. Proceed at your own risk.")

    def fit(self, data, algorithm="ALS", **kwargs):
        """
        Fit a BTD model to the given data using the specified algorithm.

        After fitting, ``self.factors``, ``self.tensor``, and ``self.fit_error``
        are updated in place.

        Parameters:
            data: np.ndarray
                Input tensor of shape ``dims`` to be decomposed.
            algorithm: str
                Algorithm to use for fitting (default: ``"ALS"``).
            **kwargs:
                Additional keyword arguments passed to the solver
                (e.g. ``init``, ``max_iter``, ``rel_tol``, ``abs_tol``).
        """
        if algorithm == "ALS":
            self.factors, self.fit_error = BTD_ALS(self, data, **kwargs)
            self.tensor = factors_to_tensor(
                *self.factors, self.get_constraint_matrix(), block_mode=self.block_mode
            )
        else:
            raise UserWarning("Algorithm not implemented yet")

    def get_constraint_matrix(self):
        """
        Return the constraint matrix for the CP-equivalent BTD model.

        The constraint matrix :math:`\\Theta` maps the R columns of the
        rank-one factor to the ``sum(L)`` columns of the block factors,
        enabling a CP-like representation of the BTD.

        Returns:
            np.ndarray:
                Constraint matrix of shape ``(R, sum(L))``.
        """
        return _constraint_matrix(self.rank, self.L)


def _validate_R_L(R, L):
    """
    Validate that R and L are positive integers.

    If L is a single integer it is broadcast to a length-R array.
    If L is a list or array it must have length R with all positive entries.

    Parameters:
        R: int
            Number of block terms (must be a positive integer).
        L: int or list[int] or np.ndarray
            Rank of each block term. A single integer is replicated R times.

    Returns:
        Tuple[int, np.ndarray]:
            Validated ``(R, L)`` where L is always a 1-D integer array of
            length R.
    """
    # check R
    if not isinstance(R, int) or R <= 0:
        raise ValueError("R should be a positive integer.")

    # Validate L
    if isinstance(L, int):
        Larray = np.ones(R, dtype=int) * L
    elif isinstance(L, (list, np.ndarray)) and len(L) == R:
        Larray = np.array(L)
        if np.any(Larray <= 0):
            raise ValueError("Each element in L should be greater than zero.")
    else:
        raise ValueError(
            "L should be either a single integer or a list/array of length R."
        )

    return R, Larray


def _constraint_matrix(R, L):
    """
    Compute the constraint matrix for the BTD model.

    Builds a block-diagonal matrix :math:`\\Theta` of shape ``(R, sum(L))``
    where each block is a row of ones of length ``L[r]``. This matrix maps
    the CP-equivalent factors back to the BTD factors.

    Parameters:
        R: int
            Number of block terms.
        L: int or list[int]
            Rank of each block term. A single integer is replicated R times.

    Returns:
        np.ndarray:
            Constraint matrix :math:`\\Theta` of shape ``(R, sum(L))``.
    """
    _, Larray = _validate_R_L(R, L)
    theta = block_diag(*[np.ones(Lr) for Lr in Larray])

    return theta


def factors_to_tensor(A, B, C, theta, block_mode="LL1"):
    """
    Convert BTD factor matrices to a full tensor.

    Reconstructs the tensor from factors A, B, C and the constraint matrix
    :math:`\\Theta`, using the CP-equivalent representation for the given
    block mode.

    Parameters:
        A: np.ndarray
            First factor matrix.
        B: np.ndarray
            Second factor matrix.
        C: np.ndarray
            Third factor matrix.
        theta: np.ndarray
            Constraint matrix from :func:`_constraint_matrix`.
        block_mode: str
            Block mode of the decomposition. One of ``"LL1"`` (default),
            ``"L1L"``, ``"1LL"``.

    Returns:
        np.ndarray:
            Reconstructed tensor of shape ``(I, J, K)``.
    """
    if block_mode == "LL1":
        Trec = cp_to_tensor((np.ones(theta.shape[1]), [A, B, C @ theta]))
    elif block_mode == "1LL":
        Trec = cp_to_tensor((np.ones(theta.shape[1]), [A @ theta, B, C]))
    elif block_mode == "L1L":
        Trec = cp_to_tensor((np.ones(theta.shape[1]), [A, B @ theta, C]))
    return Trec
