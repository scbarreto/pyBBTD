import numpy as np
from scipy.linalg import block_diag
from tensorly.cp_tensor import cp_to_tensor
from pybbtd.uniqueness import check_uniqueness_BBTD


class BBTD:
    """
    Class for tensors admitting a Block-Block Term Decomposition (BBTD)
    into rank-:math:`(L_1, L_1, L_2, L_2)` terms.

    A BBTD decomposes a 4-D tensor :math:`\\mathcal{T} \\in
    \\mathbb{R}^{I \\times J \\times K \\times M}` as a sum of :math:`R`
    block terms, each of which is a rank-:math:`(L_1, L_1, L_2, L_2)`
    term built from four factor matrices A, B, C, D and two constraint
    matrices :math:`\\Phi` and :math:`\\Psi`.

    :param dims: Dimensions ``(I, J, K, M)`` of the 4-D tensor.
    :type dims: list[int] or Tuple[int, int, int, int]
    :param R: Number of block terms (rank of the decomposition).
    :type R: int
    :param L1: Rank of the spatial block (shared by modes A and B).
    :type L1: int
    :param L2: Rank of the spectral block (shared by modes C and D).
    :type L2: int

    :ivar factors: List of factor matrices ``[A, B, C, D]`` after fitting,
        or ``None`` before fitting.
    :vartype factors: list[np.ndarray] or None
    :ivar tensor: Reconstructed tensor after fitting, or ``None`` before
        fitting.
    :vartype tensor: np.ndarray or None
    :ivar fit_error: Array of fitting errors at each iteration, or ``None``
        before fitting.
    :vartype fit_error: np.ndarray or None
    """

    def __init__(self, dims, R: int, L1: int, L2: int):
        """
        Initialize a BBTD model.

        Parameters:
            dims: list[int] or Tuple[int, int, int, int]
                Dimensions ``(I, J, K, M)`` of the 4-D tensor.  Must contain
                exactly four positive integers.
            R: int
                Number of block terms (rank of the decomposition).
            L1: int
                Rank of the spatial block (shared by modes A and B).
            L2: int
                Rank of the spectral block (shared by modes C and D).

        Raises:
            ValueError: If ``dims`` is not a list/tuple of four positive
                integers, or if R, L1, L2 are not positive integers.
        """
        # Validate dims
        if (
            not isinstance(dims, (list, tuple))
            or len(dims) != 4
            or not all(isinstance(d, int) and d > 0 for d in dims)
        ):
            raise ValueError(
                "dims should be a list or tuple of four positive integers."
            )
        self.dims = tuple(dims)

        # Validate R, L1, L2
        self.rank, self.L1, self.L2 = _validate_R_L1_L2(R, L1, L2)

        # check uniqueness, raise warning if parameters cannot be guaranteed to lead to unique solution
        self.check_uniqueness()

        # Initialize additional variables
        self.factors = None
        self.tensor = None
        self.fit_error = None

    def check_uniqueness(self):
        """
        Check if sufficient conditions for essential uniqueness are satisfied.

        Uses the algebraic conditions from the literature for the BBTD model.
        Prints whether uniqueness can be guaranteed or not.
        """
        N1, N2, N3, N4 = self.dims
        unique = check_uniqueness_BBTD(N1, N2, N3, N4, self.rank, self.L1, self.L2)
        if unique:
            print("Sufficient condition for uniqueness satisfied")
        else:
            print("Cannot guarantee uniqueness. Proceed at your own risk.")

    def get_constraint_matrices(self):
        """
        Return the constraint matrices :math:`\\Phi` and :math:`\\Psi` for
        the CP-equivalent BBTD model.

        :math:`\\Phi` maps the spatial factors (A, B) and :math:`\\Psi`
        maps the spectral factors (C, D) so that the BBTD can be
        expressed as a CP decomposition.

        Returns:
            Tuple[np.ndarray, np.ndarray]:
                ``(phi, psi)`` where ``phi`` has shape
                ``(L1 * L2 * R, L1 * R)`` and ``psi`` has shape
                ``(L2 * R, L1 * L2 * R)``.
        """
        return _constraint_matrices(self.L1, self.L2, self.rank)

    def fit(self, data, algorithm="ALS", **kwargs):
        """
        Fit a BBTD model to the given data using the specified algorithm.

        After fitting, ``self.factors``, ``self.tensor``, and
        ``self.fit_error`` are updated in place.

        Parameters:
            data: np.ndarray
                Input 4-D tensor of shape matching ``self.dims``.
            algorithm: str
                Algorithm to use for fitting (default: ``"ALS"``).
                Options: ``"ALS"`` (vanilla Alternating Least Squares),
                ``"ADMM"`` (constrained AO-ADMM with non-negativity on
                A, B and conjugate symmetry D = C*).
            **kwargs:
                Additional keyword arguments passed to the solver
                (e.g. ``init``, ``max_iter``, ``rel_tol``, ``abs_tol``).

        Raises:
            NotImplementedError: If ``algorithm`` is not ``"ALS"`` or
                ``"ADMM"``.
        """
        if algorithm == "ALS":
            from pybbtd.solvers.bbtd_vanilla_als import BBTD_ALS

            self.factors, self.fit_error = BBTD_ALS(self, data, **kwargs)
        elif algorithm == "ADMM":
            from pybbtd.solvers.bbtd_cov_admm import BBTD_COV_ADMM

            self.factors, self.fit_error = BBTD_COV_ADMM(self, data, **kwargs)
        else:
            raise NotImplementedError(f"Algorithm '{algorithm}' not implemented yet.")

        phi, psi = self.get_constraint_matrices()
        self.tensor = factors_to_tensor(*self.factors, phi, psi)


def _validate_R_L1_L2(R, L1, L2):
    """
    Validate that R, L1, and L2 are positive integers.

    Parameters:
        R: int
            Number of block terms.
        L1: int
            Rank of the spatial block.
        L2: int
            Rank of the spectral block.

    Returns:
        Tuple[int, int, int]:
            Validated ``(R, L1, L2)``.

    Raises:
        ValueError: If any argument is not a positive integer.
    """
    if not isinstance(R, int) or R <= 0:
        raise ValueError("R should be a positive integer.")
    if not isinstance(L1, int) or L1 <= 0:
        raise ValueError("L1 should be a positive integer.")
    if not isinstance(L2, int) or L2 <= 0:
        raise ValueError("L2 should be a positive integer.")

    return R, L1, L2


def _repeat_list_nested(input_list, n):
    """
    Repeat a list *n* times as nested elements.

    Parameters:
        input_list: list
            The list to repeat.
        n: int
            Number of repetitions.

    Returns:
        list:
            A list of *n* references to ``input_list``.
    """
    return [input_list for _ in range(n)]


def _constraint_matrices(L1, L2, R):
    """
    Compute the block-diagonal constraint matrices :math:`\\Phi` and
    :math:`\\Psi` for the BBTD model.

    :math:`\\Phi` expands the spatial factors (A, B) and :math:`\\Psi`
    selects the spectral factors (C, D) so that the BBTD can be written
    as a standard CP decomposition.

    Parameters:
        L1: int
            Rank of the spatial block (shared by A and B).
        L2: int
            Rank of the spectral block (shared by C and D).
        R: int
            Number of block terms.

    Returns:
        Tuple[np.ndarray, np.ndarray]:
            ``(phi, psi)`` where ``phi`` has shape
            ``(L1 * L2 * R, L1 * R)`` and ``psi`` has shape
            ``(L2 * R, L1 * L2 * R)``.
    """
    # Phi: constraint matrix for the first block (A, B)
    onesL2 = np.ones(L2)
    listL2 = _repeat_list_nested(onesL2, L1)
    phir = block_diag(*listL2)
    listphi = _repeat_list_nested(phir, R)
    phi = block_diag(*listphi)

    # Psi: constraint matrix for the second block (C, D)
    identityL2 = np.eye(L2)
    listidentities2 = _repeat_list_nested(identityL2, L1 - 1)
    psir = np.concatenate((identityL2, *listidentities2), axis=1)
    listpsi = _repeat_list_nested(psir, R)
    psi = block_diag(*listpsi)

    return phi, psi


def factors_to_tensor(A, B, C, D, phi, psi):
    """
    Reconstruct a full 4-D tensor from BBTD factor matrices and
    constraint matrices.

    Computes ``A @ phi``, ``B @ phi``, ``C @ psi``, ``D @ psi`` and
    assembles the CP-equivalent tensor.

    Parameters:
        A: np.ndarray
            Factor matrix of shape ``(I, L1 * R)``.
        B: np.ndarray
            Factor matrix of shape ``(J, L1 * R)``.
        C: np.ndarray
            Factor matrix of shape ``(K, L2 * R)``.
        D: np.ndarray
            Factor matrix of shape ``(M, L2 * R)``.
        phi: np.ndarray
            Constraint matrix for the spatial block (A, B).
        psi: np.ndarray
            Constraint matrix for the spectral block (C, D).

    Returns:
        np.ndarray:
            Reconstructed 4-D tensor of shape ``(I, J, K, M)``.
    """
    APhi = A @ phi
    BPhi = B @ phi
    CPsi = C @ psi
    DPsi = D @ psi
    fac_matrices = [np.array(APhi), np.array(BPhi), np.array(CPsi), np.array(DPsi)]
    weights = np.ones(phi.shape[1])
    cp_tensor_repr = (weights, fac_matrices)
    full_tensor = cp_to_tensor(cp_tensor_repr)

    return full_tensor
