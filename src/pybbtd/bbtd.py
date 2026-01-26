import numpy as np
from scipy.linalg import block_diag
from tensorly.cp_tensor import cp_to_tensor


class BBTD:
    """
    Class for tensors admitting a Block-Block Term Decomposition (BBTD) into rank-(L1, L2, 1, 1) terms.

    :param dims: Dimensions `(I, J, K, M)` of the 4D tensor.
    :type dims: Tuple[int, int, int, int]
    :param R: The rank of the decomposition (number of components).
    :type R: int
    :param L1: Rank of the first block (first mode).
    :type L1: int
    :param L2: Rank of the second block (second mode).
    :type L2: int

    :ivar A: Factor matrix of shape `(dims[0], L1 * R)`.
    :vartype A: np.ndarray
    :ivar B: Factor matrix of shape `(dims[1], L1 * R)`.
    :vartype B: np.ndarray
    :ivar C: Factor matrix of shape `(dims[2], L2 * R)`.
    :vartype C: np.ndarray
    :ivar D: Factor matrix of shape `(dims[3], L2 * R)`.
    :vartype D: np.ndarray
    """

    def __init__(self, dims, R: int, L1: int, L2: int):
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

        # Initialize additional variables
        self.factors = None
        self.tensor = None
        self.fit_error = None

    def get_constraint_matrices(self):
        """
        Get the constraint matrices phi and psi.

        Returns:
            phi: Constraint matrix for the first block (A, B), shape (L1 * L2 * R, L1 * R)
            psi: Constraint matrix for the second block (C, D), shape (L2 * R, L1 * L2 * R)
        """
        return _constraint_matrices(self.L1, self.L2, self.rank)

    def fit(self, data, algorithm="ALS", **kwargs):
        """
        Fit a BBTD to the given data using the specified algorithm.

        Parameters:
            data: np.ndarray
                The input 4D tensor data to be decomposed.
            algorithm: str
                The algorithm to use for fitting. Options: "ALS", "ADMM".
            **kwargs:
                Additional keyword arguments passed to the solver.

        Returns:
            None. Updates self.factors, self.tensor, and self.fit_error.
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
    """Check if R, L1, and L2 are positive integers."""
    if not isinstance(R, int) or R <= 0:
        raise ValueError("R should be a positive integer.")
    if not isinstance(L1, int) or L1 <= 0:
        raise ValueError("L1 should be a positive integer.")
    if not isinstance(L2, int) or L2 <= 0:
        raise ValueError("L2 should be a positive integer.")

    return R, L1, L2


def _repeat_list_nested(input_list, n):
    """Repeat a list n times as nested elements."""
    return [input_list for _ in range(n)]


def _constraint_matrices(L1, L2, R):
    """
    Compute the constraint matrices phi and psi for the BBTD model.

    Parameters:
        L1: Rank of the first block (used by A and B)
        L2: Rank of the second block (used by C and D)
        R: Rank of the decomposition (number of terms)

    Returns:
        phi: Constraint matrix for the first block (A, B)
        psi: Constraint matrix for the second block (C, D)
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
    Convert BBTD factor matrices to full 4D tensor.

    Parameters:
        A: np.ndarray
            Factor matrix of shape (I, L1 * R).
        B: np.ndarray
            Factor matrix of shape (J, L1 * R).
        C: np.ndarray
            Factor matrix of shape (K, L2 * R).
        D: np.ndarray
            Factor matrix of shape (M, L2 * R).
        phi: np.ndarray
            Constraint matrix for the first block (A, B).
        psi: np.ndarray
            Constraint matrix for the second block (C, D).

    Returns:
        full_tensor: np.ndarray
            Reconstructed 4D tensor.
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
