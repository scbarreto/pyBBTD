from pybbtd.btd import BTD
import numpy as np
import pybbtd.btd as btd
from pybbtd.uniqueness import check_uniqueness_LL1
import warnings


class CovLL1(BTD):
    """
    Class for tensors admitting a Block Tensor Decomposition (BTD)-LL1, where each
    vector of size :math:`K^2` in the rank-one mode represents a valid covariance
    matrix of size :math:`K \\times K`, once reshaped.

    Parameters
    ----------
    dims : tuple[int, int, int]
        Dimensions :math:`(I, J, K^2)` of the tensor.
    R : int
        The rank of the decomposition (number of components).
    L1 : int
        Rank of the spatial maps.
    L2 : int
        Rank of the covariance matrices.

    Attributes
    ----------
    A : np.ndarray
        Factor matrix of shape ``(dims[0], L * R)``.
    B : np.ndarray
        Factor matrix of shape ``(dims[1], L * R)``.
    C : np.ndarray
        Factor matrix of shape ``(K**2, R)``, where each column is a vectorized
        covariance matrix.
    """

    def __init__(self, dims, R: int, L1: int, L2: int, block_mode="LL1"):
        """
        Initialize Cov-LL1 model.
        Parameters:
            spatial_dims: Tuple[int, int]
                Spatial dimensions of the tensor (first two modes).
            R: int
                Rank of the decomposition (number of components).
            L: int
                Rank of the spatial maps.
        """
        super().__init__(dims, R=R, L=L1, block_mode=block_mode)
        self.dims = tuple(dims)
        self.L1 = L1
        self.L2 = L2

        # Validate mode
        valid_block_modes = {"LL1", "L1L", "1LL"}
        if block_mode not in valid_block_modes:
            raise ValueError(
                f"Invalid mode '{block_mode}'. Block mode must be one of {valid_block_modes}."
            )
        self.block_mode = block_mode
        self.rank, self.L1, self.L2 = validate_dimensions(dims, R, L1, L2)

        print(
            f"Cov-LL1 tensor initialized with dimensions {self.dims} on {self.block_mode} mode."
        )

    def generate_covll1_tensor(self):
        """
        Generate a random covariance tensor.
        """

        A, B, C = generate_covll1_factors(
            self.dims, self.rank, self.L1, self.L2)
        # Generate the tensor using the factors
        self.factors = [A, B, C]
        self.tensor = btd.factors_to_tensor(
            A, B, C, self.get_constraint_matrix(), block_mode=self.block_mode
        )
        return self.factors, self.tensor

    def fit(self, data, algorithm="ADMM", **kwargs):
        """
        Computes Covll1-BTD factor matrices for provided data
        using the specified algorithm.
        Parameters:
            data: np.ndarray
                Input tensor data to be approximated.
            algorithm: str
                Algorithm to use for fitting. Currently, only "ADMM" is implemented.
            **kwargs: Additional keyword arguments for the fitting algorithm.
        """
        from pybbtd.solvers.covll1_admm import CovLL1_ADMM

        if algorithm == "ADMM":
            self.factors, self.fit_error = CovLL1_ADMM(self, data, **kwargs)
            self.tensor = btd.factors_to_tensor(
                *self.factors, self.get_constraint_matrix(), block_mode=self.block_mode
            )
        else:
            raise UserWarning("Algorithm not implemented yet")

    def get_constraint_matrix(self):
        """
        Returns constraint matrix for CovLL1 model (useful to CP-equivalent model)
        """
        return btd.constraint_matrix(self.rank, self.L)


def generate_covll1_factors(dims, R, L1, L2):
    """
    Generate random factors that follow Cov-LL1 decomposition.

    Args:
        dims (Tuple[int, int, int]):
            Dimensions :math:`(I, J, K^2)` of the tensor.
        R (int):
            The rank of the decomposition (number of components).
        L (int):
            Rank of the spatial maps.

    Returns:
        Tuple[np.ndarray, np.ndarray, np.ndarray]:
            A tuple containing the three factor matrices `(A, B, C)`:

            - A (np.ndarray): Shape `(dims[0], L * R)`.
            - B (np.ndarray): Shape `(dims[1], L * R)`.
            - C (np.ndarray): Shape `(dims[2]^2, R)`, where each column is a vectorized covariance matrix
    """

    if len(dims) != 3:
        raise ValueError("Number of dimensions should be equal to 3.")

    A = np.random.rand(dims[0], L1 * R)
    B = np.random.rand(dims[1], L1 * R)

    K = int(np.sqrt(dims[2]))

    cov_factors = np.random.randn(R, K, L2) + 1j * np.random.randn(R, K, L2)
    sigmas = np.zeros((K, K, R), dtype=complex)

    for r in range(R):
        sigmas[:, :, r] = cov_factors[r, :, :] @ cov_factors[r, :, :].conj().T
    C = sigmas.reshape(K**2, R)

    return A, B, C


def validate_dimensions(dims, R, L1, L2):
    """
    Validate dimensions for Cov-LL1 model.

    Args:
        dims (Tuple[int, int, int]):
            Dimensions :math:`(I, J, K^2)` of the tensor.
        R (int):
            The rank of the decomposition (number of components).
        L1 (int):
            Rank of the spatial maps.
        L2 (int):
            Rank of the covariance matrices.
    """
    if len(dims) != 3:
        raise ValueError("dims must be a tuple of three integers (I, J, K^2).")
    if dims[2] <= 0 or int(np.sqrt(dims[2])) ** 2 != dims[2]:
        raise ValueError("The third dimension must be a perfect square (K^2).")
    if R <= 0:
        raise ValueError("R must be a positive integer.")
    if L1 <= 0:
        raise ValueError("L1 must be a positive integer.")
    if L2 <= 0:
        raise ValueError("L2 must be a positive integer.")
    if L1 > dims[0] or L1 > dims[1]:
        raise ValueError(
            "L1 must be less than the first and second dimensions (I and J)."
        )
    if L2 > int(np.sqrt(dims[2])):
        raise ValueError(
            "L2 must be less than the square root of the third dimension (K)."
        )

    return R, L1, L2


def check_uniqueness(self):
    """
    Check if, for given parameters, uniqueness can be guaranteed.
    """

    N1, N2, N3 = self.dims

    if self.block_mode == "LL1":
        unique = check_uniqueness_LL1(N1, N2, N3, self.R, self.L1)
    elif self.block_mode == "L1L":
        unique = check_uniqueness_LL1(N1, N3, N2, self.R, self.L1)
    elif self.block_mode == "1LL":
        unique = check_uniqueness_LL1(N2, N3, N1, self.R, self.L1)
    if unique is True:
        print("Sufficient condition for uniqueness satisfied")
    else:
        print("Cannot guarantee uniqueness. Proceed at your own risk.")


def validate_cov_matrices(T0):
    """
    Check if all covariance matrices satisfy the positive semidefinite constraint.
    """
    invalid_count = 0
    total_pixels = T0.shape[0] * T0.shape[1]

    for i in range(T0.shape[0]):
        for j in range(T0.shape[1]):
            cov_matrix = T0[i, j, :].reshape(
                int(np.sqrt(T0.shape[2])), int(np.sqrt(T0.shape[2]))
            )
            if is_valid_covariance(cov_matrix) == 0:
                invalid_count += 1

    if invalid_count > 0:
        percentage = (invalid_count / total_pixels) * 100
        warnings.warn(
            f"{percentage:.2f}% of pixels do not satisfy the covariance matrix constraints. (must be square, Hermitian, and positive semidefinite)",
            UserWarning,
        )
    else:
        print("All pixels carry valid covariance matrices.")


def is_valid_covariance(Sigma, tol=1e-10):
    """
    Check if a matrix is a valid covariance matrix.

    Conditions:
    1. Square
    2. Hermitian / symmetric
    3. Positive semidefinite (all eigenvalues >= -tol)

    Parameters
    ----------
    Sigma : np.ndarray
        Matrix to test.
    tol : float, optional
        Numerical tolerance for eigenvalues and symmetry (default 1e-10).

    Returns
    -------
    bool
        True if Sigma is a valid covariance matrix.
    """
    Sigma = np.asarray(Sigma)

    # 1. Must be square
    if Sigma.ndim != 2 or Sigma.shape[0] != Sigma.shape[1]:
        return False

    # 2. Must be Hermitian (Sigma == Sigmaᴴ within tolerance)
    if not np.allclose(Sigma, Sigma.conj().T, atol=tol):
        return False

    # 3. Must be positive semidefinite
    eigvals = np.linalg.eigvalsh(Sigma)  # for Hermitian matrices
    if np.any(eigvals < -tol):
        return False

    return True
