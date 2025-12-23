from pybbtd.btd import BTD
import numpy as np
import pybbtd.btd as btd

import warnings


class Stokes(BTD):
    """
    Class for tensors admitting a Block Tensor Decomposition (BTD)-LL1, where each vector in the rank-one mode respects the Stokes constraints.

    :param dims: Dimensions `(I, J, 4)` of the tensor.
    :type dims: Tuple[int, int, int]
    :param R: The rank of the decomposition (number of components).
    :type R: int
    :param L: Rank of the spatial maps.
    :type L: int

    :ivar A: Factor matrix of shape `(dims[0], L * R)`.
    :vartype A: np.ndarray
    :ivar B: Factor matrix of shape `(dims[1], L * R)`.
    :vartype B: np.ndarray
    :ivar C: Factor matrix of shape `(4, R)`, where each column is a Stokes vector.
    :vartype C: np.ndarray
    """

    def __init__(self, spatial_dims, R, L, **kwargs):
        """
        Initialize Stokes-BTD model.
        Parameters:
            spatial_dims: Tuple[int, int]
                Spatial dimensions of the tensor (first two modes).
            R: int
                Rank of the decomposition (number of components).
            L: int
                Rank of the spatial maps.
        """

        dims = (spatial_dims[0], spatial_dims[1], 4)
        kwargs["block_mode"] = "LL1"
        super().__init__(dims=dims, R=R, L=L, **kwargs)
        print(
            f"Stokes tensor initialized with dimensions {self.dims} on {self.block_mode} mode."
        )

    def generate_stokes_tensor(self):
        """
        Generate a random Stokes tensor.
        """

        A, B, C = generate_stokes_factors(self.dims, self.rank, self.L)
        # Generate the tensor using the factors
        self.factors = [A, B, C]
        self.tensor = btd.factors_to_tensor(
            A, B, C, self.get_constraint_matrix(), block_mode=self.block_mode
        )
        return self.factors, self.tensor

    def fit(self, data, algorithm="ADMM", **kwargs):
        """
        Computes Stokes-BTD factor matrices for provided data
        using the specified algorithm.

        Parameters:
            data: np.ndarray
                Input tensor data to be approximated.
            algorithm: str
                Algorithm to use for fitting. Currently, only "ADMM" is implemented.
            **kwargs: Additional keyword arguments for the fitting algorithm.
        """
        from pybbtd.solvers.stokes_admm import Stokes_ADMM

        if algorithm == "ADMM":
            self.factors, self.fit_error = Stokes_ADMM(self, data, **kwargs)
            self.tensor = btd.factors_to_tensor(
                *self.factors, self.get_constraint_matrix(), block_mode=self.block_mode
            )
        else:
            raise UserWarning("Algorithm not implemented yet")

    def _get_constraint_matrix(self):
        """
        Returns constraint matrix for BTD-LL1 model (useful to CP-equivalent model)
        """
        return btd.constraint_matrix(self.rank, self.L)


def generate_stokes_factors(dims, R, L):
    """
    Generate random factors that follow Stokes-BTD decomposition.

    Args:
        dims (Tuple[int, int, int]):
            Dimensions `(I, J, K)` of the tensor.
        R (int):
            The rank of the decomposition (number of components).
        L (int):
            Rank of the spatial maps.

    Returns:
        Tuple[np.ndarray, np.ndarray, np.ndarray]:
            A tuple containing the three factor matrices `(A, B, C)`:

            - A (np.ndarray): Shape `(dims[0], L * R)`.
            - B (np.ndarray): Shape `(dims[1], L * R)`.
            - C (np.ndarray): Shape `(dims[2], R)`, where each column is a Stokes vector.
    """

    A = np.random.rand(dims[0], L[0] * R)
    B = np.random.rand(dims[1], L[0] * R)
    C = np.zeros((dims[2], R))

    for r in range(R):
        cr = 1.0 / np.sqrt(2) * (np.random.randn(2) + 1j * np.random.randn(2))
        cr = cr / np.linalg.norm(cr)

        C[:, r] = coh2stokes(np.outer(cr, cr.conj()))

    return A, B, C


def validate_stokes_tensor(T0):
    """
    Check if all pixels satisfy the Stokes constraints.
    Shows a warning at the end with the percentage of invalid pixels (if any).
    """
    total_pixels = T0.shape[0] * T0.shape[1]
    invalid_count = 0

    for i in range(T0.shape[0]):
        for j in range(T0.shape[1]):
            stokes_vec = T0[i, j, :]
            if check_stokes_constraints(stokes_vec) == 0:
                invalid_count += 1

    if invalid_count > 0:
        percentage = (invalid_count / total_pixels) * 100
        warnings.warn(
            f"{percentage:.2f}% of pixels do not satisfy the Stokes constraints.",
            UserWarning,
        )
    else:
        print("All pixels satisfy the Stokes constraints.")


def check_stokes_constraints(S):
    """
    Check if a 4-D tensor satisfies the Stokes constraints.
    """
    return (
        0
        if (
            (round(S[0] ** 2, 5) < round(S[1] ** 2 + S[2] ** 2 + S[3] ** 2, 5))
            or (S[0] < 0)
        )
        else 1
    )


def stokes2coh(S):
    "construct Coherence matrix from Stokes parameters"
    S0 = S[0]
    S1 = S[1]
    S2 = S[2]
    S3 = S[3]
    coh = 0.5 * np.array([[S0 + S1, S2 + 1j * S3], [S2 - 1j * S3, S0 - S1]])
    return coh


def coh2stokes(coh):
    "Returns Stokes parameters from polarization (coherency) matrix"
    S0 = np.real(coh[0, 0] + coh[1, 1])
    S1 = np.real(coh[0, 0] - coh[1, 1])
    S2 = 2 * coh[0, 1].real
    S3 = 2 * coh[0, 1].imag

    S = np.array([S0, S1, S2, S3])
    return S


def projPSD(M):
    "projection of matrix M onto the set of PSD hermitian matrices"

    symM = 0.5 * (M + M.conj().T)
    w, v = np.linalg.eig(symM)

    proj = np.zeros_like(symM)
    for i in range(len(w)):
        proj += max(0, w[i].real) * np.outer(v[:, i], v[:, i].conj())

    return proj


def stokesProjection(S):
    """Projection of Stokes parameters onto the set of valid Stokes vectors."""
    coh = stokes2coh(S)
    proj = projPSD(coh)
    newS = coh2stokes(proj)
    return newS
