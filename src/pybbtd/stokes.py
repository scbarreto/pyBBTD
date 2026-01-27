from pybbtd.btd import BTD
import numpy as np
import pybbtd.btd as btd

import warnings


class Stokes(BTD):
    """
    Class for tensors admitting a Stokes-constrained BTD-LL1 decomposition.

    Extends :class:`~pybbtd.btd.BTD` for polarimetric imaging: the third mode
    is fixed to size 4 (Stokes parameters) and each column of the spectral
    factor C must be a physically valid Stokes vector.

    :param spatial_dims: Spatial dimensions ``(I, J)`` of the tensor.
        The full tensor has shape ``(I, J, 4)``.
    :type spatial_dims: list[int] or Tuple[int, int]
    :param R: Number of block terms (rank of the decomposition).
    :type R: int
    :param L: Rank of the spatial maps in each term. Can be a single integer
        or a list of R integers.
    :type L: int or list[int]

    :ivar factors: List of factor matrices ``[A, B, C]`` after fitting,
        or ``None`` before fitting.
    :vartype factors: list[np.ndarray] or None
    :ivar tensor: Reconstructed tensor after fitting, or ``None`` before fitting.
    :vartype tensor: np.ndarray or None
    :ivar fit_error: Array of fitting errors at each iteration, or ``None``
        before fitting.
    :vartype fit_error: np.ndarray or None
    """

    def __init__(self, spatial_dims, R, L, **kwargs):
        """
        Initialize a Stokes-BTD model.

        The block mode is automatically set to ``"LL1"`` and the third
        dimension is fixed to 4.

        Parameters:
            spatial_dims: list[int] or Tuple[int, int]
                Spatial dimensions ``(I, J)`` of the tensor.
            R: int
                Number of block terms (rank of the decomposition).
            L: int or list[int]
                Rank of the spatial maps in each term.
        """

        dims = (spatial_dims[0], spatial_dims[1], 4)
        kwargs["block_mode"] = "LL1"
        super().__init__(dims=dims, R=R, L=L, **kwargs)
        print(
            f"Stokes tensor initialized with dimensions {self.dims} on {self.block_mode} mode."
        )

    def generate_stokes_tensor(self):
        """
        Generate a random Stokes tensor and store the factors.

        Non-negative spatial factors A, B are drawn uniformly, and each
        column of C is a valid Stokes vector built from a random coherency
        matrix. Updates ``self.factors`` and ``self.tensor`` in place.

        Returns:
            Tuple[list, np.ndarray]:
                A tuple ``(factors, tensor)`` where ``factors = [A, B, C]``
                and ``tensor`` has shape ``(I, J, 4)``.
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
        Fit a Stokes-BTD model to the given data using the specified algorithm.

        After fitting, ``self.factors``, ``self.tensor``, and ``self.fit_error``
        are updated in place. The ADMM solver enforces Stokes constraints on C.

        Parameters:
            data: np.ndarray
                Input tensor of shape ``(I, J, 4)`` to be decomposed.
            algorithm: str
                Algorithm to use for fitting (default: ``"ADMM"``).
            **kwargs:
                Additional keyword arguments passed to the solver
                (e.g. ``init``, ``max_iter``, ``rho``, ``max_admm``,
                ``rel_tol``, ``abs_tol``, ``admm_tol``).
        """
        from pybbtd.solvers.stokes_admm import STOKES_ADMM

        if algorithm == "ADMM":
            self.factors, self.fit_error = STOKES_ADMM(self, data, **kwargs)
            self.tensor = btd.factors_to_tensor(
                *self.factors, self.get_constraint_matrix(), block_mode=self.block_mode
            )
        else:
            raise UserWarning("Algorithm not implemented yet")

    def _get_constraint_matrix(self):
        """
        Return the constraint matrix for the CP-equivalent BTD-LL1 model.

        .. note::
            This is a private alias. Use the inherited
            :meth:`~pybbtd.btd.BTD.get_constraint_matrix` instead.
        """
        return btd._constraint_matrix(self.rank, self.L)


def generate_stokes_factors(dims, R, L):
    """
    Generate random factors that follow Stokes-BTD decomposition.

    Non-negative spatial factors A, B are drawn from a uniform distribution.
    Each column of C is a valid Stokes vector built from a random 2x2
    coherency matrix.

    Parameters:
        dims (Tuple[int, int, int]):
            Dimensions ``(I, J, 4)`` of the tensor.
        R (int):
            Number of block terms (rank of the decomposition).
        L (list[int] or np.ndarray):
            Rank of the spatial maps. Uses ``L[0]`` for all terms.

    Returns:
        Tuple[np.ndarray, np.ndarray, np.ndarray]:
            A tuple ``(A, B, C)`` where:

            - A: Shape ``(I, L[0] * R)``, non-negative.
            - B: Shape ``(J, L[0] * R)``, non-negative.
            - C: Shape ``(4, R)``, each column is a valid Stokes vector.
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
    Check if all pixels in a tensor satisfy the Stokes constraints.

    Each spatial pixel ``T0[i, j, :]`` is tested with
    :func:`check_stokes_constraints`. Emits a warning with the percentage
    of invalid pixels if any are found.

    Parameters:
        T0: np.ndarray
            Tensor of shape ``(I, J, 4)`` whose third mode contains
            Stokes vectors.
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
    Check if a single Stokes vector satisfies the physical constraints.

    A valid Stokes vector must have :math:`S_0 \\geq 0` and
    :math:`S_0^2 \\geq S_1^2 + S_2^2 + S_3^2`.

    Parameters:
        S: np.ndarray
            A 4-element Stokes vector ``[S0, S1, S2, S3]``.

    Returns:
        int:
            1 if valid, 0 if invalid.
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
    """
    Construct the 2x2 coherency matrix from a Stokes vector.

    Parameters:
        S: np.ndarray
            A 4-element Stokes vector ``[S0, S1, S2, S3]``.

    Returns:
        np.ndarray:
            Complex Hermitian coherency matrix of shape ``(2, 2)``.
    """
    S0 = S[0]
    S1 = S[1]
    S2 = S[2]
    S3 = S[3]
    coh = 0.5 * np.array([[S0 + S1, S2 + 1j * S3], [S2 - 1j * S3, S0 - S1]])
    return coh


def coh2stokes(coh):
    """
    Compute the Stokes vector from a 2x2 coherency matrix.

    Parameters:
        coh: np.ndarray
            Complex Hermitian coherency matrix of shape ``(2, 2)``.

    Returns:
        np.ndarray:
            A 4-element real Stokes vector ``[S0, S1, S2, S3]``.
    """
    S0 = np.real(coh[0, 0] + coh[1, 1])
    S1 = np.real(coh[0, 0] - coh[1, 1])
    S2 = 2 * coh[0, 1].real
    S3 = 2 * coh[0, 1].imag

    S = np.array([S0, S1, S2, S3])
    return S


def proj_psd(M):
    """
    Project a matrix onto the set of positive semidefinite Hermitian matrices.

    Symmetrizes M, computes its eigendecomposition, and zeroes out any
    negative eigenvalues.

    Parameters:
        M: np.ndarray
            Square matrix to project.

    Returns:
        np.ndarray:
            Nearest PSD Hermitian matrix (same shape as M).
    """

    symM = 0.5 * (M + M.conj().T)
    w, v = np.linalg.eig(symM)

    proj = np.zeros_like(symM)
    for i in range(len(w)):
        proj += max(0, w[i].real) * np.outer(v[:, i], v[:, i].conj())

    return proj


def stokes_projection(S):
    """
    Project a Stokes vector onto the set of physically valid Stokes vectors.

    Converts to a coherency matrix, projects onto the PSD cone via
    :func:`proj_psd`, and converts back to Stokes parameters.

    Parameters:
        S: np.ndarray
            A 4-element Stokes vector ``[S0, S1, S2, S3]``.

    Returns:
        np.ndarray:
            Projected 4-element Stokes vector satisfying the physical constraints.
    """
    coh = stokes2coh(S)
    proj = proj_psd(coh)
    newS = coh2stokes(proj)
    return newS
