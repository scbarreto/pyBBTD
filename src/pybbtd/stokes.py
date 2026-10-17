"""Stokes model: a BTD-LL1 for polarization images.

Each pixel of a polarization image is described by a Stokes vector
``[S0, S1, S2, S3]``: total intensity, and three numbers describing how the
light is polarized. The tensor of shape ``(I, J, 4)`` is decomposed into R
terms, each a non-negative spatial map times one Stokes vector.
"""

import numpy as np

from pybbtd._linalg import project_psd
from pybbtd._utils import check_dims, unknown_option_error, warn_user
from pybbtd.btd import BTD, factors_to_tensor


class Stokes(BTD):
    """BTD-LL1 model for polarization images, with valid Stokes vectors.

    Extends :class:`~pybbtd.btd.BTD` with the third mode fixed to size 4.
    Each column of the factor C is a physically valid Stokes vector
    (see :func:`is_valid_stokes`), and the spatial factors are non-negative.

    Parameters
    ----------
    spatial_dims : tuple of int
        Image size ``(I, J)``. The tensor has shape ``(I, J, 4)``.
    R : int
        Number of terms (polarization states).
    L : int or sequence of int
        Rank of the spatial map of each term.

    Attributes
    ----------
    factors : list of np.ndarray or None
        Factor matrices ``[A, B, C]`` estimated by :meth:`fit`; ``C`` has
        shape ``(4, R)``.
    tensor : np.ndarray or None
        Tensor rebuilt from ``factors`` by :meth:`fit`.
    fit_error : np.ndarray or None
        Fit error at each iteration of the last :meth:`fit`.
    """

    def __init__(self, spatial_dims, R, L):
        spatial_dims = check_dims(spatial_dims, 2)
        super().__init__(dims=(*spatial_dims, 4), R=R, L=L, block_mode="LL1")

    def generate_stokes_tensor(self):
        """Draw random Stokes factors and store them with their tensor.

        Updates :attr:`factors` and :attr:`tensor`.

        Returns
        -------
        factors : list of np.ndarray
            ``[A, B, C]`` from :func:`generate_stokes_factors`.
        tensor : np.ndarray
            Tensor of shape ``(I, J, 4)``.
        """
        A, B, C = generate_stokes_factors(self.dims, self.rank, self.L)
        self.factors = [A, B, C]
        self.tensor = factors_to_tensor(
            A, B, C, self.get_constraint_matrix(), block_mode=self.block_mode
        )
        return self.factors, self.tensor

    def fit(self, data, algorithm="ADMM", **kwargs):
        """Estimate the factors that best explain ``data``.

        Updates :attr:`factors`, :attr:`tensor` and :attr:`fit_error`.

        Parameters
        ----------
        data : np.ndarray
            Tensor of shape ``(I, J, 4)`` to decompose.
        algorithm : {"ADMM"}, default "ADMM"
            Fitting algorithm (see
            :func:`~pybbtd.solvers.stokes_admm.fit_stokes_admm`).
        **kwargs
            Passed to the solver, e.g. ``init``, ``max_iter``, ``rho``,
            ``max_admm``, ``rel_tol``, ``abs_tol``, ``admm_tol``.

        Returns
        -------
        Stokes
            The fitted model.
        """
        if algorithm != "ADMM":
            raise unknown_option_error("algorithm", algorithm, ["ADMM"])

        from pybbtd.solvers.stokes_admm import fit_stokes_admm

        self.factors, self.fit_error = fit_stokes_admm(self, data, **kwargs)
        self.tensor = factors_to_tensor(
            *self.factors, self.get_constraint_matrix(), block_mode=self.block_mode
        )
        return self


def generate_stokes_factors(dims, R, L):
    """Draw random factors for a Stokes model.

    The spatial factors A, B are uniform on [0, 1). Each column of C is the
    Stokes vector of a random fully polarized state.

    Parameters
    ----------
    dims : tuple of int
        Shape ``(I, J, 4)`` of the tensor.
    R : int
        Number of terms.
    L : sequence of int
        Rank of the spatial maps; ``L[0]`` is used for every term.

    Returns
    -------
    A : np.ndarray
        Shape ``(I, L[0] * R)``, non-negative.
    B : np.ndarray
        Shape ``(J, L[0] * R)``, non-negative.
    C : np.ndarray
        Shape ``(4, R)``, one valid Stokes vector per column.
    """
    A = np.random.rand(dims[0], L[0] * R)
    B = np.random.rand(dims[1], L[0] * R)
    C = np.zeros((dims[2], R))

    for r in range(R):
        cr = 1.0 / np.sqrt(2) * (np.random.randn(2) + 1j * np.random.randn(2))
        cr = cr / np.linalg.norm(cr)
        C[:, r] = coherency_to_stokes(np.outer(cr, cr.conj()))

    return A, B, C


def validate_stokes_tensor(T):
    """Check that every pixel of a tensor holds a valid Stokes vector.

    Warns with the percentage of invalid pixels, if any.

    Parameters
    ----------
    T : np.ndarray
        Tensor of shape ``(I, J, 4)``.

    Returns
    -------
    bool
        ``True`` if all pixels are valid.
    """
    pixels = T.reshape(-1, T.shape[2])
    invalid_count = sum(not is_valid_stokes(S) for S in pixels)

    if invalid_count > 0:
        percentage = invalid_count / len(pixels) * 100
        warn_user(f"{percentage:.2f}% of pixels do not satisfy the Stokes constraints.")
    return invalid_count == 0


def is_valid_stokes(S):
    """Check that a Stokes vector is physically valid.

    A valid Stokes vector has :math:`S_0 \\geq 0` and
    :math:`S_0^2 \\geq S_1^2 + S_2^2 + S_3^2` (the polarized part cannot carry
    more power than the total intensity). The comparison is rounded to 5
    decimals.

    Parameters
    ----------
    S : np.ndarray
        Stokes vector ``[S0, S1, S2, S3]``.

    Returns
    -------
    bool
    """
    return not (
        round(S[0] ** 2, 5) < round(S[1] ** 2 + S[2] ** 2 + S[3] ** 2, 5) or S[0] < 0
    )


def stokes_to_coherency(S):
    """Convert a Stokes vector to its 2 x 2 coherency matrix.

    Parameters
    ----------
    S : np.ndarray
        Stokes vector ``[S0, S1, S2, S3]``.

    Returns
    -------
    np.ndarray
        Complex Hermitian matrix of shape ``(2, 2)``.
    """
    S0, S1, S2, S3 = S
    return 0.5 * np.array([[S0 + S1, S2 + 1j * S3], [S2 - 1j * S3, S0 - S1]])


def coherency_to_stokes(coh):
    """Convert a 2 x 2 coherency matrix to its Stokes vector.

    Parameters
    ----------
    coh : np.ndarray
        Complex Hermitian matrix of shape ``(2, 2)``.

    Returns
    -------
    np.ndarray
        Real Stokes vector ``[S0, S1, S2, S3]``.
    """
    S0 = np.real(coh[0, 0] + coh[1, 1])
    S1 = np.real(coh[0, 0] - coh[1, 1])
    S2 = 2 * coh[0, 1].real
    S3 = 2 * coh[0, 1].imag
    return np.array([S0, S1, S2, S3])


def project_stokes(S):
    """Project a vector onto the set of valid Stokes vectors.

    The vector is converted to a coherency matrix, projected onto the
    positive semidefinite matrices, and converted back.

    Parameters
    ----------
    S : np.ndarray
        Vector ``[S0, S1, S2, S3]``.

    Returns
    -------
    np.ndarray
        The closest valid Stokes vector.
    """
    return coherency_to_stokes(project_psd(stokes_to_coherency(S)))
