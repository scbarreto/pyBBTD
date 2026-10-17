"""Cov-LL1 model: a BTD-LL1 for covariance imaging.

In covariance imaging, every pixel holds a ``K x K`` covariance matrix. The
matrices are vectorized so the data form a tensor of shape ``(I, J, K**2)``,
decomposed into R terms, each a non-negative spatial map times one
vectorized covariance matrix.
"""

import numpy as np

from pybbtd._utils import unknown_option_error, warn_user
from pybbtd.btd import BTD, factors_to_tensor


class CovLL1(BTD):
    """BTD-LL1 model whose vector factor holds vectorized covariance matrices.

    Extends :class:`~pybbtd.btd.BTD`. Each column of the factor C, of length
    ``K**2``, is a vectorized ``K x K`` covariance matrix.

    Parameters
    ----------
    dims : tuple of int
        Shape ``(I, J, K**2)`` of the tensor. The last dimension must be a
        perfect square.
    R : int
        Number of terms.
    L1 : int
        Rank of the spatial maps.
    L2 : int
        Rank of the covariance matrices. Only used to generate synthetic
        data with :meth:`generate_covll1_tensor`; the fit does not constrain it.
    block_mode : {"LL1", "L1L", "1LL"}, default "LL1"
        Block mode of the decomposition.

    Attributes
    ----------
    L1, L2 : int
        Ranks of the spatial maps and of the covariance matrices.
    factors : list of np.ndarray or None
        Factor matrices ``[A, B, C]`` estimated by :meth:`fit`; ``A`` has
        shape ``(I, L1 * R)``, ``B`` has shape ``(J, L1 * R)`` and ``C`` has
        shape ``(K**2, R)``.
    tensor : np.ndarray or None
        Tensor rebuilt from ``factors`` by :meth:`fit`.
    fit_error : np.ndarray or None
        Fit error at each iteration of the last :meth:`fit`.
    """

    def __init__(self, dims, R, L1, L2, block_mode="LL1"):
        super().__init__(dims, R=R, L=L1, block_mode=block_mode)
        self.rank, self.L1, self.L2 = _validate_covll1_params(self.dims, R, L1, L2)

    def __repr__(self):
        return (
            f"CovLL1(dims={self.dims}, R={self.rank}, L1={self.L1}, L2={self.L2}, "
            f"block_mode='{self.block_mode}')"
        )

    def generate_covll1_tensor(self):
        """Draw random Cov-LL1 factors and store them with their tensor.

        Updates :attr:`factors` and :attr:`tensor`.

        Returns
        -------
        factors : list of np.ndarray
            ``[A, B, C]`` from :func:`generate_covll1_factors`.
        tensor : np.ndarray
            Tensor of shape :attr:`dims`.
        """
        A, B, C = generate_covll1_factors(self.dims, self.rank, self.L1, self.L2)
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
            Tensor of shape ``(I, J, K**2)`` to decompose.
        algorithm : {"ADMM"}, default "ADMM"
            Fitting algorithm (see
            :func:`~pybbtd.solvers.covll1_admm.fit_covll1_admm`).
        **kwargs
            Passed to the solver, e.g. ``init``, ``max_iter``, ``rho``,
            ``max_admm``, ``rel_tol``, ``abs_tol``, ``admm_tol``.

        Returns
        -------
        CovLL1
            The fitted model.
        """
        if algorithm != "ADMM":
            raise unknown_option_error("algorithm", algorithm, ["ADMM"])

        from pybbtd.solvers.covll1_admm import fit_covll1_admm

        self.factors, self.fit_error = fit_covll1_admm(self, data, **kwargs)
        self.tensor = factors_to_tensor(
            *self.factors, self.get_constraint_matrix(), block_mode=self.block_mode
        )
        return self


def generate_covll1_factors(dims, R, L1, L2):
    """Draw random factors for a Cov-LL1 model.

    The spatial factors A, B are uniform on [0, 1). Each column of C is a
    vectorized ``K x K`` positive semidefinite Hermitian matrix of rank L2.

    Parameters
    ----------
    dims : tuple of int
        Shape ``(I, J, K**2)`` of the tensor.
    R : int
        Number of terms.
    L1 : int
        Rank of the spatial maps.
    L2 : int
        Rank of the covariance matrices.

    Returns
    -------
    A : np.ndarray
        Shape ``(I, L1 * R)``, non-negative.
    B : np.ndarray
        Shape ``(J, L1 * R)``, non-negative.
    C : np.ndarray
        Shape ``(K**2, R)``, one vectorized covariance matrix per column.
    """
    if len(dims) != 3:
        raise ValueError("Number of dimensions should be equal to 3.")

    A = np.random.rand(dims[0], L1 * R)
    B = np.random.rand(dims[1], L1 * R)

    K = int(np.sqrt(dims[2]))
    cov_factors = np.random.randn(R, K, L2) + 1j * np.random.randn(R, K, L2)
    sigmas = np.zeros((K, K, R), dtype=complex)
    for r in range(R):
        sigmas[:, :, r] = cov_factors[r] @ cov_factors[r].conj().T
    C = sigmas.reshape(K**2, R)

    return A, B, C


def _validate_covll1_params(dims, R, L1, L2):
    """Check that the Cov-LL1 parameters fit the tensor dimensions.

    The third dimension must be a perfect square ``K**2``, ``L1`` cannot
    exceed the image size and ``L2`` cannot exceed ``K``.
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
        raise ValueError("L1 cannot exceed the first and second dimensions (I and J).")
    if L2 > int(np.sqrt(dims[2])):
        raise ValueError("L2 cannot exceed the square root of the third dimension (K).")
    return R, L1, L2


def validate_covariance_tensor(T):
    """Check that every pixel of a tensor holds a valid covariance matrix.

    Each pixel ``T[i, j, :]`` is reshaped to ``K x K`` and tested with
    :func:`is_valid_covariance`. Warns with the percentage of invalid pixels,
    if any.

    Parameters
    ----------
    T : np.ndarray
        Tensor of shape ``(I, J, K**2)``.

    Returns
    -------
    bool
        ``True`` if all pixels are valid.
    """
    K = int(np.sqrt(T.shape[2]))
    matrices = T.reshape(-1, K, K)
    invalid_count = sum(not is_valid_covariance(Sigma) for Sigma in matrices)

    if invalid_count > 0:
        percentage = invalid_count / len(matrices) * 100
        warn_user(
            f"{percentage:.2f}% of pixels do not satisfy the covariance matrix "
            "constraints (must be square, Hermitian, and positive semidefinite)."
        )
    return invalid_count == 0


def is_valid_covariance(Sigma, tol=1e-10):
    """Check that a matrix is a valid covariance matrix.

    A valid covariance matrix is square, Hermitian and positive semidefinite.

    Parameters
    ----------
    Sigma : np.ndarray
        Matrix to test.
    tol : float, default 1e-10
        Tolerance on the Hermitian symmetry and on negative eigenvalues.

    Returns
    -------
    bool
    """
    Sigma = np.asarray(Sigma)

    if Sigma.ndim != 2 or Sigma.shape[0] != Sigma.shape[1]:
        return False
    if not np.allclose(Sigma, Sigma.conj().T, atol=tol):
        return False
    return not np.any(np.linalg.eigvalsh(Sigma) < -tol)
