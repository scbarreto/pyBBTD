"""Small linear-algebra helpers shared by the models and solvers."""

import numpy as np


def project_psd(M):
    """Project a square matrix onto the positive semidefinite Hermitian matrices.

    Takes the Hermitian part of ``M`` and sets its negative eigenvalues to zero.
    """
    M_herm = 0.5 * (M + M.conj().T)
    eigenvalues, eigenvectors = np.linalg.eigh(M_herm)
    eigenvalues[eigenvalues < 0] = 0.0
    return eigenvectors @ np.diag(eigenvalues) @ eigenvectors.conj().T


def hermitian_to_real(X, dtype="float64"):
    """Pack Hermitian matrices into real vectors of the same size.

    Each ``K x K`` matrix becomes a vector of length ``K**2`` (row-major)
    holding the real part on and above the diagonal, and the imaginary part
    of the upper triangle in the positions below the diagonal.

    Parameters
    ----------
    X : np.ndarray
        Array of shape ``(..., K, K)``.
    dtype : str or np.dtype, default "float64"
        Output dtype.

    Returns
    -------
    np.ndarray
        Array of shape ``(..., K**2)``.
    """
    K = X.shape[-1]
    below_diagonal = np.tril(np.ones((K, K), dtype=bool), k=-1)
    packed = np.where(below_diagonal, np.swapaxes(X, -1, -2).imag, X.real)
    return packed.reshape(*X.shape[:-2], K * K).astype(dtype)
