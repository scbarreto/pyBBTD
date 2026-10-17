"""Alternating least squares (ALS) solver for the unconstrained BBTD model."""

import logging

import numpy as np
from scipy.linalg import solve
from tensorly import unfold
from tensorly.tenalg import khatri_rao

from pybbtd._utils import check_solver_inputs, unknown_option_error, warn_user
from pybbtd.bbtd import BBTD, factors_to_tensor

logger = logging.getLogger(__name__)

INIT_STRATEGIES = ("random", "svd")


def fit_bbtd_als(model, T, init="random", max_iter=1000, abs_tol=1e-8, rel_tol=1e-3):
    """Fit an unconstrained BBTD with alternating least squares (ALS).

    Each iteration solves a least-squares problem for A, B, C and D in turn,
    keeping the other factors fixed.

    Parameters
    ----------
    model : BBTD
        Model defining the dimensions and ranks.
    T : np.ndarray
        Fourth-order tensor of shape ``model.dims`` to decompose.
    init : {"random", "svd"}, default "random"
        Initialization strategy (see :func:`init_bbtd_factors`).
    max_iter : int, default 1000
        Maximum number of iterations.
    abs_tol : float, default 1e-8
        Stop when the fit error falls below this value.
    rel_tol : float, default 1e-3
        Stop when the relative decrease of the fit error falls below this
        value.

    Returns
    -------
    factors : list of np.ndarray
        Estimated ``[A, B, C, D]``.
    fit_error : np.ndarray
        Relative reconstruction error
        :math:`\\|\\mathcal{T} - \\hat{\\mathcal{T}}\\| / \\|\\mathcal{T}\\|`
        at initialization and after each iteration.
    """
    check_solver_inputs(model, BBTD, T)

    # precompute unfoldings
    T0 = unfold(T, 0)
    T1 = unfold(T, 1)
    T2 = unfold(T, 2)
    T3 = unfold(T, 3)

    # init factors
    Ak, Bk, Ck, Dk = init_bbtd_factors(model, strategy=init, T=T)
    phi, psi = model.get_constraint_matrices()

    k = 0
    exit_criterion = False
    eps_zero = 1e-13
    Tfit_0 = factors_to_tensor(Ak, Bk, Ck, Dk, phi, psi)
    fit_error = [np.linalg.norm(Tfit_0 - T) / np.linalg.norm(T)]

    while not exit_criterion:
        # update A
        M0 = phi @ khatri_rao((Bk @ phi, Ck @ psi, Dk @ psi)).T
        Ak1 = _solve_ls(T0, M0)

        # update B
        M1 = phi @ khatri_rao((Ak1 @ phi, Ck @ psi, Dk @ psi)).T
        Bk1 = _solve_ls(T1, M1)

        # update C
        M2 = psi @ khatri_rao((Ak1 @ phi, Bk1 @ phi, Dk @ psi)).T
        Ck1 = _solve_ls(T2, M2)

        # update D
        M3 = psi @ khatri_rao((Ak1 @ phi, Bk1 @ phi, Ck1 @ psi)).T
        Dk1 = _solve_ls(T3, M3)

        # compute reconstruction error
        Tfit_k = factors_to_tensor(Ak1, Bk1, Ck1, Dk1, phi, psi)
        fit_error.append(np.linalg.norm(Tfit_k - T) / np.linalg.norm(T))

        # exit criterion
        k += 1
        if np.abs(fit_error[-1] - fit_error[-2]) / (fit_error[-2] + eps_zero) < rel_tol:
            logger.info("Iteration %d: fit error stopped decreasing (rel_tol).", k)
            exit_criterion = True
        if fit_error[-1] < abs_tol:
            logger.info("Iteration %d: fit error below abs_tol.", k)
            exit_criterion = True
        if k >= max_iter and not exit_criterion:
            warn_user(
                f"Reached max_iter={max_iter} before converging. Check the fit error.",
                RuntimeWarning,
            )
            exit_criterion = True

        # pass variables
        Ak, Bk, Ck, Dk = Ak1, Bk1, Ck1, Dk1

    return [Ak, Bk, Ck, Dk], np.array(fit_error)


def _solve_ls(Tn, M):
    """Least-squares update ``argmin_X ||Tn - X @ M||`` via the normal equations."""
    MMH = np.nan_to_num(M @ M.T.conj(), nan=0, posinf=0, neginf=0)
    RHS = np.nan_to_num(Tn @ M.T.conj(), nan=0, posinf=0, neginf=0)
    return solve(MMH.T, RHS.T, assume_a="pos").T


def init_bbtd_factors(model, strategy="random", T=None):
    """Initialize the BBTD factors.

    Parameters
    ----------
    model : BBTD
        Model defining the dimensions and ranks.
    strategy : {"random", "svd"}, default "random"
        ``"random"`` draws standard Gaussian entries. ``"svd"`` uses the
        leading singular vectors of each unfolding of ``T``.
    T : np.ndarray, optional
        Data tensor, required for ``"svd"``.

    Returns
    -------
    A, B, C, D : np.ndarray
        Initial factor matrices.
    """
    dims = model.dims
    R = model.rank
    L1 = model.L1
    L2 = model.L2

    if strategy == "random":
        A = np.random.randn(dims[0], L1 * R)
        B = np.random.randn(dims[1], L1 * R)
        C = np.random.randn(dims[2], L2 * R)
        D = np.random.randn(dims[3], L2 * R)
    elif strategy == "svd":
        if T is None:
            raise ValueError("SVD init requires input data T.")
        u0, s0, _ = np.linalg.svd(unfold(T, 0), full_matrices=False)
        u1, s1, _ = np.linalg.svd(unfold(T, 1), full_matrices=False)
        u2, s2, _ = np.linalg.svd(unfold(T, 2), full_matrices=False)
        u3, s3, _ = np.linalg.svd(unfold(T, 3), full_matrices=False)

        A = u0[:, : L1 * R] @ np.diag(s0[: L1 * R] ** 0.5)
        B = u1[:, : L1 * R] @ np.diag(s1[: L1 * R] ** 0.5)
        C = u2[:, : L2 * R] @ np.diag(s2[: L2 * R] ** 0.5)
        D = u3[:, : L2 * R] @ np.diag(s3[: L2 * R] ** 0.5)
    else:
        raise unknown_option_error("initialization", strategy, INIT_STRATEGIES)

    return A, B, C, D
