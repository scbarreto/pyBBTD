"""Alternating least squares (ALS) solver for the BTD model."""

import logging

import numpy as np
from scipy.linalg import solve
from tensorly import transpose, unfold
from tensorly.tenalg import khatri_rao

from pybbtd._utils import check_solver_inputs, unknown_option_error, warn_user
from pybbtd.btd import BTD, _constraint_matrix, factors_to_tensor

logger = logging.getLogger(__name__)

INIT_STRATEGIES = ("random", "svd")

# The ALS updates are written for the LL1 mode. The other modes are solved
# as the LL1 problem of the tensor with its axes permuted by these orders.
_LL1_AXES = {"LL1": (0, 1, 2), "L1L": (0, 2, 1), "1LL": (1, 2, 0)}


def fit_btd_als(model, T, init="random", max_iter=1000, abs_tol=1e-8, rel_tol=1e-3):
    """Fit a BTD with alternating least squares (ALS).

    Each iteration solves a least-squares problem for A, then B, then C,
    keeping the other factors fixed. All three block modes are supported.

    Parameters
    ----------
    model : BTD
        Model defining the dimensions, ranks and block mode.
    T : np.ndarray
        Tensor of shape ``model.dims`` to decompose.
    init : {"random", "svd"}, default "random"
        Initialization strategy (see :func:`init_btd_factors`).
    max_iter : int, default 1000
        Maximum number of iterations.
    abs_tol : float, default 1e-8
        Stop when ``fit_error / ||T||`` falls below this value.
    rel_tol : float, default 1e-3
        Stop when the relative decrease of the fit error falls below this
        value.

    Returns
    -------
    factors : list of np.ndarray
        Estimated ``[A, B, C]``.
    fit_error : np.ndarray
        Squared reconstruction error :math:`\\|\\mathcal{T} - \\hat{\\mathcal{T}}\\|^2`
        at initialization and after each iteration.
    """
    check_solver_inputs(model, BTD, T)

    # algo is for LL1; permute to the equivalent LL1 problem if needed
    axes = _LL1_AXES[model.block_mode]
    T_ll1 = transpose(T, axes)
    dims_ll1 = tuple(model.dims[axis] for axis in axes)

    # precompute unfoldings
    T0 = unfold(T_ll1, 0)
    T1 = unfold(T_ll1, 1)
    T2 = unfold(T_ll1, 2)

    # init factors
    Ak, Bk, Ck = _init_factors(dims_ll1, model.rank, model.L, init, T_ll1)
    theta = _constraint_matrix(model.rank, model.L)

    k = 0
    exit_criterion = False
    Tfit_0 = factors_to_tensor(Ak, Bk, Ck, theta)
    fit_error = [np.linalg.norm(Tfit_0 - T_ll1) ** 2]

    while not exit_criterion:
        # update A
        M0 = khatri_rao([Bk, Ck @ theta])
        LHS = M0.T @ M0
        RHS = M0.T @ T0.T
        Ak1 = solve(LHS, RHS).T

        # update B
        M1 = khatri_rao([Ak1, Ck @ theta])
        LHS = M1.T @ M1
        RHS = M1.T @ T1.T
        Bk1 = solve(LHS, RHS).T

        # update C
        M2 = khatri_rao([Ak1, Bk1]) @ theta.T
        LHS = M2.T @ M2
        RHS = M2.T @ T2.T
        Ck1 = solve(LHS, RHS).T

        # compute reconstruction error
        Tfit_k = factors_to_tensor(Ak1, Bk1, Ck1, theta)
        fit_error.append(np.linalg.norm(Tfit_k - T_ll1) ** 2)

        # exit criterion
        k += 1
        if np.abs(fit_error[-1] - fit_error[-2]) / fit_error[-1] < rel_tol:
            logger.info("Iteration %d: fit error stopped decreasing (rel_tol).", k)
            exit_criterion = True
        if fit_error[-1] / np.linalg.norm(T_ll1) < abs_tol:
            logger.info("Iteration %d: fit error below abs_tol.", k)
            exit_criterion = True
        if k >= max_iter and not exit_criterion:
            warn_user(
                f"Reached max_iter={max_iter} before converging. Check the fit error.",
                RuntimeWarning,
            )
            exit_criterion = True

        # pass variables
        Ak, Bk, Ck = Ak1, Bk1, Ck1

    # put the factors back in the order of the model's modes
    factors = [None, None, None]
    for factor, axis in zip([Ak, Bk, Ck], axes, strict=True):
        factors[axis] = factor

    return factors, np.array(fit_error)


def init_btd_factors(model, strategy="random", T=None):
    """Initialize the BTD factors.

    The factors are returned in the LL1 layout: ``A`` and ``B`` have
    ``sum(L)`` columns and ``C`` has ``R`` columns.

    Parameters
    ----------
    model : BTD
        Model defining the dimensions and ranks.
    strategy : {"random", "svd"}, default "random"
        ``"random"`` draws standard Gaussian entries. ``"svd"`` uses the
        leading singular vectors of each unfolding of ``T``.
    T : np.ndarray, optional
        Data tensor, required for ``"svd"``.

    Returns
    -------
    A, B, C : np.ndarray
        Initial factor matrices.
    """
    return _init_factors(model.dims, model.rank, model.L, strategy, T)


def _init_factors(dims, R, L, strategy, T):
    """Initialize LL1 factors for a tensor of shape ``dims``."""
    Lsum = np.array(L).sum()

    if strategy == "random":
        A = np.random.randn(dims[0], Lsum)
        B = np.random.randn(dims[1], Lsum)
        C = np.random.randn(dims[2], R)
    elif strategy == "svd":
        if T is None:
            raise ValueError("SVD init requires input data T")
        u0, s0, _ = np.linalg.svd(unfold(T, 0))
        u1, s1, _ = np.linalg.svd(unfold(T, 1))
        u2, s2, _ = np.linalg.svd(unfold(T, 2))

        A = u0[:, :Lsum] @ np.diag(s0[:Lsum] ** 0.5)
        B = u1[:, :Lsum] @ np.diag(s1[:Lsum] ** 0.5)
        C = u2[:, :R] @ np.diag(s2[:R] ** 0.5)
    else:
        raise unknown_option_error("initialization", strategy, INIT_STRATEGIES)

    return A, B, C
