"""Inner ADMM loop shared by the AO-ADMM solvers."""

import numpy as np
from scipy.linalg import solve


def admm_update(
    Y, M, X_init, Xt_init, rho, project, max_iter=100, tol=1e-14, real=False
):
    """Constrained least-squares update of one factor with ADMM.

    Approximately solves

    .. math::

        \\min_{\\mathbf{X}} \\|\\mathbf{Y} - \\mathbf{X}\\mathbf{M}\\|_F^2
        \\quad \\text{s.t.} \\quad \\mathbf{X} \\in \\mathcal{S},

    by splitting the factor into an unconstrained variable X and a
    constrained copy :math:`\\tilde{\\mathbf{X}}`, linked by a scaled dual
    variable U. The penalty ``rho`` is adapted to balance the primal and
    dual residuals.

    Parameters
    ----------
    Y : np.ndarray
        Unfolding of the data tensor along the mode of the factor.
    M : np.ndarray
        Matrix such that ``Y ≈ X @ M`` (built from the other factors).
    X_init : np.ndarray
        Initial unconstrained variable X.
    Xt_init : np.ndarray
        Initial constrained variable :math:`\\tilde{\\mathbf{X}}`.
    rho : float
        Initial ADMM penalty parameter.
    project : callable
        Euclidean projection onto the constraint set :math:`\\mathcal{S}`.
    max_iter : int, default 100
        Maximum number of ADMM iterations.
    tol : float, default 1e-14
        Convergence tolerance on the primal and dual residuals.
    real : bool, default False
        Restrict the factor to real values, even when ``Y`` and ``M`` are
        complex (e.g. non-negative spatial factors of complex data).

    Returns
    -------
    Xt : np.ndarray
        Constrained variable (the factor estimate).
    X : np.ndarray
        Unconstrained variable.
    """
    # init variables
    Xl = X_init.copy()
    Xtl = Xt_init.copy()
    Ul = np.zeros_like(X_init)

    # normal equations of the least-squares term
    MMH = M @ M.conj().T
    RHS = Y @ M.conj().T
    if real:
        MMH = MMH.real
        RHS = RHS.real

    eps_pri = np.size(Xl) ** (1 / 2) * tol
    eps_dual = np.size(Xtl) ** (1 / 2) * tol

    n = 0
    exit_criterion = False
    while n < max_iter and not exit_criterion:
        # update X_{l+1}
        X_T = solve((MMH + rho * np.eye(X_init.shape[1])).T, (RHS + rho * (Xtl - Ul)).T)
        Xl1 = X_T.T

        # update \tilde{X}_{l+1}
        Xtl1 = project(Xl1 + Ul)

        # update dual
        Ul1 = Ul + Xl1 - Xtl1

        # compute convergence metrics residuals
        primal_residue = np.linalg.norm(Xl1 - Xtl1)
        dual_residue = rho * np.linalg.norm(Xtl1 - Xtl)

        if (primal_residue < eps_pri) and (dual_residue < eps_dual):
            exit_criterion = True
        if primal_residue > 10 * dual_residue:
            rho *= 2
            Ul1 /= 2
        elif dual_residue > 10 * primal_residue:
            rho /= 2
            Ul1 *= 2

        # update all variables
        Xl = Xl1
        Xtl = Xtl1
        Ul = Ul1

        n += 1

    return Xtl, Xl


def project_nonnegative(X):
    """Projection onto the non-negative matrices."""
    return np.maximum(0, X)
