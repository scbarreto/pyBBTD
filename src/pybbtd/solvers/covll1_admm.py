"""AO-ADMM solver for the Cov-LL1 model."""

import logging

import numpy as np
from scipy import linalg
from scipy.linalg import solve
from sklearn.decomposition import NMF
from tensorly import unfold
from tensorly.tenalg import khatri_rao

from pybbtd._linalg import hermitian_to_real
from pybbtd._utils import check_solver_inputs, unknown_option_error, warn_user
from pybbtd.btd import factors_to_tensor
from pybbtd.covll1 import CovLL1, generate_covll1_factors
from pybbtd.solvers._admm import admm_update, project_nonnegative
from pybbtd.solvers._kmeans import cluster_maps, cluster_pixels, match_norm, nmf_factors

logger = logging.getLogger(__name__)

INIT_STRATEGIES = ("random", "kmeans")


def fit_covll1_admm(
    model,
    T,
    init="random",
    max_iter=1000,
    rho=1,
    max_admm=1,
    rel_tol=1e-10,
    abs_tol=1e-10,
    admm_tol=1e-8,
):
    """Fit a Cov-LL1 model with AO-ADMM.

    The factors are updated in turn (A, then B, then C), each with a few
    ADMM iterations. A and B are kept non-negative; each column of C is
    passed through :func:`_project_psd_svd`.

    Parameters
    ----------
    model : CovLL1
        Model defining the dimensions and ranks.
    T : np.ndarray
        Tensor of shape ``(I, J, K**2)`` to decompose.
    init : {"random", "kmeans"}, default "random"
        Initialization strategy (see :func:`init_covll1_factors`).
    max_iter : int, default 1000
        Maximum number of outer iterations.
    rho : float, default 1
        Initial ADMM penalty parameter.
    max_admm : int, default 1
        Maximum number of inner ADMM iterations per factor update.
    rel_tol : float, default 1e-10
        Stop when the relative decrease of the fit error falls below this
        value.
    abs_tol : float, default 1e-10
        Stop when ``fit_error / ||T||`` falls below this value.
    admm_tol : float, default 1e-8
        Tolerance of the inner ADMM loops.

    Returns
    -------
    factors : list of np.ndarray
        Estimated ``[A, B, C]``.
    fit_error : np.ndarray
        Normalized squared reconstruction error
        :math:`\\|\\mathcal{T} - \\hat{\\mathcal{T}}\\|^2 / \\|\\mathcal{T}\\|^2`
        at initialization and after each iteration.
    """
    check_solver_inputs(model, CovLL1, T)

    if init == "random":
        Ak, Bk, Ck = init_covll1_factors(model, strategy="random")
        Atk, Btk, Ctk = init_covll1_factors(model, strategy="random")
    elif init == "kmeans":
        Atk, Btk, Ctk = init_covll1_factors(model, strategy="kmeans", T=T)
        Ak, Bk, Ck = init_covll1_factors(model, strategy="random")
    else:
        raise unknown_option_error("initialization", init, INIT_STRATEGIES)

    T1 = unfold(T, 0)
    T2 = unfold(T, 1)
    T3 = unfold(T, 2)

    theta = model.get_constraint_matrix()

    Tfit_0 = factors_to_tensor(Atk, Btk, Ctk, theta)
    fit_error = [np.linalg.norm(Tfit_0 - T) ** 2 / np.linalg.norm(T) ** 2]

    progress_interval = max(max_iter // 5, 1)
    k = 0
    exit_criterion = False
    while not exit_criterion:
        if k != 0 and k % progress_interval == 0:
            logger.info("Progress: %.0f%%", k / max_iter * 100)

        # update A
        M1 = khatri_rao([Btk, Ctk @ theta]).T
        Atk1, Ak1 = admm_update(
            T1, M1, Ak, Atk, rho, project_nonnegative, max_iter=max_admm, tol=admm_tol
        )

        # update B
        M2 = khatri_rao([Atk1, Ctk @ theta]).T
        Btk1, Bk1 = admm_update(
            T2, M2, Bk, Btk, rho, project_nonnegative, max_iter=max_admm, tol=admm_tol
        )

        # update C
        Ctk1, Ck1 = _admm_update_c(
            T3, Atk1, Btk1, Ck, Ctk, theta, rho, max_iter=max_admm, tol=admm_tol
        )

        # update variables
        Ak, Bk, Ck = Ak1, Bk1, Ck1
        Atk, Btk, Ctk = Atk1, Btk1, Ctk1
        k += 1

        # compute reconstruction error
        Tfit_k = factors_to_tensor(Atk1, Btk1, Ctk1, theta)
        fit_error.append(np.linalg.norm(Tfit_k - T) ** 2 / np.linalg.norm(T) ** 2)

        # exit criterion
        if np.abs(fit_error[-1] - fit_error[-2]) / fit_error[-1] < rel_tol:
            logger.info("Iteration %d: fit error stopped decreasing (rel_tol).", k)
            exit_criterion = True
        if fit_error[-1] / np.linalg.norm(T) < abs_tol:
            logger.info("Iteration %d: fit error below abs_tol.", k)
            exit_criterion = True
        if k >= max_iter and not exit_criterion:
            warn_user(
                f"Reached max_iter={max_iter} before converging. Check the fit error.",
                RuntimeWarning,
            )
            exit_criterion = True

    return [Atk, Btk, Ctk], np.array(fit_error)


def _admm_update_c(Y3, Ak, Bk, Cinit, Ctinit, theta, rho, max_iter=100, tol=1e-14):
    """ADMM update of the covariance factor C.

    Same loop as :func:`~pybbtd.solvers._admm.admm_update`, except that the
    dual variable is added to the least-squares variable before the
    projection, and each column of C goes through :func:`_project_psd_svd`.

    Returns
    -------
    Ct : np.ndarray
        Projected variable (the factor estimate).
    C : np.ndarray
        Least-squares variable.
    """
    K = int(np.sqrt(Cinit.shape[0]))

    # init variables
    Cl = Cinit.copy()
    Ctl = Ctinit.copy()
    Ul = np.zeros_like(Cl)

    M3 = theta @ khatri_rao([Ak, Bk]).T
    M3M3T = M3 @ M3.T
    RHS = Y3 @ M3.T

    eps_pri = np.size(Cl) ** (1 / 2) * tol
    eps_dual = np.size(Ctl) ** (1 / 2) * tol

    n = 0
    exit_criterion = False
    while n < max_iter and not exit_criterion:
        # update C_(l+1)
        C_T = solve(
            (M3M3T + rho * np.eye(Cinit.shape[1])).T, (RHS + rho * (Ctl - Ul)).T
        )
        Cl1 = C_T.T

        # update \tilde{C}_{l+1}
        Ctl1 = np.zeros_like(Ctl)
        Cl1 += Ul
        for r in range(Cinit.shape[1]):
            Ctl1[:, r] = _project_psd_svd(Cl1[:, r].reshape(K, K)).reshape(-1)

        # update dual
        Ul1 = Ul + Cl1 - Ctl1

        # compute convergence metrics residuals
        primal_residue = np.linalg.norm(Cl1 - Ctl1)
        dual_residue = rho * np.linalg.norm(Ctl1 - Ctl)

        if (primal_residue < eps_pri) and (dual_residue < eps_dual):
            exit_criterion = True
        if primal_residue > 10 * dual_residue:
            rho *= 2
            Ul1 /= 2
        elif dual_residue > 10 * primal_residue:
            rho /= 2
            Ul1 *= 2

        # update all variables
        Cl = Cl1
        Ctl = Ctl1
        Ul = Ul1

        n += 1

    return Ctl, Cl


def _project_psd_svd(cov_matrix):
    """Rebuild a ``K x K`` matrix from its SVD with negative singular values set to zero.

    Parameters
    ----------
    cov_matrix : np.ndarray
        Matrix of shape ``(K, K)``.

    Returns
    -------
    np.ndarray
        Matrix of shape ``(K, K)``.
    """
    U, s, Vh = linalg.svd(cov_matrix)
    return (U * np.clip(s, a_min=0, a_max=None)) @ Vh


def init_covll1_factors(model, strategy="random", T=None):
    """Initialize the Cov-LL1 factors.

    Parameters
    ----------
    model : CovLL1
        Model defining the dimensions and ranks.
    strategy : {"random", "kmeans"}, default "random"
        ``"random"`` draws factors with :func:`~pybbtd.covll1.generate_covll1_factors`.
        ``"kmeans"`` clusters the pixels by their covariance matrices
        (see :func:`_kmeans_init`).
    T : np.ndarray, optional
        Data tensor, required for ``"kmeans"``.

    Returns
    -------
    A, B, C : np.ndarray
        Initial factor matrices; A and B are non-negative.
    """
    if strategy == "random":
        return generate_covll1_factors(model.dims, model.rank, model.L1, model.L2)
    if strategy == "kmeans":
        if T is None:
            raise ValueError("K-means init requires input data T.")
        return _kmeans_init(T, model.rank, model.L1, model.get_constraint_matrix())
    raise unknown_option_error("initialization", strategy, INIT_STRATEGIES)


def _kmeans_init(T, R, L1, theta):
    """K-means initialization for the Cov-LL1 model.

    Clusters the pixels by their covariance matrices, factors each cluster's
    map with NMF to get non-negative A and B, and builds C from the cluster
    centers.
    """
    N1, N2, K2 = T.shape
    K = int(np.sqrt(K2))

    # cluster pixels by their (real-packed) covariance matrices
    labels, centers = cluster_pixels(
        hermitian_to_real(T.reshape(N1, N2, K, K), dtype="float32"), R
    )

    # NMF on each cluster's map gives A_r, B_r
    maps = cluster_maps(labels, R, (N1, N2), random_weights=True)
    nmf = NMF(n_components=L1, init="nndsvda", random_state=0, max_iter=1000)
    initA, initB = nmf_factors(maps, nmf)

    # initialize C from the cluster centers
    initC = np.zeros((K2, R), dtype=np.complex128)
    for r in range(R):
        center = centers[:, r].reshape(K, K).astype(np.float64)
        lower = 1j * np.tril(center, -1)
        center_matrix = np.diag(np.diag(center)) + lower + lower.conj().T
        initC[:, r] = _project_psd_svd(center_matrix).reshape(K2)

    # scale to match the tensor norm
    initT = factors_to_tensor(initA, initB, initC, theta)
    initA, initB = match_norm(initA, initB, T, initT)

    return initA, initB, initC
