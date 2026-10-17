"""AO-ADMM solver for the Stokes model."""

import logging

import numpy as np
from sklearn.decomposition import NMF
from tensorly import unfold
from tensorly.tenalg import khatri_rao

from pybbtd._utils import check_solver_inputs, unknown_option_error, warn_user
from pybbtd.btd import factors_to_tensor
from pybbtd.solvers._admm import admm_update, project_nonnegative
from pybbtd.solvers._kmeans import cluster_maps, cluster_pixels, match_norm, nmf_factors
from pybbtd.stokes import Stokes, generate_stokes_factors, project_stokes

logger = logging.getLogger(__name__)

INIT_STRATEGIES = ("random", "kmeans")


def fit_stokes_admm(
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
    """Fit a Stokes model with AO-ADMM.

    The factors are updated in turn (A, then B, then C), each with a few
    ADMM iterations. A and B are kept non-negative and every column of C is
    kept a valid Stokes vector.

    Parameters
    ----------
    model : Stokes
        Model defining the dimensions and ranks.
    T : np.ndarray
        Tensor of shape ``(I, J, 4)`` to decompose.
    init : {"random", "kmeans"}, default "random"
        Initialization strategy (see :func:`init_stokes_factors`).
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
        Squared reconstruction error :math:`\\|\\mathcal{T} - \\hat{\\mathcal{T}}\\|^2`
        at initialization and after each iteration.
    """
    check_solver_inputs(model, Stokes, T)

    if init == "random":
        Ak, Bk, Ck = init_stokes_factors(model, strategy="random")
        Atk, Btk, Ctk = init_stokes_factors(model, strategy="random")
    elif init == "kmeans":
        Atk, Btk, Ctk = init_stokes_factors(model, strategy="kmeans", T=T)
        Ak, Bk, Ck = init_stokes_factors(model, strategy="random")
    else:
        raise unknown_option_error("initialization", init, INIT_STRATEGIES)

    T1 = unfold(T, 0)
    T2 = unfold(T, 1)
    T3 = unfold(T, 2)

    theta = model.get_constraint_matrix()

    Tfit_0 = factors_to_tensor(Atk, Btk, Ctk, theta)
    fit_error = [np.linalg.norm(Tfit_0 - T) ** 2]

    progress_interval = max(max_iter // 5, 1)
    k = 0
    exit_criterion = False
    while not exit_criterion:
        if k != 0 and k % progress_interval == 0:
            logger.info("Progress: %.0f%%", k / max_iter * 100)

        # update A
        M1 = khatri_rao([Btk, Ctk @ theta]).T
        Atk1, Ak1 = admm_update(
            T1, M1, Ak, Atk, rho, project_nonnegative, max_admm, admm_tol, real=True
        )

        # update B
        M2 = khatri_rao([Atk1, Ctk @ theta]).T
        Btk1, Bk1 = admm_update(
            T2, M2, Bk, Btk, rho, project_nonnegative, max_admm, admm_tol, real=True
        )

        # update C
        M3 = theta @ khatri_rao([Atk1, Btk1]).T
        Ctk1, Ck1 = admm_update(
            T3,
            M3,
            Ck,
            Ctk,
            rho,
            _project_stokes_columns,
            max_iter=max_admm,
            tol=admm_tol,
        )

        # update variables
        Ak, Bk, Ck = Ak1, Bk1, Ck1
        Atk, Btk, Ctk = Atk1, Btk1, Ctk1
        k += 1

        # compute reconstruction error
        Tfit_k = factors_to_tensor(Atk1, Btk1, Ctk1, theta)
        fit_error.append(np.linalg.norm(Tfit_k - T) ** 2)

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


def _project_stokes_columns(C):
    """Project every column of C onto the valid Stokes vectors."""
    return np.column_stack([project_stokes(column) for column in C.T])


def init_stokes_factors(model, strategy="random", T=None):
    """Initialize the Stokes factors.

    Parameters
    ----------
    model : Stokes
        Model defining the dimensions and ranks.
    strategy : {"random", "kmeans"}, default "random"
        ``"random"`` draws factors with
        :func:`~pybbtd.stokes.generate_stokes_factors`. ``"kmeans"`` clusters
        the pixels by their Stokes vectors (see :func:`_kmeans_init`).
    T : np.ndarray, optional
        Data tensor, required for ``"kmeans"``.

    Returns
    -------
    A, B, C : np.ndarray
        Initial factor matrices; A and B are non-negative and each column of
        C is a valid Stokes vector.
    """
    if strategy == "random":
        return generate_stokes_factors(model.dims, model.rank, model.L)
    if strategy == "kmeans":
        if T is None:
            raise ValueError("K-means init requires input data T.")
        return _kmeans_init(T, model.rank, model.L, model.get_constraint_matrix())
    raise unknown_option_error("initialization", strategy, INIT_STRATEGIES)


def _kmeans_init(T, R, L, theta):
    """K-means initialization for the Stokes model.

    Clusters the pixels by their Stokes vectors, factors each cluster's map
    with NMF to get non-negative A and B, and uses the projected cluster
    centers (normalized to unit intensity) as C.
    """
    # cluster pixels by their Stokes vectors
    labels, centers = cluster_pixels(T, R)

    # project each center onto the valid Stokes vectors, with unit intensity
    C = _project_stokes_columns(centers)
    for r in range(R):
        if C[0, r] > 0:
            C[:, r] = C[:, r] / C[0, r]

    # NMF on each cluster's map gives A_r, B_r
    maps = cluster_maps(labels, R, T.shape[:2], random_weights=True)
    nmf = NMF(n_components=L[0], init="random", max_iter=1000)
    A, B = nmf_factors(maps, nmf)

    # scale to match the tensor norm
    A, B = match_norm(A, B, T, factors_to_tensor(A, B, C, theta))

    return A, B, C
