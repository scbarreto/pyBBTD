"""AO-ADMM solver for the BBTD model of covariance images.

The constrained model has non-negative spatial maps ``A_r @ B_r.T`` and
conjugate-symmetric spectral factors ``D = C*``, so the second block of each
term is a covariance matrix ``C_r @ C_r^H``.
"""

import logging

import numpy as np
from scipy.linalg import block_diag
from sklearn.decomposition import NMF
from tensorly import unfold
from tensorly.cp_tensor import cp_to_tensor
from tensorly.tenalg import khatri_rao

from pybbtd._linalg import hermitian_to_real, project_psd
from pybbtd._utils import check_solver_inputs, unknown_option_error, warn_user
from pybbtd.bbtd import BBTD, _constraint_matrices, factors_to_tensor
from pybbtd.btd import _constraint_matrix, spatial_maps
from pybbtd.solvers._kmeans import cluster_maps, cluster_pixels, match_norm, nmf_factors

logger = logging.getLogger(__name__)

INIT_STRATEGIES = ("random", "svd", "kmeans")


def fit_bbtd_cov_admm(
    model,
    T,
    init="random",
    max_iter=1000,
    gamma=1.0,
    rho=1.0,
    max_admm=50,
    admm_tol=1e-8,
    rel_tol=1e-6,
    abs_tol=1e-12,
):
    """Fit the covariance BBTD with AO-ADMM.

    Enforces non-negative spatial maps ``A_r @ B_r.T`` and conjugate
    symmetry ``D = C*``.

    Parameters
    ----------
    model : BBTD
        Model defining the dimensions and ranks. The data have shape
        ``(I, J, K, K)``.
    T : np.ndarray
        Tensor of shape ``model.dims`` whose pixels ``T[i, j]`` are
        ``K x K`` covariance matrices.
    init : {"random", "svd", "kmeans"}, default "random"
        Initialization strategy (see :func:`init_bbtd_cov_factors`).
    max_iter : int, default 1000
        Maximum number of outer iterations.
    gamma : float, default 1.0
        Weight of the penalty tying ``A_r @ B_r.T`` to its non-negative part.
    rho : float, default 1.0
        ADMM penalty parameter for the constraint ``D = C*``.
    max_admm : int, default 50
        Maximum number of inner ADMM iterations for the C, D update.
    admm_tol : float, default 1e-8
        Tolerance of the inner ADMM loop.
    rel_tol : float, default 1e-6
        Stop when the relative change of the fit error falls below this value.
    abs_tol : float, default 1e-12
        Stop when the fit error falls below this value.

    Returns
    -------
    factors : list of np.ndarray
        Estimated ``[A, B, C, D]``, with ``D = C.conj()``.
    fit_error : np.ndarray
        Relative reconstruction error after each iteration.
    """
    check_solver_inputs(model, BBTD, T)

    J, M, K, K = model.dims
    R = model.rank
    L1 = model.L1
    L2 = model.L2

    # precompute constraint matrices
    phi, psi = model.get_constraint_matrices()
    psibtd = _constraint_matrix(R, L2)

    # precompute unfoldings for A, B (4D)
    T1 = unfold(T, 0)
    T2 = unfold(T, 1)

    # reshape to 3D for C, D updates
    T3D = T.reshape(J * M, K, K)
    T3 = unfold(T3D, 1)
    T4 = unfold(T3D, 2)

    # init factors
    Aest, Best, Cest = init_bbtd_cov_factors(model, strategy=init, T=T)
    Dest = Cest.conj()

    # initial spatial maps, scaled to match the data norm
    Sout = spatial_maps(Aest, Best, L1)
    vecSout = Sout.reshape(R, J * M).T
    tensor_init = cp_to_tensor((np.ones(L2 * R), [vecSout @ psibtd, Cest, Dest]))
    scale_factor = np.linalg.norm(T3D) / np.linalg.norm(tensor_init)
    Sout = Sout * scale_factor

    cost = np.zeros(max_iter)
    progress_interval = max(max_iter // 5, 1)
    eps_zero = 1e-13

    for it in range(max_iter):
        if it != 0 and it % progress_interval == 0:
            logger.info("Progress: %.0f%%", it / max_iter * 100)

        # update A
        M1 = phi @ khatri_rao((Best @ phi, Cest @ psi, Cest.conj() @ psi)).T
        Aest = _solve_a(Best, Sout, T1, M1, gamma, L1)
        Aest = np.nan_to_num(Aest, nan=0, posinf=0, neginf=0)

        # update B
        M2 = phi @ khatri_rao((Aest @ phi, Cest @ psi, Cest.conj() @ psi)).T
        Best = _solve_b(Aest, Sout, T2, M2, gamma, L1)
        Best = np.nan_to_num(Best, nan=0, posinf=0, neginf=0)

        # project spatial maps to non-negative
        Sout = np.maximum(0, spatial_maps(Aest, Best, L1))
        vecSout = Sout.reshape(R, J * M).T

        # update C and D via ADMM (enforcing D = C*)
        Cest, Dest = _admm_cd(
            Cest, Dest, T3, T4, vecSout, psibtd, rho, max_admm, admm_tol, L2, R
        )

        # compute reconstruction error
        tensor_rec = cp_to_tensor((np.ones(L2 * R), [vecSout @ psibtd, Cest, Dest]))
        cost[it] = np.linalg.norm(T3D - tensor_rec) / np.linalg.norm(T3D)

        # exit criterion
        if it > 0:
            change = np.abs(cost[it] - cost[it - 1])
            if (
                change / (cost[it] + eps_zero) < rel_tol
                or change / (cost[it - 1] + eps_zero) < rel_tol
            ):
                logger.info("Iteration %d: fit error stopped changing (rel_tol).", it)
                cost = cost[: it + 1]
                break
        if cost[it] < abs_tol:
            logger.info("Iteration %d: fit error below abs_tol.", it)
            cost = cost[: it + 1]
            break
    else:
        warn_user(
            f"Reached max_iter={max_iter} before converging. Check the fit error.",
            RuntimeWarning,
        )

    return [Aest, Best, Cest, Dest], cost


def _solve_a(estB, estS, T1, M1, gamma, L1):
    """Update A: least squares on the first unfolding plus the spatial-map penalty.

    Minimizes ``||T1 - A @ M1||^2 + gamma * sum_r ||S_r - A_r @ B_r.T||^2``.
    """
    R = estS.shape[0]

    M1_M1H = (M1 @ M1.T.conj()).real
    RHS = (T1.conj() @ M1.T).real

    Brs = []
    SBs = []
    for r in range(R):
        B_r = estB[:, r * L1 : (r + 1) * L1]
        Brs.append(B_r.T @ B_r)
        SBs.append(estS[r] @ B_r)
    BTB = block_diag(*Brs)
    SBfinal = np.hstack(SBs)

    term1 = np.nan_to_num(M1_M1H + gamma * BTB, nan=0, posinf=0, neginf=0)
    term2 = np.nan_to_num(RHS + gamma * SBfinal, nan=0, posinf=0, neginf=0)

    return np.linalg.solve(term1.T, term2.T).T


def _solve_b(estA, estS, T2, M2, gamma, L1):
    """Update B: least squares on the second unfolding plus the spatial-map penalty.

    Minimizes ``||T2 - B @ M2||^2 + gamma * sum_r ||S_r - A_r @ B_r.T||^2``.
    """
    R = estS.shape[0]

    M2_M2H = (M2 @ M2.T.conj()).real
    RHS = (T2.conj() @ M2.T).real

    ATA_blocks = []
    SAfinal = np.zeros((estS.shape[2], R * L1))
    for r in range(R):
        A_r = estA[:, r * L1 : (r + 1) * L1]
        ATA_blocks.append(A_r.T @ A_r)
        SAfinal[:, r * L1 : (r + 1) * L1] = estS[r].T @ A_r
    ATA = block_diag(*ATA_blocks)

    term1 = np.nan_to_num(M2_M2H + gamma * ATA, nan=0, posinf=0, neginf=0)
    term2 = np.nan_to_num(RHS + gamma * SAfinal, nan=0, posinf=0, neginf=0)

    return np.linalg.solve(term1.T, term2.T).T


def _admm_cd(estC, estD, T3, T4, vecS0, psibtd, rho, max_admm, admm_tol, L2, R):
    """ADMM update of C and D enforcing the conjugate constraint ``D = C*``.

    C and D are updated in turn, with a dual variable pushing ``D`` towards
    ``C*``. The result is the average of ``C`` and ``D*``, with each block
    re-expressed through its SVD.
    """
    talincr = taldecr = 2
    mi = 10

    eps_pri = np.size(estC) ** (1 / 2) * admm_tol
    eps_dual = np.size(estC) ** (1 / 2) * admm_tol

    n = 0
    primal_residue = np.zeros(max_admm)
    dual_residue = np.zeros(max_admm)
    Ul = np.zeros_like(estC)
    exit_criterion = False

    while n < max_admm and not exit_criterion:
        # update C
        M3 = khatri_rao((vecS0 @ psibtd, estD)).T
        term1 = M3 @ M3.T.conj() + rho * np.eye(estC.shape[1])
        term2 = T3 @ M3.T.conj() + rho * (estD.conj() - Ul)
        estC1 = np.linalg.solve(term1.T, term2.T).T

        # update D
        M4 = khatri_rao((vecS0 @ psibtd, estC1)).T
        term1 = M4 @ M4.T.conj() + rho * np.eye(estD.shape[1])
        term2 = T4 @ M4.T.conj() + rho * (estC1.conj() + Ul.conj())
        estD1 = np.linalg.solve(term1.T, term2.T).T

        # update dual variable
        Ul1 = Ul + estC1 - estD1.conj()

        # compute primal and dual residuals
        primal_residue[n] = np.linalg.norm(estC1 - estD1.conj()) / np.linalg.norm(estC1)
        dual_residue[n] = rho * np.linalg.norm(estD1 - estD) / np.linalg.norm(estD1)

        estC = estC1
        estD = estD1
        Ul = Ul1

        if (primal_residue[n] < eps_pri) and (dual_residue[n] < eps_dual):
            exit_criterion = True

        # adjust rho based on residuals
        if primal_residue[n] > (mi * dual_residue[n]):
            rho *= talincr
            Ul /= talincr
        elif dual_residue[n] > (mi * primal_residue[n]):
            rho /= taldecr
            Ul *= taldecr

        # stop early if both residuals stagnate
        if n > 0:
            incr_pri = (
                np.abs(primal_residue[n] - primal_residue[n - 1]) / primal_residue[n]
            )
            incr_dual = np.abs(dual_residue[n] - dual_residue[n - 1]) / dual_residue[n]
            if (incr_pri < admm_tol) and (incr_dual < admm_tol):
                exit_criterion = True

        n += 1

    # average C and D* for the final estimate
    estC = (estC1 + estD1.conj()) / 2

    # re-express each block of C through its SVD
    newC = np.zeros_like(estC)
    for r in range(R):
        U, s, _ = np.linalg.svd(estC[:, r * L2 : (r + 1) * L2])
        newC[:, r * L2 : (r + 1) * L2] = U[:, :L2] @ np.diag(s[:L2])

    return newC, newC.conj()


def init_bbtd_cov_factors(model, strategy="random", T=None):
    """Initialize the factors of the covariance BBTD.

    A and B are non-negative and C is complex; ``D = C.conj()`` is implied.

    Parameters
    ----------
    model : BBTD
        Model defining the dimensions and ranks.
    strategy : {"random", "svd", "kmeans"}, default "random"
        ``"random"`` draws uniform A, B and complex Gaussian C. ``"svd"`` uses
        the leading singular vectors of the unfoldings of ``T`` (absolute
        values for A and B). ``"kmeans"`` clusters the pixels by their
        covariance matrices (see :func:`_kmeans_init`).
    T : np.ndarray, optional
        Data tensor, required for ``"svd"`` and ``"kmeans"``.

    Returns
    -------
    A, B, C : np.ndarray
        Initial factor matrices.
    """
    dims = model.dims
    R = model.rank
    L1 = model.L1
    L2 = model.L2

    if strategy == "random":
        A = np.random.rand(dims[0], L1 * R)
        B = np.random.rand(dims[1], L1 * R)
        C = np.random.randn(dims[2], L2 * R) + 1j * np.random.randn(dims[2], L2 * R)
    elif strategy == "svd":
        if T is None:
            raise ValueError("SVD init requires input data T.")
        u0, s0, _ = np.linalg.svd(unfold(T, 0), full_matrices=False)
        u1, s1, _ = np.linalg.svd(unfold(T, 1), full_matrices=False)
        u2, s2, _ = np.linalg.svd(unfold(T, 2), full_matrices=False)

        n0 = min(len(s0), L1 * R)
        n1 = min(len(s1), L1 * R)
        n2 = min(len(s2), L2 * R)

        A = np.abs(u0[:, :n0] @ np.diag(s0[:n0] ** 0.5))
        B = np.abs(u1[:, :n1] @ np.diag(s1[:n1] ** 0.5))
        C = u2[:, :n2] @ np.diag(s2[:n2] ** 0.5)

        # pad with random columns if the SVD did not produce enough
        if n0 < L1 * R:
            A = np.hstack([A, np.random.rand(dims[0], L1 * R - n0)])
        if n1 < L1 * R:
            B = np.hstack([B, np.random.rand(dims[1], L1 * R - n1)])
        if n2 < L2 * R:
            pad = np.random.randn(dims[2], L2 * R - n2) + 1j * np.random.randn(
                dims[2], L2 * R - n2
            )
            C = np.hstack([C, pad])
    elif strategy == "kmeans":
        if T is None:
            raise ValueError("K-means init requires input data T.")
        A, B, C = _kmeans_init(T, R, L1, L2)
    else:
        raise unknown_option_error("initialization", strategy, INIT_STRATEGIES)

    return A, B, C


def _kmeans_init(T, R, L1, L2):
    """K-means initialization for the covariance BBTD.

    Clusters the pixels by their covariance matrices, factors each cluster's
    indicator map with NMF to get non-negative A and B, and builds C from
    the cluster centers projected onto the positive semidefinite matrices.
    """
    J, M, K, _ = T.shape

    # cluster pixels by their (real-packed) covariance matrices
    labels, centers = cluster_pixels(hermitian_to_real(T), R)

    # NMF on each cluster's indicator map gives A_r, B_r
    maps = cluster_maps(labels, R, (J, M), random_weights=False)
    nmf = NMF(n_components=L1, init="nndsvda", random_state=0, max_iter=5000)
    initA, initB = nmf_factors(maps, nmf)

    # initialize C from the cluster centers
    initC = np.zeros((K, L2 * R), dtype=np.complex128)
    for r in range(R):
        # symmetric matrix from the upper triangle of the center
        upper = np.triu(centers[:, r].reshape(K, K))
        center_matrix = (upper + np.triu(upper, 1).T).astype(complex)

        U, S, _ = np.linalg.svd(project_psd(center_matrix))
        initC[:, r * L2 : (r + 1) * L2] = U[:, :L2] @ np.diag(np.sqrt(S[:L2]))

    # scale to match the tensor norm
    phi, psi = _constraint_matrices(R, L1, L2)
    initT = factors_to_tensor(initA, initB, initC, initC.conj(), phi, psi)
    initA, initB = match_norm(initA, initB, T, initT)

    return initA, initB, initC
