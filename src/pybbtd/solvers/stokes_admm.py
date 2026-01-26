import numpy as np
from tensorly import unfold
from scipy import linalg
from scipy.linalg import solve
import pybbtd.stokes as stokes
from tensorly.tenalg import khatri_rao
import pybbtd.btd as btd
from sklearn.decomposition import NMF
from sklearn.cluster import KMeans
import pybbtd.solvers.covll1_admm as covll1_admm
import warnings


def ADMM_C(Y3, Ak, Bk, Cinit, Ctinit, theta, rho, L, R, nitermax=100, tol=1e-14):
    """
    Inner ADMM update for factor C with Stokes constraint.

    Solves the augmented Lagrangian subproblem for C using the third
    mode unfolding, projecting each column onto the set of physically
    valid Stokes vectors via
    :func:`~pybbtd.stokes.stokesProjection`.

    Parameters:
        Y3: np.ndarray
            Third mode unfolding of the input tensor.
        Ak: np.ndarray
            Current estimate of factor A.
        Bk: np.ndarray
            Current estimate of factor B.
        Cinit: np.ndarray
            Initial primal variable for C.
        Ctinit: np.ndarray
            Initial projected (valid Stokes) variable for C.
        theta: np.ndarray
            Constraint matrix from the BTD model.
        rho: float
            ADMM penalty parameter.
        L: int
            Rank of the spatial maps.
        R: int
            Number of block terms.
        nitermax: int
            Maximum number of inner ADMM iterations (default: 100).
        tol: float
            Convergence tolerance (default: 1e-14).

    Returns:
        Tuple[np.ndarray, np.ndarray, float]:
            ``(Ctilde, C, primal_residue)`` where ``Ctilde`` is the
            Stokes-projected factor, ``C`` is the unconstrained factor,
            and ``primal_residue`` is the final primal residual.
    """
    # init variables

    Cl = Cinit.copy()
    Ctl = Ctinit.copy()
    Ul = np.zeros_like(Cl)

    # iteration
    M3 = theta @ khatri_rao([Ak, Bk]).T
    M3M3T = M3 @ M3.T
    RHS = Y3 @ M3.T

    epsPri = np.size(Cl) ** (1 / 2) * tol
    epsDual = np.size(Ctl) ** (1 / 2) * tol

    n = 0
    primalResidue = epsPri + 1
    dualResidue = epsDual + 1

    exitCriterion = True
    while (n < nitermax) & exitCriterion:
        # update C_(l+1)
        C_T = solve(
            (M3M3T + rho * np.eye(Cinit.shape[1])).T, (RHS + rho * (Ctl - Ul)).T
        )
        Cl1 = C_T.T

        # update \tilde{C}_{l+1}
        Ctl1 = np.zeros_like(Ctl)
        for r in range(R):
            Ctl1[:, r] = stokes.stokesProjection((Cl1 + Ul)[:, r])

        # update dual
        Ul1 = Ul + Cl1 - Ctl1

        # compute convergence metrics residuals
        primalResidue = np.linalg.norm(Cl1 - Ctl1)
        dualResidue = rho * np.linalg.norm(Ctl1 - Ctl)

        if (primalResidue < epsPri) and (dualResidue < epsDual):
            exitCriterion = False

        if (primalResidue < epsPri) and (dualResidue < epsDual):
            exitCriterion = False
        if primalResidue > 10 * dualResidue:
            rho *= 2
            Ul1 /= 2
        elif dualResidue > 10 * primalResidue:
            rho /= 2
            Ul1 *= 2
        # update all variables
        Cl = Cl1.copy()
        Ctl = Ctl1.copy()
        Ul = Ul1.copy()

        n += +1

    return Ctl, Cl, primalResidue


def kmeans_init(T, R, Lr, theta):
    """
    K-means based initialization for the Stokes-BTD decomposition.

    Clusters spatial pixels via K-means, applies NMF on each cluster's
    spatial map to obtain non-negative factors A, B, and builds C from
    the projected cluster centers (valid Stokes vectors).

    Parameters:
        T: np.ndarray
            Input tensor of shape ``(I, J, 4)``.
        R: int
            Number of clusters / block terms.
        Lr: list[int] or np.ndarray
            Rank of the spatial maps. Uses ``Lr[0]`` as the NMF rank.
        theta: np.ndarray
            Constraint matrix from the BTD model.

    Returns:
        Tuple[np.ndarray, np.ndarray, np.ndarray]:
            Initialized factors ``(A, B, C)`` where A, B are non-negative
            and each column of C is a valid Stokes vector.
    """

    # -------------------------
    # 1. KMEANS-BASED INITIALIZATION OF C AND FEATURE MAPS
    # -------------------------

    unfolding = unfold(T, 2).T

    # KMeans clustering
    kmeans = KMeans(n_clusters=R, random_state=0, n_init="auto")
    kmeans.fit(unfolding)

    labels = kmeans.labels_

    # Initial cluster centers (shape ~ [features, R])
    kmeansC = kmeans.cluster_centers_.T

    # Project each cluster center using Stokes projection
    for r in range(R):
        kmeansC[:, r] = stokes.stokesProjection(kmeansC[:, r])

    # Normalize each cluster center so that its first component is 1
    for r in range(R):
        if kmeansC[0, r] > 0:
            kmeansC[:, r] = kmeansC[:, r] / kmeansC[0, r]

    # Rterms is (R, I*J) if T is (I, J, ...)
    Rterms = np.zeros((R, T.shape[0] * T.shape[1]))
    featureMaps = np.ones((R, T.shape[0], T.shape[1]))

    for idx in range(len(labels)):
        Rterms[labels[idx], idx] = np.random.rand()

    # Reshape each cluster's assignment map into spatial maps
    for r in range(R):
        featureMaps[r] = Rterms[r].reshape(T.shape[0], T.shape[1])

    # -------------------------
    # 2. NMF ON FEATURE MAPS TO BUILD A AND B
    # -------------------------

    # NMF rank per block
    L = Lr[0]

    model = NMF(n_components=L, init="random", max_iter=1000)

    # We'll factor each feature map with its own NMF,
    # then concatenate all the W/H across clusters.
    W = np.zeros((R, featureMaps.shape[1], L))
    H = np.zeros((R, L, featureMaps.shape[2]))

    product = np.zeros((R, featureMaps.shape[1], featureMaps.shape[2]))

    for r in range(R):
        W[r] = model.fit_transform(featureMaps[r])  # shape (I, L)
        H[r] = model.components_  # shape (L, J)
        product[r] = W[r] @ H[r]

    # Concatenate per-cluster factors across columns
    kmeansA = W[0]  # shape (I, L)
    kmeansB = H[0].T  # shape (J, L)

    if R > 1:
        for r in range(1, R):
            kmeansA = np.concatenate((kmeansA, W[r]), axis=1)  # (I, R*L)
            kmeansB = np.concatenate((kmeansB, H[r].T), axis=1)  # (J, R*L)

    # -------------------------
    # 3. SCALE A AND B USING GLOBAL NORM MATCHING
    # -------------------------

    Tkmeans = btd.factors_to_tensor(kmeansA, kmeansB, kmeansC, theta)

    lamb = linalg.norm(T) / linalg.norm(Tkmeans)
    scale = lamb**0.5

    kmeansA = kmeansA * scale
    kmeansB = kmeansB * scale

    return kmeansA, kmeansB, kmeansC


def init_Stokes_factors(Stokes_model, init="random", T=None):
    """
    Initialize factor matrices for the Stokes-BTD decomposition.

    Parameters:
        Stokes_model: Stokes
            An instance of the :class:`~pybbtd.stokes.Stokes` class.
        init: str
            Initialization strategy (default: ``"random"``).
            Options: ``"random"``, ``"kmeans"``.
        T: np.ndarray or None
            Input tensor (required for ``"kmeans"`` initialization).

    Returns:
        Tuple[np.ndarray, np.ndarray, np.ndarray]:
            Initialized factor matrices ``(A, B, C)`` where A, B are
            non-negative and each column of C is a valid Stokes vector.
    """
    import pybbtd.stokes as stokes

    theta = Stokes_model.get_constraint_matrix()
    # get BTD model params
    dims = Stokes_model.dims
    R = Stokes_model.rank
    L = Stokes_model.L

    if init == "random":
        A, B, C = stokes.generate_stokes_factors(dims, R, L)

    elif init == "kmeans":
        A, B, C = kmeans_init(T, R, L, theta)

    return A, B, C


def Stokes_ADMM(
    Stokes_model,
    T,
    init="random",
    max_iter=1000,
    rho=1,
    max_admm=1,
    rel_tol=10**-10,
    abs_tol=10**-10,
    admm_tol=10**-8,
):
    """
    AO-ADMM solver for the Stokes-constrained BTD-LL1 decomposition.

    Enforces non-negativity on spatial factors A, B and Stokes
    physical constraints on C via alternating ADMM updates.

    Parameters:
        Stokes_model: Stokes
            An instance of the :class:`~pybbtd.stokes.Stokes` class.
        T: np.ndarray
            Input tensor of shape ``(I, J, 4)`` to be decomposed.
        init: str
            Initialization strategy (default: ``"random"``).
            Options: ``"random"``, ``"kmeans"``.
        max_iter: int
            Maximum number of outer iterations (default: 1000).
        rho: float
            ADMM penalty parameter (default: 1).
        max_admm: int
            Maximum number of inner ADMM iterations per factor update
            (default: 1).
        rel_tol: float
            Relative tolerance for outer convergence (default: 1e-10).
        abs_tol: float
            Absolute tolerance for outer convergence (default: 1e-10).
        admm_tol: float
            Tolerance for inner ADMM convergence (default: 1e-8).

    Returns:
        Tuple[list, list]:
            ``(factors, fit_error)`` where ``factors = [A, B, C]`` and
            ``fit_error`` is a list of squared reconstruction errors.
    """
    import pybbtd.stokes as stokes

    # Check that Stokes_model is an instance of the Stokes class
    if not isinstance(Stokes_model, stokes.Stokes):
        raise TypeError("Stokes_model must be an instance of the Stokes class.")

    # Check that T is a numpy array
    if not isinstance(T, np.ndarray):
        raise TypeError("T must be a numpy array.")

    # Check that T's dimensions match Stokes_model.dims
    if T.shape != Stokes_model.dims:
        raise ValueError(
            f"T's dimensions ({T.shape}) do not match Stokes_model.dims ({Stokes_model.dims})."
        )
    if init == "random":
        Ak, Bk, Ck = init_Stokes_factors(Stokes_model, init="random")
        Atk, Btk, Ctk = init_Stokes_factors(Stokes_model, init="random")
    elif init == "kmeans":
        Atk, Btk, Ctk = init_Stokes_factors(Stokes_model, "kmeans", T)
        Ak, Bk, Ck = init_Stokes_factors(Stokes_model, "random")
    else:
        raise ValueError("not implemented")

    T1 = unfold(T, 0)
    T2 = unfold(T, 1)
    T3 = unfold(T, 2)

    theta = Stokes_model.get_constraint_matrix()

    L = Stokes_model.L
    R = Stokes_model.rank

    Tfit_0 = btd.factors_to_tensor(Atk, Btk, Ctk, theta)
    fit_error = [np.linalg.norm(Tfit_0 - T) ** 2]

    k = 0
    exit_criterion = False
    while exit_criterion is False:
        if (k != 0) and (k % int(max_iter / 5)) == 0:
            print("Progress:", (k / max_iter * 100), "%")
        # update A
        Atk1, Ak1, _ = covll1_admm.ADMM_A(
            T1, Btk, (Ctk @ theta), Ak, Atk, rho, L, nitermax=max_admm, tol=admm_tol
        )

        # update B
        Btk1, Bk1, _ = covll1_admm.ADMM_B(
            T2, Atk1, (Ctk @ theta), Bk, Btk, rho, L, nitermax=max_admm, tol=admm_tol
        )

        # update C
        Ctk1, Ck1, _ = ADMM_C(
            T3, Atk1, Btk1, Ck, Ctk, theta, rho, L, R, nitermax=max_admm, tol=admm_tol
        )

        # update variables
        Ak = Ak1.copy()
        Bk = Bk1.copy()
        Ck = Ck1.copy()

        Atk = Atk1.copy()
        Btk = Btk1.copy()
        Ctk = Ctk1.copy()

        estimated_factors = [Atk1, Btk1, Ctk1]
        k += 1

        # compute reconstruction error

        Tfit_k = btd.factors_to_tensor(Atk1, Btk1, Ctk1, theta)
        fit_error.append(np.linalg.norm(Tfit_k - T) ** 2)

        if np.abs(fit_error[-1] - fit_error[-2]) / fit_error[-1] < rel_tol:
            print("Exiting early due to unsufficient decrease of cost")
            exit_criterion = True
        if fit_error[-1] / np.linalg.norm(T) < abs_tol:
            print("Reached absolute tolerance threshold. Exiting.")
            exit_criterion = True

        if k >= max_iter:
            exit_criterion = True
            warnings.warn(
                "Reached max number of iteration. Check convergence.", RuntimeWarning
            )

    return estimated_factors, fit_error
