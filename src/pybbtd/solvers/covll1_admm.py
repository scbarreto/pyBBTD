import numpy as np
from tensorly import unfold
from scipy import linalg
from scipy.linalg import solve
from tensorly.tenalg import khatri_rao
import warnings
import pybbtd.btd as btd
from sklearn.cluster import KMeans
from sklearn.decomposition import NMF


def ADMM_A(Y1, Bk, Ck, Ainit, Atinit, rho, L, nitermax=100, tol=1e-14):
    """
    Inner ADMM update for factor A with non-negativity constraint.

    Solves the augmented Lagrangian subproblem for A using the first
    mode unfolding, with a proximal non-negativity projection.

    Parameters:
        Y1: np.ndarray
            First mode unfolding of the input tensor.
        Bk: np.ndarray
            Current estimate of factor B.
        Ck: np.ndarray
            Current estimate of factor C (pre-multiplied by theta).
        Ainit: np.ndarray
            Initial primal variable for A.
        Atinit: np.ndarray
            Initial projected (non-negative) variable for A.
        rho: float
            ADMM penalty parameter.
        L: int
            Rank of the spatial maps.
        nitermax: int
            Maximum number of inner ADMM iterations (default: 100).
        tol: float
            Convergence tolerance (default: 1e-14).

    Returns:
        Tuple[np.ndarray, np.ndarray, float]:
            ``(Atilde, A, primal_residue)`` where ``Atilde`` is the
            non-negative projected factor, ``A`` is the unconstrained
            factor, and ``primal_residue`` is the final primal residual.
    """
    # init variables
    Al = Ainit.copy()
    Atl = Atinit.copy()
    Ul = np.zeros_like(Ainit)

    M1 = khatri_rao([Bk, Ck]).T

    # Build system
    M1M1T = M1 @ M1.T  # K x K
    RHS = Y1 @ M1.T  # I x K

    epsPri = np.size(Al) ** (1 / 2) * tol
    epsDual = np.size(Atl) ** (1 / 2) * tol

    n = 0
    primalResidue = epsPri + 1
    dualResidue = epsDual + 1

    exitCriterion = True
    while (n < nitermax) & exitCriterion:
        # update A_{l+1}
        # Solve A @ X = B  ⇒  A = B @ inv(X)
        A_T = solve(
            (M1M1T + rho * np.eye(Ainit.shape[1])).T, (RHS + rho * (Atl - Ul)).T
        )
        Al1 = A_T.T

        # update \tilde{A}_{l+1}
        Atl1 = np.maximum(0, Al1 + Ul)

        # update dual
        Ul1 = Ul + Al1 - Atl1

        # compute convergence metrics residuals
        primalResidue = np.linalg.norm(Al1 - Atl1)
        dualResidue = rho * np.linalg.norm(Atl1 - Atl)

        if (primalResidue < epsPri) and (dualResidue < epsDual):
            exitCriterion = False
        if primalResidue > 10 * dualResidue:
            rho *= 2
            Ul1 /= 2
        elif dualResidue > 10 * primalResidue:
            rho /= 2
            Ul1 *= 2
        # update all variables
        Al = Al1.copy()
        Atl = Atl1.copy()
        Ul = Ul1.copy()

        n += +1

    return Atl, Al, primalResidue


def ADMM_B(Y2, Ak, Ck, Binit, Btinit, rho, L, nitermax=100, tol=1e-14):
    """
    Inner ADMM update for factor B with non-negativity constraint.

    Solves the augmented Lagrangian subproblem for B using the second
    mode unfolding, with a proximal non-negativity projection.

    Parameters:
        Y2: np.ndarray
            Second mode unfolding of the input tensor.
        Ak: np.ndarray
            Current estimate of factor A.
        Ck: np.ndarray
            Current estimate of factor C (pre-multiplied by theta).
        Binit: np.ndarray
            Initial primal variable for B.
        Btinit: np.ndarray
            Initial projected (non-negative) variable for B.
        rho: float
            ADMM penalty parameter.
        L: int
            Rank of the spatial maps.
        nitermax: int
            Maximum number of inner ADMM iterations (default: 100).
        tol: float
            Convergence tolerance (default: 1e-14).

    Returns:
        Tuple[np.ndarray, np.ndarray, float]:
            ``(Btilde, B, primal_residue)`` where ``Btilde`` is the
            non-negative projected factor, ``B`` is the unconstrained
            factor, and ``primal_residue`` is the final primal residual.
    """
    # init variables
    Bl = Binit.copy()
    Btl = Btinit.copy()
    Ul = np.zeros_like(Bl)

    # iteration
    M2 = khatri_rao([Ak, Ck]).T

    # Build system
    M2M2T = M2 @ M2.T  # K x K
    RHS = Y2 @ M2.T  # I x K

    epsPri = np.size(Bl) ** (1 / 2) * tol
    epsDual = np.size(Btl) ** (1 / 2) * tol

    n = 0
    primalResidue = epsPri + 1
    dualResidue = epsDual + 1

    exitCriterion = True
    while (n < nitermax) & exitCriterion:
        # update B_(l+1)
        B_T = solve(
            (M2M2T + rho * np.eye(Binit.shape[1])).T, (RHS + rho * (Btl - Ul)).T
        )
        Bl1 = B_T.T

        # update \tilde{B}_{l+1}
        Btl1 = np.maximum(0, Bl1 + Ul)

        # update dual
        Ul1 = Ul + Bl1 - Btl1

        # compute convergence metrics residuals

        primalResidue = np.linalg.norm(Bl1 - Btl1)
        dualResidue = rho * np.linalg.norm(Btl1 - Btl)

        if (primalResidue < epsPri) and (dualResidue < epsDual):
            exitCriterion = False
        if primalResidue > 10 * dualResidue:
            rho *= 2
            Ul1 /= 2
        elif dualResidue > 10 * primalResidue:
            rho /= 2
            Ul1 *= 2
        # update all variables
        Bl = Bl1.copy()
        Btl = Btl1.copy()
        Ul = Ul1.copy()

        n += +1

    return Btl, Bl, primalResidue


def ADMM_C(Y3, Ak, Bk, Cinit, Ctinit, theta, rho, L, R, nitermax=100, tol=1e-14):
    """
    Inner ADMM update for factor C with PSD covariance constraint.

    Solves the augmented Lagrangian subproblem for C using the third
    mode unfolding, projecting each column onto the positive
    semidefinite cone via :func:`project_to_psd`.

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
            Initial projected (PSD) variable for C.
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
            PSD-projected factor, ``C`` is the unconstrained factor,
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
        Cl1 += Ul
        for r in range(R):
            cov_matrix = Cl1[:, r].reshape(
                int(np.sqrt(Cinit.shape[0])), int(np.sqrt(Cinit.shape[0]))
            )
            vec_cov, _ = project_to_psd(cov_matrix)
            Ctl1[:, r] = vec_cov

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


def project_to_psd(cov_matrix):
    """
    Project a covariance matrix onto the PSD (positive semidefinite) cone.

    Computes the SVD and clips negative singular values to zero, then
    reconstructs the matrix.

    Parameters:
        cov_matrix: np.ndarray
            Covariance matrix of shape ``(K, K)``.

    Returns:
        Tuple[np.ndarray, np.ndarray]:
            ``(vec_cov_psd, cov_matrix_psd)`` where ``vec_cov_psd`` is
            the vectorized PSD-projected matrix of shape ``(K^2,)`` and
            ``cov_matrix_psd`` is the ``(K, K)`` PSD matrix.
    """
    # SVD decomposition
    U, s, Vh = linalg.svd(cov_matrix)

    # Project onto PSD cone: zero out negative singular values
    s_psd = np.clip(s, a_min=0, a_max=None)

    # Reconstruct PSD matrix
    cov_matrix_psd = (U * s_psd) @ Vh

    # Return vectorized PSD version
    vec_cov_psd = cov_matrix_psd.reshape(-1)
    return vec_cov_psd, cov_matrix_psd


def COVLL1_ADMM(
    CovLL1_model,
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
    AO-ADMM solver for the Cov-LL1 decomposition.

    Enforces non-negativity on spatial factors A, B and positive
    semidefiniteness on the covariance columns of C via alternating
    ADMM updates.

    Parameters:
        CovLL1_model: CovLL1
            An instance of the :class:`~pybbtd.covll1.CovLL1` class.
        T: np.ndarray
            Input tensor of shape ``(I, J, K^2)`` to be decomposed.
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
            ``fit_error`` is a list of normalized squared reconstruction
            errors.
    """
    if init == "random":
        Ak, Bk, Ck = init_COVLL1_factors(CovLL1_model, init="random")
        Atk, Btk, Ctk = init_COVLL1_factors(CovLL1_model, init="random")
    elif init == "kmeans":
        Atk, Btk, Ctk = init_COVLL1_factors(CovLL1_model, "kmeans", T)
        Ak, Bk, Ck = init_COVLL1_factors(CovLL1_model, "random")
    else:
        raise ValueError("not implemented")

    T1 = unfold(T, 0)
    T2 = unfold(T, 1)
    T3 = unfold(T, 2)

    L1 = CovLL1_model.L1
    R = CovLL1_model.rank

    theta = CovLL1_model.get_constraint_matrix()

    Tfit_0 = btd.factors_to_tensor(Atk, Btk, Ctk, theta)
    fit_error = [np.linalg.norm(Tfit_0 - T) ** 2 / np.linalg.norm(T) ** 2]

    k = 0
    exit_criterion = False
    while exit_criterion is False:
        if (k != 0) and (k % int(max_iter / 5)) == 0:
            print("Progress:", (k / max_iter * 100), "%")
        # update A
        Atk1, Ak1, _ = ADMM_A(
            T1, Btk, (Ctk @ theta), Ak, Atk, rho, L1, nitermax=max_admm, tol=admm_tol
        )

        # update B
        Btk1, Bk1, _ = ADMM_B(
            T2, Atk1, (Ctk @ theta), Bk, Btk, rho, L1, nitermax=max_admm, tol=admm_tol
        )

        # update C
        Ctk1, Ck1, _ = ADMM_C(
            T3, Atk1, Btk1, Ck, Ctk, theta, rho, L1, R, nitermax=max_admm, tol=admm_tol
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

        fit_error.append(np.linalg.norm(Tfit_k - T) ** 2 / np.linalg.norm(T) ** 2)

        if np.abs(fit_error[-1] - fit_error[-2]) / fit_error[-1] < rel_tol:
            print("Exiting early due to insufficient decrease of cost")
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


def init_COVLL1_factors(CovLL1_model, init="random", T=None):
    """
    Initialize factor matrices for the Cov-LL1 decomposition.

    Parameters:
        CovLL1_model: CovLL1
            An instance of the :class:`~pybbtd.covll1.CovLL1` class.
        init: str
            Initialization strategy (default: ``"random"``).
            Options: ``"random"``, ``"kmeans"``.
        T: np.ndarray or None
            Input tensor (required for ``"kmeans"`` initialization).

    Returns:
        Tuple[np.ndarray, np.ndarray, np.ndarray]:
            Initialized factor matrices ``(A, B, C)`` where A, B are
            non-negative and C contains vectorized covariance columns.
    """
    import pybbtd.covll1 as covll1

    # get BTD model params
    dims = CovLL1_model.dims
    R = CovLL1_model.rank
    L1 = CovLL1_model.L1
    L2 = CovLL1_model.L2

    theta = CovLL1_model.get_constraint_matrix()

    if init == "random":
        A, B, C = covll1.generate_covll1_factors(dims, R, L1, L2)

    elif init == "kmeans":
        A, B, C = kmeans_init(T, R, L1, theta)

    return A, B, C


def kmeans_init(T, R, L1, theta):
    """
    K-means based initialization for the Cov-LL1 decomposition.

    Clusters the spatial pixels by their covariance vectors using K-means,
    then applies NMF on each cluster's spatial map to obtain non-negative
    factors A, B. The covariance factor C is initialized from the cluster
    centers projected onto the PSD cone.

    Parameters:
        T: np.ndarray
            Input tensor of shape ``(I, J, K^2)`` to be decomposed.
        R: int
            Number of block terms.
        L1: int
            Rank of the spatial maps (NMF rank per cluster).
        theta: np.ndarray
            Constraint matrix from the BTD model.

    Returns:
        Tuple[np.ndarray, np.ndarray, np.ndarray]:
            Initialized factor matrices ``(A, B, C)`` where A, B are
            non-negative and C contains vectorized PSD covariance columns.
    """
    Treal = _complex_to_real(T)

    kmeans = KMeans(n_clusters=R, random_state=0, n_init="auto")

    # Initialize the KMeans model
    unfolding = unfold(Treal, 2).T
    kmeans.fit(unfolding)
    # Get the labels for each vectorized pixel
    labels = kmeans.labels_

    Rterms = np.zeros((R, Treal.shape[0] * Treal.shape[1]))

    featuresMaps = np.ones((R, Treal.shape[0], Treal.shape[1]))

    # for i in range(len(labels)):
    #     Rterms[labels[i], i] = 1

    for idx in range(len(labels)):
        Rterms[labels[idx], idx] = np.random.rand()

    for r in range(R):
        featuresMaps[r] = Rterms[r].reshape(Treal.shape[0], Treal.shape[1])

    model = NMF(n_components=L1, init="nndsvda", random_state=0, max_iter=1000)

    W = np.zeros((R, featuresMaps.shape[1], L1))
    H = np.zeros((R, L1, featuresMaps.shape[2]))

    product = np.zeros((R, featuresMaps.shape[1], featuresMaps.shape[2]))

    for r in range(R):
        Wr = model.fit_transform(featuresMaps[r])
        Hr = model.components_
        W[r] = Wr
        H[r] = Hr
        product[r] = Wr @ Hr

    initA = W[0]
    initB = H[0].T

    if R > 1:
        for i in range(1, R):
            initA = np.concatenate((initA, W[i]), axis=1)
            initB = np.concatenate((initB, H[i].T), axis=1)

    K = int(np.sqrt(T.shape[2]))
    initC = np.zeros((K**2, R), dtype=np.complex128)
    initialC = kmeans.cluster_centers_.T
    for r in range(R):
        col_C = initialC[:, r]

        reconstructed_matrix = np.zeros((K, K), dtype=complex)
        k = 0
        for i in range(K):
            for j in range(K):
                if i == j:
                    # Diagonal: real values only
                    reconstructed_matrix[i, j] = col_C[k].real
                elif i < j:
                    # Upper triangle: real part stored directly
                    reconstructed_matrix[i, j] = col_C[k]
                else:  # i > j
                    # Lower triangle: imaginary part was stored (as col_C[k])
                    # So reconstruct as purely imaginary conjugate
                    reconstructed_matrix[i, j] = 1j * col_C[k]
                    # Fill upper triangle’s conjugate accordingly
                    reconstructed_matrix[j, i] = reconstructed_matrix[i, j].conjugate()
                k += 1

                vec_cov_matrix_psd, _ = project_to_psd(reconstructed_matrix)
                initC[:, r] = vec_cov_matrix_psd.reshape(K**2)

    initT = btd.factors_to_tensor(initA, initB, initC, theta)
    lamb = linalg.norm(T) / linalg.norm(initT)
    initA = initA * lamb ** (1 / 2)
    initB = initB * lamb ** (1 / 2)

    return initA, initB, initC


def _complex_to_real(X):
    real_tensor = np.zeros((X.shape[0], X.shape[1], X.shape[2]), dtype="float32")
    K = int(np.sqrt(X.shape[2]))

    for x in range(X.shape[0]):
        for y in range(X.shape[1]):
            cov_matrix = X[x, y, :].reshape(K, K)
            vectorized = np.zeros(K * K)

            # Fill the vector with real and imaginary parts
            k = 0
            for i in range(K):
                for j in range(K):
                    if i == j:
                        # Diagonal element: only the real part
                        vectorized[k] = cov_matrix[i, j].real
                    elif i < j:
                        # Upper triangular: real part
                        vectorized[k] = cov_matrix[i, j].real
                    else:  # i > j
                        # Lower triangular: imaginary part of conjugate
                        vectorized[k] = cov_matrix[j, i].imag
                    k += 1

            real_tensor[x, y, :] = vectorized

    return real_tensor
