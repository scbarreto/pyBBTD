import numpy as np
from tensorly import unfold
from scipy import linalg
from scipy.linalg import solve
import pybbtd.stokes as stokes
from tensorly.tenalg import khatri_rao
import pybbtd.btd as btd
from sklearn.decomposition import NMF
from sklearn.cluster import KMeans


def ADMM_A(Y1, Bk, Ck, Ainit, Atinit, rho, L, nitermax=100, tol=1e-14):
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
        A_T = solve(M1M1T.T, RHS.T, assume_a="sym")
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

        # update all variables
        Al = Al1.copy()
        Atl = Atl1.copy()
        Ul = Ul1.copy()

        n += +1

    return Al, Atl, Ul


def ADMM_B(Y2, Ak, Ck, Binit, Btinit, rho, L, nitermax=100, tol=1e-14):
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
        B_T = solve(M2M2T.T, RHS.T, assume_a="sym")
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

        # update all variables
        Bl = Bl1.copy()
        Btl = Btl1.copy()
        Ul = Ul1.copy()

        n += +1

    return Bl, Btl, Ul


def ADMM_C(Y3, Ak, Bk, Cinit, Ctinit, theta, rho, L, R, nitermax=100, tol=1e-14):
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
        C_T = solve(M3M3T.T, RHS.T, assume_a="sym")
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

        # update all variables
        Cl = Cl1.copy()
        Ctl = Ctl1.copy()
        Ul = Ul1.copy()

        n += +1

    return Cl, Ctl, Ul


def stokes_kmeans(R, T):
    unfolding = unfold(T, 2).T

    clustered = KMeans(n_clusters=R, random_state=0, n_init="auto").fit(unfolding)

    initialC = clustered.cluster_centers_.T

    for i in range(R):
        initialC[:, i] = stokes.stokesProjection(initialC[:, i])

    for i in range(R):
        if initialC[0, i] > 0:
            initialC[:, i] = initialC[:, i] / initialC[0, i]

    Rterms = np.zeros((R, T.shape[0] * T.shape[1]))

    features = np.ones((R, T.shape[0], T.shape[1]))

    for i in range(len(clustered.labels_)):
        # Rterms[clustered.labels_[i], i] = np.random.rand(
        #     np.size(clustered.labels_[i]))
        Rterms[clustered.labels_[i], i] = np.random.rand()

    for i in range(R):
        features[i] = Rterms[i].reshape(T.shape[0], T.shape[1])

    return features, initialC


def stokes_NMF(Lr, R, maps):
    model = NMF(n_components=Lr, init="random", max_iter=1000)

    W = np.zeros((R, maps.shape[1], Lr))
    H = np.zeros((R, Lr, maps.shape[2]))

    product = np.zeros((R, maps.shape[1], maps.shape[2]))

    for i in range(R):
        W[i] = model.fit_transform(maps[i])

        H[i] = model.components_
        product[i] = W[i] @ H[i]

    initA = W[0]
    initB = H[0].T

    if R > 1:
        for i in range(1, R):
            initA = np.concatenate((initA, W[i]), axis=1)
            initB = np.concatenate((initB, H[i].T), axis=1)

    return product, initA, initB


def kmeans_init(Lr, R, T, theta):
    L = Lr[0]

    featureMaps, kmeansC = stokes_kmeans(R, T)

    outputNMF, kmeansA, kmeansB = stokes_NMF(L, R, featureMaps)

    Tkmeans = btd.factors_to_tensor(kmeansA, kmeansB, kmeansC, theta)
    lamb = linalg.norm(T) / linalg.norm(Tkmeans)

    kmeansA = kmeansA * lamb ** (1 / 2)
    kmeansB = kmeansB * lamb ** (1 / 2)

    return kmeansA, kmeansB, kmeansC


def init_Stokes_factors(Stokes_model, init="random", T=None):
    # Add check to see if T has coherent dimensions
    import pybbtd.stokes as stokes

    theta = Stokes_model.get_constraint_matrix()
    # get BTD model params
    dims = Stokes_model.dims
    R = Stokes_model.rank
    L = Stokes_model.L

    if init == "random":
        A, B, C = stokes.generate_stokes_factors(dims, R, L)

    elif init == "kmeans":
        # Check that T is provided
        if T is None:
            raise ValueError("T must be provided for kmeans initialization.")
        # Check that T's dimensions match Stokes_model.dims
        if T.shape != Stokes_model.dims:
            raise ValueError(
                f"T's dimensions ({T.shape}) do not match Stokes_model.dims ({Stokes_model.dims})."
            )
        A, B, C = kmeans_init(L, R, T, theta)
    else:
        raise ValueError("Unknown initialization strategy.")

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
    import pybbtd.stokes as stokes

    theta = Stokes_model.get_constraint_matrix()
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
        Ak, Bk, Ck = init_Stokes_factors(Stokes_model, "kmeans", T)
        Atk, Btk, Ctk = init_Stokes_factors(Stokes_model, init="random")
    else:
        raise ValueError("not implemented")

    T1 = unfold(T, 0)
    T2 = unfold(T, 1)
    T3 = unfold(T, 2)

    L = Stokes_model.L
    R = Stokes_model.rank

    Tfit_0 = btd.factors_to_tensor(Ak, Bk, Ck, theta)
    fit_error = [np.linalg.norm(Tfit_0 - T) ** 2]

    k = 0
    exit_criterion = False
    while exit_criterion is False:
        # update A
        Ak1, Atk1, _ = ADMM_A(
            T1, Bk, (Ck @ theta), Ak, Atk, rho, L, nitermax=max_admm, tol=admm_tol
        )

        # update B
        Bk1, Btk1, _ = ADMM_B(
            T2, Ak1, (Ck @ theta), Bk, Btk, rho, L, nitermax=max_admm, tol=admm_tol
        )

        # update C
        Ck1, Ctk1, _ = ADMM_C(
            T3, Ak1, Bk1, Ck, Ctk, theta, rho, L, R, nitermax=max_admm, tol=admm_tol
        )

        # update variables
        Ak = Ak1.copy()
        Bk = Bk1.copy()
        Ck = Ck1.copy()

        Atk = Atk1.copy()
        Btk = Btk1.copy()
        Ctk = Ctk1.copy()
        estimated_factors = [Ak1, Bk1, Ck1]
        k += 1

        # compute reconstruction error

        Tfit_k = btd.factors_to_tensor(Ak, Bk, Ck, theta)
        fit_error.append(np.linalg.norm(Tfit_k - T) ** 2)

        if np.abs(fit_error[-1] - fit_error[-2]) / fit_error[-1] < rel_tol:
            print("Exiting early due to unsufficient decrease of cost")
            exit_criterion = True
        if fit_error[-1] / np.linalg.norm(T) < abs_tol:
            print("Reached absolute tolerance threshold. Exiting.")
            exit_criterion = True

        if k >= max_iter:
            exit_criterion = True
            print("Reached max number of iteration. Check convergence.")

    return estimated_factors, fit_error
