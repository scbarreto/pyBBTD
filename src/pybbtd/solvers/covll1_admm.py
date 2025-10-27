import numpy as np
from tensorly import unfold
from scipy import linalg
from scipy.linalg import solve
from tensorly.tenalg import khatri_rao
import warnings
import pybbtd.btd as btd


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
        A_T = solve(
            (M1M1T + rho * np.eye(Ainit.shape[1])
             ).T, (RHS + rho * (Atl - Ul)).T
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
            (M2M2T + rho * np.eye(Binit.shape[1])
             ).T, (RHS + rho * (Btl - Ul)).T
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
            (M3M3T + rho * np.eye(Cinit.shape[1])
             ).T, (RHS + rho * (Ctl - Ul)).T
        )
        Cl1 = C_T.T

        # update \tilde{C}_{l+1}
        Ctl1 = np.zeros_like(Ctl)
        Cl1 += Ul
        for r in range(R):
            cov_matrix = Cl1[:, r].reshape(
                int(np.sqrt(Cinit.shape[0])), int(np.sqrt(Cinit.shape[0]))
            )
            U, s, Vh = linalg.svd(cov_matrix)

            # Project onto PSD cone: set negative eigenvalues to zero
            s_psd = np.clip(s, a_min=0, a_max=None)

            # Reconstruct PSD matrix
            cov_matrix_psd = (U * s_psd) @ Vh

            # Vectorize
            vec_cov = cov_matrix_psd.reshape(-1)  # K^2 vector
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


def CovLL1_ADMM(
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
    if init == "random":
        Ak, Bk, Ck = init_covll1_factors(CovLL1_model, init="random")
        Atk, Btk, Ctk = init_covll1_factors(CovLL1_model, init="random")
    # elif init == "kmeans":
    #     Atk, Btk, Ctk = init_covll1_factors(CovLL1_model, "kmeans", T)
    #     Ak, Bk, Ck = init_covll1_factors(CovLL1_model, "random")
    else:
        raise ValueError("not implemented")

    T1 = unfold(T, 0)
    T2 = unfold(T, 1)
    T3 = unfold(T, 2)

    L1 = CovLL1_model.L1
    R = CovLL1_model.rank

    theta = CovLL1_model.get_constraint_matrix()

    Tfit_0 = btd.factors_to_tensor(Atk, Btk, Ctk, theta)
    fit_error = [np.linalg.norm(Tfit_0 - T) ** 2]

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

        fit_error.append(np.linalg.norm(Tfit_k - T) **
                         2 / np.linalg.norm(T) ** 2)

        if np.abs(fit_error[-1] - fit_error[-2]) < rel_tol:
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


def init_covll1_factors(CovLL1_model, init="random", T=None):
    # Add check to see if T has coherent dimensions
    import pybbtd.covll1 as covll1

    # get BTD model params
    dims = CovLL1_model.dims
    R = CovLL1_model.rank
    L1 = CovLL1_model.L1
    L2 = CovLL1_model.L2

    if init == "random":
        A, B, C = covll1.generate_covll1_factors(dims, R, L1, L2)

    # elif init == "kmeans":
    #     A, B, C = kmeans_init(L, R, T, theta)

    return A, B, C
