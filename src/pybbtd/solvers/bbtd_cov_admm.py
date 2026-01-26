import numpy as np
import scipy
from tensorly import unfold
from tensorly.tenalg import khatri_rao
from tensorly.cp_tensor import cp_to_tensor
import pybbtd.bbtd as bbtd


def BBTD_COV_ADMM(
    BBTD_model,
    T,
    init="random",
    max_iter=1000,
    gamma=1.0,
    rho=1.0,
    inner_admm=50,
    tol_admm=1e-8,
    rel_tol=1e-6,
    abs_tol=1e-12,
):
    """
    ADMM solver for the constrained BBTD decomposition with:
    - Non-negativity on spatial maps (A, B)
    - Conjugate symmetry on C and D (D = C*)

    Parameters:
        BBTD_model: BBTD
            An instance of the BBTD class containing model parameters.
        T: np.ndarray
            The input 4D tensor to be decomposed, shape (I, J, K, K).
        init: str
            Initialization strategy. Options: "random".
        max_iter: int
            Maximum number of outer iterations.
        gamma: float
            Penalty parameter for the A, B spatial map coupling.
        rho: float
            ADMM penalty parameter for the C, D conjugate constraint.
        inner_admm: int
            Maximum number of inner ADMM iterations for C update.
        tol_admm: float
            Tolerance for inner ADMM convergence.
        rel_tol: float
            Relative tolerance for outer convergence.
        abs_tol: float
            Absolute tolerance for outer convergence.

    Returns:
        factors: list
            List of factor matrices [A, B, C, D].
        fit_error: np.ndarray
            Array of fitting errors at each iteration.
    """
    # Validate inputs
    if not isinstance(BBTD_model, bbtd.BBTD):
        raise TypeError("BBTD_model must be an instance of the BBTD class.")
    if not isinstance(T, np.ndarray):
        raise TypeError("T must be a numpy array.")
    if T.shape != BBTD_model.dims:
        raise ValueError(
            f"T's dimensions ({T.shape}) do not match BBTD_model.dims ({BBTD_model.dims})."
        )

    J, M, K, K = BBTD_model.dims
    R = BBTD_model.rank
    L1 = BBTD_model.L1
    L2 = BBTD_model.L2

    # Precompute constraint matrices
    phi, psi = BBTD_model.get_constraint_matrices()
    _, psibtd = bbtd._constraint_matrices(L2, 1, R)

    # Precompute unfoldings for A, B (4D)
    T1 = unfold(T, 0).T
    T2 = unfold(T, 1).T

    # Reshape to 3D for C, D updates
    T3D = T.reshape(J * M, K, K)
    T3 = unfold(T3D, 1)
    T4 = unfold(T3D, 2)

    # Initialize factors
    Aest, Best, Cest = init_BBTD_cov_factors(BBTD_model, strat=init)
    Dest = Cest.conj()

    # Compute initial spatial maps
    Sout = np.zeros((R, J, M))
    for r in range(R):
        Sout[r] = Aest[:, r * L1 : (r + 1) * L1] @ Best[:, r * L1 : (r + 1) * L1].T
    vecSout = Sout.reshape(R, J * M).T

    # Scale initialization to match data norm
    factors_init = [vecSout @ psibtd, Cest, Dest]
    weights = np.ones(L2 * R)
    tensor_init = cp_to_tensor((weights, factors_init))
    scale_factor = np.linalg.norm(T3D) / np.linalg.norm(tensor_init)
    Sout = Sout * scale_factor
    vecSout = Sout.reshape(R, J * M).T

    # Initialize metrics
    cost = np.zeros(max_iter)
    outer_check_interval = int(max_iter / 5) if max_iter >= 5 else 1

    for it in range(max_iter):
        if it != 0 and it % outer_check_interval == 0:
            print("Progress:", (it / max_iter * 100), "%")

        # Update A
        M1 = phi @ khatri_rao((Best @ phi, Cest @ psi, Cest.conj() @ psi)).T
        Aest = _solve_A(Best, Sout, T1, M1, gamma, L1)
        Aest = np.nan_to_num(Aest, nan=0, posinf=0, neginf=0)

        # Update B
        M2 = phi @ khatri_rao((Aest @ phi, Cest @ psi, Cest.conj() @ psi)).T
        Best = _solve_B(Aest, Sout, T2, M2, gamma, L1)
        Best = np.nan_to_num(Best, nan=0, posinf=0, neginf=0)

        # Project spatial maps to non-negative
        Sout = np.zeros((R, J, M))
        for r in range(R):
            Sout[r] = np.maximum(
                0, Aest[:, r * L1 : (r + 1) * L1] @ Best[:, r * L1 : (r + 1) * L1].T
            )
        vecSout = Sout.reshape(R, J * M).T

        # Update C and D via ADMM (enforcing D = C*)
        Cest, Dest, _, _ = _solve_C(
            Cest, Dest, T3, T4, vecSout, psibtd, rho, inner_admm, tol_admm, L2, R
        )

        # Compute reconstruction error
        factors_rec = [vecSout @ psibtd, Cest, Dest]
        weights = np.ones(L2 * R)
        tensor_rec = cp_to_tensor((weights, factors_rec))
        cost[it] = np.linalg.norm(T3D - tensor_rec) / np.linalg.norm(T3D)

        eps_zero = 1e-13
        if it > 0 and np.abs(cost[it] - cost[it - 1]) / (cost[it] + eps_zero) < rel_tol:
            print(f"Exiting early at iteration {it} due to relative tolerance.")
            cost = cost[: it + 1]
            break
        if (
            it > 0
            and np.abs(cost[it] - cost[it - 1]) / (cost[it - 1] + eps_zero) < rel_tol
        ):
            print(f"Exiting early at iteration {it} due to relative tolerance.")
            cost = cost[: it + 1]
            break
        if cost[it] < abs_tol:
            print(f"Reached absolute tolerance at iteration {it}.")
            cost = cost[: it + 1]
            break
    else:
        print("Reached max number of iterations. Check convergence.")

    factors = [Aest, Best, Cest, Dest]
    return factors, cost


def _solve_A(estB, estS, T1, M1, gamma, L1):
    """
    Solve for A with non-negativity coupling penalty.

    The update minimizes the LS criterion for the first unfolding
    plus a penalty term gamma * ||S_r - A_r @ B_r^T||^2.
    """
    R = estS.shape[0]

    M1_M1H = (M1 @ M1.T.conj()).real
    RHS = (T1.T.conj() @ M1.T).real

    Brs = []
    SBs = []
    for r in range(R):
        Brs.append(estB[:, r * L1 : (r + 1) * L1].T @ estB[:, r * L1 : (r + 1) * L1])
        SBs.append(estS[r, :, :] @ estB[:, r * L1 : (r + 1) * L1])
    SBfinal = np.hstack(SBs)
    BTB = scipy.linalg.block_diag(*Brs)

    term1 = M1_M1H + (gamma * BTB)
    term2 = RHS + (gamma * SBfinal)

    term1 = np.nan_to_num(term1, nan=0, posinf=0, neginf=0)
    term2 = np.nan_to_num(term2, nan=0, posinf=0, neginf=0)

    A_min = np.linalg.solve(term1.T, term2.T)
    return A_min.T


def _solve_B(estA, estS, T2, M2, gamma, L1):
    """
    Solve for B with non-negativity coupling penalty.

    The update minimizes the LS criterion for the second unfolding
    plus a penalty term gamma * ||S_r - A_r @ B_r^T||^2.
    """
    R = estS.shape[0]

    M2M2H = (M2 @ M2.T.conj()).real
    RHS = (T2.T.conj() @ M2.T).real

    ATA_blocks = []
    SAfinal = np.zeros((estS.shape[2], R * L1))
    for r in range(R):
        A_r = estA[:, r * L1 : (r + 1) * L1]
        ATA_blocks.append(A_r.T @ A_r)
        SAfinal[:, r * L1 : (r + 1) * L1] = estS[r].T @ A_r
    ATA = scipy.linalg.block_diag(*ATA_blocks)

    term1 = M2M2H + gamma * ATA
    term2 = RHS + gamma * SAfinal

    term1 = np.nan_to_num(term1, nan=0, posinf=0, neginf=0)
    term2 = np.nan_to_num(term2, nan=0, posinf=0, neginf=0)

    B_min = np.linalg.solve(term1.T, term2.T)
    return B_min.T


def _solve_C(estC, estD, T3, T4, vecS0, psibtd, rho, inner_admm, tol_admm, L2, R):
    """
    ADMM update for C and D enforcing the conjugate constraint D = C*.

    Alternately updates C and D = C* with a dual variable to enforce consistency.
    After convergence, normalizes each block of C via SVD.
    """
    talincr = taldecr = 2
    mi = 10

    epsPri = np.size(estC) ** (1 / 2) * tol_admm
    epsDual = np.size(estC) ** (1 / 2) * tol_admm

    n = 0
    primalResidue = np.zeros(inner_admm)
    dualResidue = np.zeros(inner_admm)
    Ul = np.zeros_like(estC)
    exitCriterion = True

    while (n < inner_admm) & exitCriterion:
        # Update C
        M3 = khatri_rao((vecS0 @ psibtd, estD)).T
        term1 = M3 @ M3.T.conj() + rho * np.eye(estC.shape[1])
        term2 = T3 @ M3.T.conj() + rho * (estD.conj() - Ul)

        estC1 = np.linalg.solve(term1.T, term2.T).T

        # Update D
        M4 = khatri_rao((vecS0 @ psibtd, estC1)).T
        term1 = M4 @ M4.T.conj() + rho * np.eye(estD.shape[1])
        term2 = T4 @ M4.T.conj() + rho * (estC1.conj() + Ul.conj())

        estD1 = np.linalg.solve(term1.T, term2.T).T

        # Update dual variable
        Ul1 = Ul + estC1 - estD1.conj()

        # Compute primal and dual residuals
        norm_C1 = np.linalg.norm(estC1)
        primalResidue[n] = np.linalg.norm(estC1 - estD1.conj()) / norm_C1
        dualResidue[n] = rho * np.linalg.norm(estD1 - estD) / np.linalg.norm(estD1)

        estC = estC1
        estD = estD1
        Ul = Ul1

        if (primalResidue[n] < epsPri) and (dualResidue[n] < epsDual):
            exitCriterion = False

        # Adjust rho based on residuals
        if primalResidue[n] > (mi * dualResidue[n]):
            rho *= talincr
            Ul /= talincr
        elif dualResidue[n] > (mi * primalResidue[n]):
            rho /= taldecr
            Ul *= taldecr

        # Check for early exit if primal and dual both stagnate
        if n > 0:
            incrPri = np.abs(primalResidue[n] - primalResidue[n - 1]) / primalResidue[n]
            incrDual = np.abs(dualResidue[n] - dualResidue[n - 1]) / dualResidue[n]
            if (incrPri < tol_admm) and (incrDual < tol_admm):
                exitCriterion = False

        n += 1

    # Average C and D* for final estimate
    estC = (estC1 + estD1.conj()) / 2

    # Normalize each block of C via SVD
    newC = np.zeros_like(estC)
    for r in range(R):
        U, s, _ = np.linalg.svd(estC[:, r * L2 : (r + 1) * L2])
        newC[:, r * L2 : (r + 1) * L2] = U[:, :L2] @ np.diag(s[:L2])
    estC = newC
    estD = estC.conj()

    return estC, estD, primalResidue, dualResidue


def init_BBTD_cov_factors(BBTD_model, strat="random"):
    """
    Initialize factor matrices for the constrained BBTD decomposition.

    Generates non-negative A, B and complex C (with D = C*).

    Parameters:
        BBTD_model: BBTD
            An instance of the BBTD class.
        strat: str
            Initialization strategy. Options: "random".

    Returns:
        A, B, C: tuple of np.ndarray
            Initialized factor matrices. D = C.conj() is implicit.
    """
    dims = BBTD_model.dims
    R = BBTD_model.rank
    L1 = BBTD_model.L1
    L2 = BBTD_model.L2

    if strat == "random":
        A = np.random.rand(dims[0], L1 * R)
        B = np.random.rand(dims[1], L1 * R)
        C = np.random.randn(dims[2], L2 * R) + 1j * np.random.randn(dims[2], L2 * R)
    else:
        raise ValueError(f"Initialization strategy '{strat}' not implemented.")

    return A, B, C
