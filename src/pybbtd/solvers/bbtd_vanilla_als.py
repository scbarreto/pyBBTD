import numpy as np
from tensorly import unfold
from tensorly.tenalg import khatri_rao
from scipy.linalg import solve
import pybbtd.bbtd as bbtd


def BBTD_ALS(BBTD_model, T, init="random", max_iter=1000, abs_tol=1e-8, rel_tol=1e-3):
    """
    Vanilla ALS solver for the unconstrained BBTD decomposition.

    Parameters:
        BBTD_model: BBTD
            An instance of the BBTD class containing model parameters.
        T: np.ndarray
            The input 4D tensor to be decomposed.
        init: str
            Initialization strategy. Options: "random", "svd".
        max_iter: int
            Maximum number of iterations.
        abs_tol: float
            Absolute tolerance for convergence.
        rel_tol: float
            Relative tolerance for convergence.

    Returns:
        factors: list
            List of factor matrices [A, B, C, D].
        fit_error: np.ndarray
            Array of fitting errors at each iteration.
    """
    # Check that BBTD_model is an instance of the BBTD class
    if not isinstance(BBTD_model, bbtd.BBTD):
        raise TypeError("BBTD_model must be an instance of the BBTD class.")

    # Check that T is a numpy array
    if not isinstance(T, np.ndarray):
        raise TypeError("T must be a numpy array.")

    # Check that T's dimensions match BBTD_model.dims
    if T.shape != BBTD_model.dims:
        raise ValueError(
            f"T's dimensions ({T.shape}) do not match BBTD_model.dims ({BBTD_model.dims})."
        )

    # Precompute unfoldings (transposed to match the baseline convention)
    T0 = unfold(T, 0).T
    T1 = unfold(T, 1).T
    T2 = unfold(T, 2).T
    T3 = unfold(T, 3).T

    # Initialize factors
    Ak, Bk, Ck, Dk = init_BBTD_factors(BBTD_model, strat=init, T=T)

    # Get constraint matrices
    phi, psi = BBTD_model.get_constraint_matrices()

    k = 0
    exit_criterion = False
    Tfit_0 = bbtd.factors_to_tensor(Ak, Bk, Ck, Dk, phi, psi)
    fit_error = [np.linalg.norm(Tfit_0 - T) / np.linalg.norm(T)]

    while exit_criterion is False:
        # Update A
        M0 = phi @ khatri_rao((Bk @ phi, Ck @ psi, Dk @ psi)).T
        M0_M0H = M0 @ M0.T.conj()
        RHS = T0.T @ M0.T.conj()

        M0_M0H = np.nan_to_num(M0_M0H, nan=0, posinf=0, neginf=0)
        RHS = np.nan_to_num(RHS, nan=0, posinf=0, neginf=0)
        Ak1 = solve(M0_M0H.T, RHS.T, assume_a="pos").T

        # Update B
        M1 = phi @ khatri_rao((Ak1 @ phi, Ck @ psi, Dk @ psi)).T
        M1_M1H = M1 @ M1.T.conj()
        RHS = T1.T @ M1.T.conj()

        M1_M1H = np.nan_to_num(M1_M1H, nan=0, posinf=0, neginf=0)
        RHS = np.nan_to_num(RHS, nan=0, posinf=0, neginf=0)
        Bk1 = solve(M1_M1H.T, RHS.T, assume_a="pos").T

        # Update C
        M2 = psi @ khatri_rao((Ak1 @ phi, Bk1 @ phi, Dk @ psi)).T
        M2_M2H = M2 @ M2.T.conj()
        RHS = T2.T @ M2.T.conj()

        M2_M2H = np.nan_to_num(M2_M2H, nan=0, posinf=0, neginf=0)
        RHS = np.nan_to_num(RHS, nan=0, posinf=0, neginf=0)
        Ck1 = solve(M2_M2H.T, RHS.T, assume_a="pos").T

        # Update D
        M3 = psi @ khatri_rao((Ak1 @ phi, Bk1 @ phi, Ck1 @ psi)).T
        M3_M3H = M3 @ M3.T.conj()
        RHS = T3.T @ M3.T.conj()

        M3_M3H = np.nan_to_num(M3_M3H, nan=0, posinf=0, neginf=0)
        RHS = np.nan_to_num(RHS, nan=0, posinf=0, neginf=0)
        Dk1 = solve(M3_M3H.T, RHS.T, assume_a="pos").T

        # Compute reconstruction error
        Tfit_k = bbtd.factors_to_tensor(Ak1, Bk1, Ck1, Dk1, phi, psi)
        fit_error.append(np.linalg.norm(Tfit_k - T) / np.linalg.norm(T))

        # Exit criterion
        k += 1
        eps_zero = 1e-13

        if np.abs(fit_error[-1] - fit_error[-2]) / (fit_error[-2] + eps_zero) < rel_tol:
            print(
                f"Exiting early at iteration {k} due to insufficient decrease of cost."
            )
            exit_criterion = True
        if fit_error[-1] < abs_tol:
            print(f"Reached absolute tolerance threshold at iteration {k}. Exiting.")
            exit_criterion = True
        if k >= max_iter:
            exit_criterion = True
            print("Reached max number of iterations. Check convergence.")

        # Pass variables
        Ak = Ak1.copy()
        Bk = Bk1.copy()
        Ck = Ck1.copy()
        Dk = Dk1.copy()

    factors = [Ak, Bk, Ck, Dk]
    return factors, np.array(fit_error)


def init_BBTD_factors(BBTD_model, strat="random", T=None):
    """
    Initialize factor matrices for BBTD decomposition.

    Parameters:
        BBTD_model: BBTD
            An instance of the BBTD class.
        strat: str
            Initialization strategy. Options: "random", "svd".
        T: np.ndarray
            Input tensor (required for SVD initialization).

    Returns:
        A, B, C, D: tuple of np.ndarray
            Initialized factor matrices.
    """
    dims = BBTD_model.dims
    R = BBTD_model.rank
    L1 = BBTD_model.L1
    L2 = BBTD_model.L2

    if strat == "random":
        A = np.random.randn(dims[0], L1 * R)
        B = np.random.randn(dims[1], L1 * R)
        C = np.random.randn(dims[2], L2 * R)
        D = np.random.randn(dims[3], L2 * R)
    elif strat == "svd":
        if T is None:
            raise ValueError("SVD init requires input data T.")
        u0, s0, __ = np.linalg.svd(unfold(T, 0), full_matrices=False)
        u1, s1, __ = np.linalg.svd(unfold(T, 1), full_matrices=False)
        u2, s2, __ = np.linalg.svd(unfold(T, 2), full_matrices=False)
        u3, s3, __ = np.linalg.svd(unfold(T, 3), full_matrices=False)

        A = u0[:, : L1 * R] @ np.diag(s0[: L1 * R] ** 0.5)
        B = u1[:, : L1 * R] @ np.diag(s1[: L1 * R] ** 0.5)
        C = u2[:, : L2 * R] @ np.diag(s2[: L2 * R] ** 0.5)
        D = u3[:, : L2 * R] @ np.diag(s3[: L2 * R] ** 0.5)
    else:
        raise ValueError(f"Initialization strategy '{strat}' not implemented.")

    return A, B, C, D
