import numpy as np
from tensorly import unfold
from tensorly.tenalg import khatri_rao
from scipy.linalg import solve
import pybbtd.btd as btd


def BTD_ALS(BTD_model, T, init="random", max_iter=1000, abs_tol=1e-8, rel_tol=1e-3):
    # Check that BTD_model is an instance of the BTD class
    if not isinstance(BTD_model, btd.BTD):
        raise TypeError("BTD_model must be an instance of the BTD class.")

    # Check that T is a numpy array
    if not isinstance(T, np.ndarray):
        raise TypeError("T must be a numpy array.")

    # Check that T's dimensions match BTD_model.dims
    if T.shape != BTD_model.dims:
        raise ValueError(
            f"T's dimensions ({T.shape}) do not match BTD_model.dims ({BTD_model.dims})."
        )

    # precompute unfoldings
    T0 = unfold(T, 0)
    T1 = unfold(T, 1)
    T2 = unfold(T, 2)

    # init factors
    Ak, Bk, Ck = init_BTD_factors(BTD_model, mode=init)
    theta = BTD_model.get_constraint_matrix()

    k = 0
    exit_criterion = False
    Tfit_0 = btd.factors_to_tensor(Ak, Bk, Ck, theta)
    fit_error = [np.linalg.norm(Tfit_0 - T) ** 2]

    while exit_criterion is False:
        # udpdate A
        M0 = khatri_rao([Bk, Ck @ theta])

        LHS = M0.T @ M0
        RHS = M0.T @ T0.T
        Ak1 = solve(LHS, RHS).T

        # udpdate B
        M1 = khatri_rao([Ak1, Ck @ theta])
        LHS = M1.T @ M1
        RHS = M1.T @ T1.T
        Bk1 = solve(LHS, RHS).T

        # update C
        M2 = khatri_rao([Ak1, Bk1]) @ theta.T
        LHS = M2.T @ M2
        RHS = M2.T @ T2.T
        Ck1 = solve(LHS, RHS).T

        # compute reconstruction error
        Tfit_k = btd.factors_to_tensor(Ak1, Bk1, Ck1, theta)
        fit_error.append(np.linalg.norm(Tfit_k - T) ** 2)

        # exit criterion
        k += 1

        if np.abs(fit_error[-1] - fit_error[-2]) / fit_error[-1] < rel_tol:
            print("Exiting early due to unsufficient decrease of cost")
            exit_criterion = True
        if fit_error[-1] / np.linalg.norm(T) < abs_tol:
            print("Reached absolute tolerance threshold. Exiting.")
            exit_criterion = True

        if k >= max_iter:
            exit_criterion = True
            print("Reached max number of iteration. Check convergence.")

        # pass variables
        Ak = Ak1.copy()
        Bk = Bk1.copy()
        Ck = Ck1.copy()
    return [Ak, Bk, Ck], np.array(fit_error)


def init_BTD_factors(BTD_model, mode="random"):
    # get BTD model params
    dims = BTD_model.dims
    R = BTD_model.rank
    L = BTD_model.L
    Lsum = np.array(L).sum()

    if mode == "random":
        # init factor matrices and constraint matrix
        A = np.random.rand(dims[0], Lsum)
        B = np.random.rand(dims[1], Lsum)
        C = np.random.rand(dims[2], R)
    else:
        raise ValueError("not implemented")

    return A, B, C
