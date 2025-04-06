import numpy as np
from tensorly import unfold, transpose
from tensorly.tenalg import khatri_rao
from scipy.linalg import solve
import pybbtd.btd as btd
from copy import deepcopy


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
    # algo is for LL1; convert to equivalent LL1 if not
    if BTD_model.block_mode == "1LL":
        newdims = (BTD_model.dims[1], BTD_model.dims[2], BTD_model.dims[0])
        transp_BTD_model = btd.BTD(dims=newdims, R=BTD_model.rank, L=BTD_model.L)
        transp_T = transpose(T, (1, 2, 0))
    elif BTD_model.block_mode == "L1L":
        newdims = (BTD_model.dims[0], BTD_model.dims[2], BTD_model.dims[1])
        transp_BTD_model = btd.BTD(dims=newdims, R=BTD_model.rank, L=BTD_model.L)
        transp_T = transpose(T, (0, 2, 1))
    else:
        transp_BTD_model = deepcopy(BTD_model)
        transp_T = deepcopy(T)

    # precompute unfoldings
    T0 = unfold(transp_T, 0)
    T1 = unfold(transp_T, 1)
    T2 = unfold(transp_T, 2)

    # init factors
    Ak, Bk, Ck = init_BTD_factors(transp_BTD_model, strat=init, T=transp_T)
    theta = transp_BTD_model.get_constraint_matrix()

    k = 0
    exit_criterion = False
    Tfit_0 = btd.factors_to_tensor(Ak, Bk, Ck, theta)
    fit_error = [np.linalg.norm(Tfit_0 - transp_T) ** 2]

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
        fit_error.append(np.linalg.norm(Tfit_k - transp_T) ** 2)

        # exit criterion
        k += 1

        if np.abs(fit_error[-1] - fit_error[-2]) / fit_error[-1] < rel_tol:
            print("Exiting early due to unsufficient decrease of cost")
            exit_criterion = True
        if fit_error[-1] / np.linalg.norm(transp_T) < abs_tol:
            print("Reached absolute tolerance threshold. Exiting.")
            exit_criterion = True

        if k >= max_iter:
            exit_criterion = True
            print("Reached max number of iteration. Check convergence.")

        # pass variables
        Ak = Ak1.copy()
        Bk = Bk1.copy()
        Ck = Ck1.copy()

    # transpose variables back to actual model
    if BTD_model.block_mode == "LL1":
        factors = [Ak, Bk, Ck]
    elif BTD_model.block_mode == "1LL":
        factors = [Ck, Ak, Bk]
    elif BTD_model.block_mode == "L1L":
        factors = [Ak, Ck, Bk]

    return factors, np.array(fit_error)


def init_BTD_factors(BTD_model, strat="random", T=None):
    """LL1-based init

    Args:
        BTD_model (_type_): _description_
        strat (str, optional): _description_. Defaults to "random".
        T (_type_, optional): _description_. Defaults to None.

    Raises:
        ValueError: _description_
        ValueError: _description_

    Returns:
        _type_: _description_
    """

    # get BTD model params
    dims = BTD_model.dims
    R = BTD_model.rank
    L = BTD_model.L
    Lsum = np.array(L).sum()

    if strat == "random":
        # init factor matrices and constraint matrix
        A = np.random.randn(dims[0], Lsum)
        B = np.random.randn(dims[1], Lsum)
        C = np.random.randn(dims[2], R)
    elif strat == "svd":
        if T is None:
            raise ValueError("SVD init requires input data T")
        u0, s0, __ = np.linalg.svd(unfold(T, 0))
        u1, s1, __ = np.linalg.svd(unfold(T, 1))
        u2, s2, __ = np.linalg.svd(unfold(T, 2))

        A = u0[:, :Lsum] @ np.diag(s0[:Lsum] ** 0.5)
        B = u1[:, :Lsum] @ np.diag(s1[:Lsum] ** 0.5)
        C = u2[:, :R] @ np.diag(s2[:R] ** 0.5)

    else:
        raise ValueError("not implemented")

    # if normalize is True:
    #     if T is None:
    #         raise ValueError("normalization requires input data T")
    #     Tnorm = np.linalg.norm(T)
    #     Tnorm_rec = np.linalg.norm(btd.factors_to_tensor(A, B, C, BTD_model.get_constraint_matrix()))
    #     ratio = Tnorm/Tnorm_rec
    #     A = A/(ratio)**(1/3)
    #     B = B/(ratio)**(1/3)
    #     C = C/(ratio)**(1/3)
    #     print(np.linalg.norm(btd.factors_to_tensor(A, B, C, BTD_model.get_constraint_matrix()))/Tnorm)

    return A, B, C
