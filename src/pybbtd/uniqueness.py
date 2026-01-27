from math import floor

import numpy as np


def check_uniqueness_BBTD(N1, N2, N3, N4, R, L1, L2):
    """
    Check if a fourth-order BBTD model with rank-(L1, L1, L2, L2) terms is unique
    for the given dimensions by examining two equivalent third-order models.

    The fourth-order tensor is unfolded into two third-order BTD models:

    - **Vectorize first block** (modes 1 and 2): third-order BTD with
      dimensions ``(N3, N4, N1*N2)`` and block rank ``L2``.
    - **Vectorize second block** (modes 3 and 4): third-order BTD with
      dimensions ``(N1, N2, N3*N4)`` and block rank ``L1``.

    Parameters:
        N1: int
            First dimension of the 4D tensor.
        N2: int
            Second dimension of the 4D tensor.
        N3: int
            Third dimension of the 4D tensor.
        N4: int
            Fourth dimension of the 4D tensor.
        R: int
            Rank of the decomposition (number of components).
        L1: int
            Rank of the first block (modes 1 and 2).
        L2: int
            Rank of the second block (modes 3 and 4).

    Returns:
        bool:
            ``True`` if a sufficient condition for uniqueness is satisfied,
            ``False`` otherwise.
    """
    if not isinstance(L1, (int, np.integer)):
        raise NotImplementedError("L1 should be an integer")
    if not isinstance(L2, (int, np.integer)):
        raise NotImplementedError("L2 should be an integer")

    # Model 1: vectorize first block (modes 1,2) -> BTD(N3, N4, N1*N2, L2, R)
    model1_unique = check_uniqueness_LL1(N3, N4, N1 * N2, R, L2)

    # Model 2: vectorize second block (modes 3,4) -> BTD(N1, N2, N3*N4, L1, R)
    model2_unique = check_uniqueness_LL1(N1, N2, N3 * N4, R, L1)

    return model1_unique or model2_unique


def check_uniqueness_LL1(N1, N2, N3, R, L):
    """
    Check if a third-order BTD model with rank-(L, L, 1) terms is unique
    for the given dimensions.

    Five sufficient conditions from the BTD uniqueness theory are evaluated. If any one of them is satisfied, uniqueness of the
    decomposition is guaranteed.

    Parameters:
        N1: int
            First dimension of the 3D tensor.
        N2: int
            Second dimension of the 3D tensor.
        N3: int
            Third dimension of the 3D tensor.
        R: int
            Rank of the decomposition (number of components).
        L: int
            Rank of the spatial maps (block rank).

    Returns:
        bool:
            ``True`` if a sufficient condition for uniqueness is satisfied,
            ``False`` otherwise.
    """
    if not isinstance(L, (int, np.integer)):
        raise NotImplementedError("L should be an integer")

    conditions = [
        (min(N1, N2) >= L * R) and (R <= N3),
        (N3 >= R) and (min(floor(N1 / L), R) + min(floor(N2 / L), R) >= R + 2),
        (N1 >= L * R) and (min(floor(N2 / L), R) + min(N3, R) >= R + 2),
        (N2 >= L * R) and (min(floor(N1 / L), R) + min(N3, R) >= R + 2),
        (floor(N1 * N2 / L**2) >= R)
        and (min(floor(N1 / L), R) + min(floor(N2 / L), R) + min(N3, R) >= 2 * R + 2),
    ]

    return any(conditions)
