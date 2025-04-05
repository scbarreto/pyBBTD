from math import floor
import numpy as np


def check_uniqueness_LL1(N1, N2, N3, R, L):
    if not (isinstance(L, np.integer) or isinstance(L, int)):
        raise NotImplementedError("L should be an integer")

    def cond_BTD_41(N1, N2, N3, L, R):
        return (min(N1, N2) >= L * R) and (R <= N3)

    def cond_BTD_42(N1, N2, N3, L, R):
        return (N3 >= R) and (min(floor(N1 / L), R) + min(floor(N2 / L), R) >= R + 2)

    def cond_BTD_43(N1, N2, N3, L, R):
        return (N1 >= L * R) and (min(floor(N2 / L), R) + min(N3, R) >= R + 2)

    def cond_BTD_44(N1, N2, N3, L, R):
        return (N2 >= L * R) and (min(floor(N1 / L), R) + min(N3, R) >= R + 2)

    def cond_BTD_45(N1, N2, N3, L, R):
        return (floor(N1 * N2 / L**2) >= R) and (
            min(floor(N1 / L), R) + min(floor(N2 / L), R) + min(N3, R) >= 2 * R + 2
        )

    # Check if any condition is true
    if any(
        cond(N1, N2, N3, L, R)
        for cond in [cond_BTD_41, cond_BTD_42, cond_BTD_43, cond_BTD_44, cond_BTD_45]
    ):
        return True
    # otherwise return False
    return False
