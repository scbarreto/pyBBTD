import numpy as np


def _constraint_matrix(N, R, L):
    """Compute the constraint matrix repeating columsn

    Args:
        N (_type_): dim of the mode to be repeated
        R (_type_): rank of the decomposition
        L (_type_): sequence of L values for each block
    """

    Lsum = np.array(L).sum()
    theta = np.zeros(N, Lsum)

    return theta
