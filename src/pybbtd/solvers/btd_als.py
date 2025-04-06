import numpy as np
from scipy.linalg import block_diag

from pybbtd.btd import validate_R_L


def _constraint_matrix(R, L):
    """Compute the constraint matrix repeating columsn

    Args:
        R: rank of the decomposition
        L: int or sequence of L values for each block
    """
    _, Larray = validate_R_L(R, L)
    theta = block_diag(*[np.ones(Lr) for Lr in Larray])

    return theta
