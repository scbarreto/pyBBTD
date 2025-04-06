import numpy as np
from pybbtd.uniqueness import check_uniqueness_LL1
from scipy.linalg import block_diag
from tensorly.cp_tensor import cp_to_tensor
from pybbtd.solvers.btd_als import BTD_ALS


class BTD:
    """
    Class for Tensors admitting a Block Terms Decomposition (BTD) into rank-(L, L, 1) terms.
    """

    def __init__(self, dims, R: int, L: int, block_mode="LL1"):
        # Validate dims
        if (
            not isinstance(dims, (list, tuple))
            or len(dims) != 3
            or not all(isinstance(d, int) and d > 0 for d in dims)
        ):
            raise ValueError(
                "dims should be a list or tuple of three positive integers."
            )
        self.dims = tuple(dims)

        # validate R, L values
        self.rank, self.L = validate_R_L(R, L)

        # Validate mode
        valid_block_modes = {"LL1", "L1L", "1LL"}
        if block_mode not in valid_block_modes:
            raise ValueError(
                f"Invalid mode '{block_mode}'. Block mode must be one of {valid_block_modes}."
            )
        self.block_mode = block_mode

        # check uniqueness, raise warning if parameters cannot be guaranteed to lead to unique solution
        self.check_uniqueness()

        # Initialize additional variables
        self.factors = None
        self.tensor = None
        self.fit_error = None  # stores last fit error

    def check_uniqueness(self):
        if not np.all(self.L == self.L[0]):
            return print(
                "Uniqueness checks are currently implemented for the constant L case.\nSkipping uniqueness tests."
            )

        N1, N2, N3 = self.dims

        if self.block_mode == "LL1":
            unique = check_uniqueness_LL1(
                N1, N2, N3, self.rank, self.L[0]
            )  # should fix value of L
        elif self.block_mode == "L1L":
            unique = check_uniqueness_LL1(
                N1, N3, N2, self.rank, self.L[0]
            )  # should fix value of L
        elif self.block_mode == "1LL":
            unique = check_uniqueness_LL1(
                N2, N3, N1, self.rank, self.L[0]
            )  # should fix value of L
        if unique is True:
            print("Sufficient condition for uniqueness satisfied")
        else:
            print("Cannot guarantee uniqueness. Proceed at your own risk.")

    def fit(self, data, algorithm="ALS", **kwargs):
        if algorithm == "ALS":
            self.factors, self.fit_error = BTD_ALS(self, data, **kwargs)
            self.tensor = factors_to_tensor(
                *self.factors, self.get_constraint_matrix(), block_mode=self.block_mode
            )
        else:
            raise UserWarning("Algorithm not implemented yet")

    def get_constraint_matrix(self):
        return constraint_matrix(self.rank, self.L)

    def to_cpd_format():
        """
        Convert the BTD to CPD format.
        """
        pass


def validate_R_L(R, L):
    # check R
    if not isinstance(R, int) or R <= 0:
        raise ValueError("R should be a positive integer.")

    # Validate L
    if isinstance(L, int):
        Larray = np.ones(R, dtype=int) * L
    elif isinstance(L, (list, np.ndarray)) and len(L) == R:
        Larray = np.array(L)
    else:
        raise ValueError(
            "L should be either a single integer or a list/array of length R."
        )

    return R, Larray


def constraint_matrix(R, L):
    """Compute the constraint matrix repeating columns
    for the BTD model

    Args:
        R: rank of the decomposition
        L: int or sequence of L values for each block
    """
    _, Larray = validate_R_L(R, L)
    theta = block_diag(*[np.ones(Lr) for Lr in Larray])

    return theta


def factors_to_tensor(A, B, C, theta, block_mode="LL1"):
    if block_mode == "LL1":
        Trec = cp_to_tensor((np.ones(theta.shape[1]), [A, B, C @ theta]))
    elif block_mode == "1LL":
        Trec = cp_to_tensor((np.ones(theta.shape[1]), [A @ theta, B, C]))
    elif block_mode == "L1L":
        Trec = cp_to_tensor((np.ones(theta.shape[1]), [A, B @ theta, C]))
    return Trec
