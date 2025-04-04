import numpy as np


class BTD:
    """
    Class for Tensors admitting a Block Terms Decomposition (BTD) into rank-(L, L, 1) terms. Block-Term Decomposition
    """

    def __init__(self, R: int, L: int, mode="LL1"):
        # Validate R
        if not isinstance(R, int) or R <= 0:
            raise ValueError("R should be a positive integer.")
        self.rank = R

        # Validate L
        if isinstance(L, int):
            self.L = np.ones(self.rank) * L
        elif isinstance(L, (list, np.ndarray)) and len(L) == R:
            self.L = np.array(L)
        else:
            raise ValueError(
                "L should be either a single integer or a list/array of length R."
            )

        # Validate mode
        valid_modes = {"LL1", "L1L", "1LL"}
        if mode not in valid_modes:
            raise ValueError(
                f"Invalid mode '{mode}'. Mode must be one of {valid_modes}."
            )
        self.mode = mode

        # Initialize additional variables
        self.factors = None
        self.T = None
        self.unfoldings = None

    def fit(algorithm="ALS"):
        pass

    def get_constraint_matrices(self):
        pass

    def to_cpd_format():
        """
        Convert the BTD to CPD format.
        """
        pass
