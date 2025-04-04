class BTD:
    """
    Class for Tensors admitting a Block Terms Decomposition (BTD) into rank-(L, L, 1) terms. Block-Term Decomposition
    """

    def __init__(self, rank, L1):
        self.rank = rank
        self.L1 = L1
