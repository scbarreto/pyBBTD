from pybbtd.btd import BTD
import numpy as np
import pybbtd.btd as btd


# Class for Stokes tensors


class Stokes(BTD):
    def __init__(self, spatial_dims, R, L, **kwargs):
        dims = (spatial_dims[0], spatial_dims[1], 4)
        self.block_mode = "LL1"
        super().__init__(dims=dims, R=R, L=L, block_mode=self.block_mode, **kwargs)
        self.validate_dims()
        print(
            f"Stokes tensor initialized with dimensions {self.dims} on {self.block_mode} mode."
        )

    def load_stokes_tensor(self, X):
        """
        Load a Stokes tensor from a numpy array.
        """
        self.tensor = X
        self.dims = X.shape
        self.validate_dims()
        self.validate_stokes_tensor()

    def validate_dims(self):
        if self.block_mode != "LL1":
            raise ValueError("Error: Stokes Class only admits LL1 block mode.")
        if self.block_mode == "LL1" and self.dims[2] != 4:
            raise ValueError("Error: Stokes dimension (Rank-1) of tensor must be 4.")

    def generate_stokes_factors(self, dims, R, L):
        """
        Generate random factors for Stokes tensor.
        """

        A = np.random.rand(dims[0], L[0] * R)
        B = np.random.rand(dims[1], L[0] * R)
        C = np.zeros((dims[2], R))

        for r in range(R):
            cr = 1.0 / np.sqrt(2) * (np.random.randn(2) + 1j * np.random.randn(2))
            cr = cr / np.linalg.norm(cr)

            C[:, r] = coh2Stokes(np.outer(cr, cr.conj()))

        return A, B, C

    def generate_stokes_tensor(self):
        """
        Generate a random Stokes tensor."""

        A, B, C = self.generate_stokes_factors(self.dims, self.rank, self.L)
        # Generate the tensor using the factors
        self.factors = [A, B, C]
        self.tensor = btd.factors_to_tensor(
            A, B, C, self.get_constraint_matrix(), block_mode=self.block_mode
        )
        return self.factors, self.tensor

    def validate_stokes_tensor(self):
        """
        Check if all pixels satisfy the Stokes constraints.
        """
        for i in range(self.dims[0]):
            for j in range(self.dims[1]):
                stokes_vec = self.tensor[i, j, :]
                if check_stokes_constraints(stokes_vec) == 0:
                    raise ValueError(
                        f"Stokes tensor at index ({i}, {j}) does not satisfy the constraints."
                    )
        print("All pixels satisfy the Stokes constraints.")


def check_stokes_constraints(S):
    """
    Check the constraints for Stokes tensors.
    """
    # Check if the first element is non-negative and the sum of squares of the other elements is less than or equal to the square of the first element
    return (
        0
        if (
            (round(S[0] ** 2, 5) < round(S[1] ** 2 + S[2] ** 2 + S[3] ** 2, 5))
            or (S[0] < 0)
        )
        else 1
    )


def Stokes2coh(S):
    "construct Coherence matrix from Stokes parameters"
    S0 = S[0]
    S1 = S[1]
    S2 = S[2]
    S3 = S[3]
    coh = 0.5 * np.array([[S0 + S1, S2 + 1j * S3], [S2 - 1j * S3, S0 - S1]])
    return coh


def coh2Stokes(coh):
    "return Stokes parameters from coherence matrix"
    S0 = np.real(coh[0, 0] + coh[1, 1])
    S1 = np.real(coh[0, 0] - coh[1, 1])
    S2 = 2 * coh[0, 1].real
    S3 = 2 * coh[0, 1].imag

    S = np.array([S0, S1, S2, S3])
    return S


def projPSD(M):
    "projection of matrix M onto the set of PSD hermitian matrices"

    symM = 0.5 * (M + M.conj().T)
    w, v = np.linalg.eig(symM)

    proj = np.zeros_like(symM)
    for i in range(len(w)):
        proj += max(0, w[i].real) * np.outer(v[:, i], v[:, i].conj())

    return proj


def stokesProjection(S):
    coh = Stokes2coh(S)
    proj = projPSD(coh)
    newS = coh2Stokes(proj)
    return newS


def elip_2_stokes(psi, chi, p=1):
    S1 = p * np.cos(2 * psi) * np.cos(2 * chi)
    S2 = p * np.sin(2 * psi) * np.cos(2 * chi)
    S3 = p * np.sin(2 * chi)
    return np.array([1, S1, S2, S3])


def stokes_2_elip(S):
    S0 = S[0]
    p = np.sqrt(S[1] ** 2 + S[2] ** 2 + S[3] ** 2) / S[0]
    psi = 1 / 2 * np.arctan2(S[2], S[1])
    chi = 1 / 2 * np.arctan2(S[3], np.sqrt(S[1] ** 2 + S[2] ** 2))
    # chi = 1/2 * np.arcsin(S[3] / S[0])

    return (psi, chi, p, S0)
