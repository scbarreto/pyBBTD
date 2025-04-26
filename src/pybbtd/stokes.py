from pybbtd.btd import BTD
import numpy as np
import pybbtd.btd as btd

import warnings
# Class for Stokes tensors


class Stokes(BTD):
    def __init__(self, spatial_dims, R, L, **kwargs):
        dims = (spatial_dims[0], spatial_dims[1], 4)
        kwargs["block_mode"] = "LL1"
        super().__init__(dims=dims, R=R, L=L, **kwargs)
        print(
            f"Stokes tensor initialized with dimensions {self.dims} on {self.block_mode} mode."
        )

    def generate_stokes_tensor(self):
        """
        Generate a random Stokes tensor."""

        A, B, C = generate_stokes_factors(self.dims, self.rank, self.L)
        # Generate the tensor using the factors
        self.factors = [A, B, C]
        self.tensor = btd.factors_to_tensor(
            A, B, C, self.get_constraint_matrix(), block_mode=self.block_mode
        )
        return self.factors, self.tensor

    def fit(self, data, algorithm="ADMM", **kwargs):
        from pybbtd.solvers.stokes_admm import Stokes_ADMM

        if algorithm == "ADMM":
            self.factors, self.fit_error = Stokes_ADMM(self, data, **kwargs)
            self.tensor = btd.factors_to_tensor(
                *self.factors, self.get_constraint_matrix(), block_mode=self.block_mode
            )
        else:
            raise UserWarning("Algorithm not implemented yet")

    def get_constraint_matrix(self):
        return btd.constraint_matrix(self.rank, self.L)


def generate_stokes_factors(dims, R, L):
    """
    Generate random factors for Stokes tensor.
    """

    A = np.random.rand(dims[0], L[0] * R)
    B = np.random.rand(dims[1], L[0] * R)
    C = np.zeros((dims[2], R))

    for r in range(R):
        cr = 1.0 / np.sqrt(2) * (np.random.randn(2) + 1j * np.random.randn(2))
        cr = cr / np.linalg.norm(cr)

        C[:, r] = coh2stokes(np.outer(cr, cr.conj()))

    return A, B, C


def validate_stokes_tensor(T0):
    """
    Check if all pixels satisfy the Stokes constraints.
    Shows a warning at the end with the percentage of invalid pixels (if any).
    """
    total_pixels = T0.shape[0] * T0.shape[1]
    invalid_count = 0

    for i in range(T0.shape[0]):
        for j in range(T0.shape[1]):
            stokes_vec = T0[i, j, :]
            if check_stokes_constraints(stokes_vec) == 0:
                invalid_count += 1

    if invalid_count > 0:
        percentage = (invalid_count / total_pixels) * 100
        warnings.warn(
            f"{percentage:.2f}% of pixels do not satisfy the Stokes constraints.",
            UserWarning,
        )
    else:
        print("All pixels satisfy the Stokes constraints.")


def check_stokes_constraints(S):
    return (
        0
        if (
            (round(S[0] ** 2, 5) < round(S[1] ** 2 + S[2] ** 2 + S[3] ** 2, 5))
            or (S[0] < 0)
        )
        else 1
    )


def stokes2coh(S):
    "construct Coherence matrix from Stokes parameters"
    S0 = S[0]
    S1 = S[1]
    S2 = S[2]
    S3 = S[3]
    coh = 0.5 * np.array([[S0 + S1, S2 + 1j * S3], [S2 - 1j * S3, S0 - S1]])
    return coh


def coh2stokes(coh):
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
    coh = stokes2coh(S)
    proj = projPSD(coh)
    newS = coh2stokes(proj)
    return newS


# def elip_2_stokes(psi, chi, p=1):
#     S1 = p * np.cos(2 * psi) * np.cos(2 * chi)
#     S2 = p * np.sin(2 * psi) * np.cos(2 * chi)
#     S3 = p * np.sin(2 * chi)
#     return np.array([1, S1, S2, S3])


# def stokes_2_elip(S):
#     S0 = S[0]
#     p = np.sqrt(S[1] ** 2 + S[2] ** 2 + S[3] ** 2) / S[0]
#     psi = 1 / 2 * np.arctan2(S[2], S[1])
#     chi = 1 / 2 * np.arctan2(S[3], np.sqrt(S[1] ** 2 + S[2] ** 2))
#     # chi = 1/2 * np.arcsin(S[3] / S[0])

#     return (psi, chi, p, S0)
