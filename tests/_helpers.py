"""Reference implementations used to check the package independently.

The term tensors are built with einsum straight from the model definitions,
not with the package's own reconstruction code.
"""

import itertools

import numpy as np


def relative_error(estimate, reference):
    """Frobenius-norm error of ``estimate`` relative to ``reference``."""
    return np.linalg.norm(estimate - reference) / np.linalg.norm(reference)


def btd_terms(A, B, C, L):
    """Term tensors (A_r B_r^T) o c_r of an LL1 BTD, with block sizes L."""
    offsets = np.concatenate([[0], np.cumsum(L)])
    return [
        np.einsum("ia,ja,k->ijk", A[:, s:e], B[:, s:e], C[:, r])
        for r, (s, e) in enumerate(itertools.pairwise(offsets))
    ]


def bbtd_terms(A, B, C, D, L1, L2):
    """Term tensors (A_r B_r^T) o (C_r D_r^T) of a BBTD."""
    R = A.shape[1] // L1
    return [
        np.einsum(
            "ia,ja,kb,mb->ijkm",
            A[:, r * L1 : (r + 1) * L1],
            B[:, r * L1 : (r + 1) * L1],
            C[:, r * L2 : (r + 1) * L2],
            D[:, r * L2 : (r + 1) * L2],
        )
        for r in range(R)
    ]


def term_recovery_error(estimated_terms, true_terms):
    """Worst relative error between matched terms, over the best term order.

    Decompositions are only unique up to the order of their terms, so the
    estimated terms are compared with the true ones under every permutation
    and the best one is kept.
    """
    return min(
        max(
            relative_error(estimated_terms[i], true_terms[j])
            for i, j in enumerate(order)
        )
        for order in itertools.permutations(range(len(true_terms)))
    )


def add_noise(T, level):
    """Add Gaussian noise whose norm is ``level`` times the norm of T."""
    noise = np.random.randn(*T.shape)
    if np.iscomplexobj(T):
        noise = noise + 1j * np.random.randn(*T.shape)
    return T + level * np.linalg.norm(T) / np.linalg.norm(noise) * noise


def perturb(factors, level):
    """Return copies of the factors with relative Gaussian perturbations."""
    out = []
    for F in factors:
        noise = np.random.randn(*F.shape)
        out.append(F + level * np.linalg.norm(F) / np.linalg.norm(noise) * noise)
    return out
