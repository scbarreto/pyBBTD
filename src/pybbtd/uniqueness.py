"""Sufficient conditions for the BTD and BBTD models to have a unique solution.

A decomposition is *essentially unique* when the terms it finds are the only
ones that explain the data, up to the order of the terms and scalings inside
each term. When uniqueness holds, the recovered terms can be interpreted as
the true underlying sources.
"""

import logging
from math import floor

import numpy as np

from pybbtd._utils import warn_user

logger = logging.getLogger(__name__)


def check_uniqueness_bbtd(N1, N2, N3, N4, R, L1, L2):
    """Check a sufficient uniqueness condition for a rank-(L1, L1, L2, L2) BBTD.

    The fourth-order model is reshaped into two third-order BTD models, and
    the BBTD is unique if either of them is:

    - merging modes 1 and 2 gives a BTD of size ``(N3, N4, N1 * N2)`` with
      block rank ``L2``;
    - merging modes 3 and 4 gives a BTD of size ``(N1, N2, N3 * N4)`` with
      block rank ``L1``.

    Parameters
    ----------
    N1, N2, N3, N4 : int
        Dimensions of the fourth-order tensor.
    R : int
        Number of terms.
    L1 : int
        Rank of the first block (modes 1 and 2).
    L2 : int
        Rank of the second block (modes 3 and 4).

    Returns
    -------
    bool
        ``True`` if a sufficient condition for uniqueness is satisfied.
    """
    if not isinstance(L1, (int, np.integer)):
        raise NotImplementedError("L1 should be an integer")
    if not isinstance(L2, (int, np.integer)):
        raise NotImplementedError("L2 should be an integer")

    # Model 1: vectorize first block (modes 1, 2) -> BTD(N3, N4, N1*N2, L2, R)
    model1_unique = check_uniqueness_ll1(N3, N4, N1 * N2, R, L2)

    # Model 2: vectorize second block (modes 3, 4) -> BTD(N1, N2, N3*N4, L1, R)
    model2_unique = check_uniqueness_ll1(N1, N2, N3 * N4, R, L1)

    return model1_unique or model2_unique


def check_uniqueness_ll1(N1, N2, N3, R, L):
    """Check a sufficient uniqueness condition for a rank-(L, L, 1) BTD.

    Five sufficient conditions from the BTD uniqueness theory are evaluated;
    if any of them holds, the decomposition is essentially unique.

    Parameters
    ----------
    N1, N2, N3 : int
        Dimensions of the third-order tensor (the rank-one mode is the third).
    R : int
        Number of terms.
    L : int
        Rank of the matrix part of each term.

    Returns
    -------
    bool
        ``True`` if a sufficient condition for uniqueness is satisfied.
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


def _report_uniqueness(unique):
    """Log a satisfied condition, or warn the user when it is not satisfied."""
    if unique:
        logger.info("Sufficient condition for uniqueness satisfied.")
    else:
        warn_user(
            "Cannot guarantee uniqueness for these dimensions and ranks: the "
            "recovered terms may not match the true ones. Proceed at your own risk."
        )
