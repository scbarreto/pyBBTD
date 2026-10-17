"""Plots for the Stokes model."""

import matplotlib.pyplot as plt
import numpy as np
from palettable.cmocean.sequential import Ice_15

from pybbtd.btd import spatial_maps


def stokes_to_ellipse(S, n_points=200):
    """Trace the polarization ellipse described by a Stokes vector.

    Parameters
    ----------
    S : array-like
        Stokes vector ``[S0, S1, S2, S3]``.
    n_points : int, default 200
        Number of points along the ellipse.

    Returns
    -------
    x, y : np.ndarray
        Coordinates of the ellipse, scaled so its largest radius is 1,
        each of shape ``(n_points,)``.
    psi : float
        Orientation angle, in radians.
    chi : float
        Ellipticity angle, in radians. Its sign gives the handedness
        (positive: left-handed, negative: right-handed).
    """
    _, S1, S2, S3 = S

    # orientation (psi) and ellipticity (chi) angles
    psi = 0.5 * np.arctan2(S2, S1)
    chi = 0.5 * np.arctan2(S3, np.sqrt(S1**2 + S2**2))

    # ellipse in its principal frame (sin(chi) carries the handedness sign)
    t = np.linspace(0, 2 * np.pi, n_points)
    xp = np.cos(chi) * np.cos(t)
    yp = np.sin(chi) * np.sin(t)

    # rotate by psi into the lab frame
    x = xp * np.cos(psi) - yp * np.sin(psi)
    y = xp * np.sin(psi) + yp * np.cos(psi)

    # normalize the size for display
    rmax = np.sqrt(x**2 + y**2).max()
    if rmax > 0:
        x = x / rmax
        y = y / rmax

    return x, y, psi, chi


def plot_stokes_terms(A, B, C, R, L, cmap=None):
    """Show the spatial map and polarization ellipse of each Stokes term.

    For each term *r*, the top row shows the spatial map
    :math:`\\mathbf{A}_r \\mathbf{B}_r^{\\top}` (min-max normalized) and the
    bottom row shows the polarization ellipse of the Stokes vector
    :math:`\\mathbf{c}_r`, with its orientation :math:`\\psi`, ellipticity
    :math:`\\chi` and handedness (LH or RH).

    Parameters
    ----------
    A : np.ndarray
        Spatial factor of shape ``(I, R * L)``; the columns of term *r* are
        the *r*-th block of ``L`` columns.
    B : np.ndarray
        Spatial factor of shape ``(J, R * L)``, with the same blocks as A.
    C : np.ndarray
        Stokes factor of shape ``(4, R)``, one Stokes vector per column.
    R : int
        Number of terms.
    L : int
        Rank of the spatial maps.
    cmap : matplotlib.colors.Colormap, optional
        Colormap of the spatial maps (cmocean "ice" when ``None``).
    """
    if C.ndim != 2 or C.shape[0] != 4:
        raise ValueError("C must have shape (4, R) with rows [S0,S1,S2,S3].")
    if C.shape[1] != R:
        raise ValueError(f"C has {C.shape[1]} columns, but R={R}.")
    if A.ndim != 2 or B.ndim != 2:
        raise ValueError("A and B must be 2D arrays.")
    if A.shape[1] != R * L:
        raise ValueError(f"A has {A.shape[1]} columns, expected R*L = {R * L}.")
    if B.shape[1] != R * L:
        raise ValueError(f"B has {B.shape[1]} columns, expected R*L = {R * L}.")

    if cmap is None:
        cmap = Ice_15.mpl_colormap

    # top row: R maps and a narrow column for the colorbar; bottom row: R ellipses
    fig, axes = plt.subplots(
        2,
        R + 1,
        figsize=(3.5 * (2 + 2 * R), 3.5 * 3),
        gridspec_kw={"width_ratios": [1] * R + [0.35], "height_ratios": [1, 0.35]},
        layout="constrained",
    )

    # top row: A_r B_r^T maps, min-max normalized so contrast is comparable
    for r, block in enumerate(spatial_maps(A, B, L)):
        span = block.max() - block.min()
        normalized = np.zeros_like(block) if span == 0 else (block - block.min()) / span

        im = axes[0, r].imshow(normalized, cmap=cmap, origin="upper")
        axes[0, r].axis("off")
        axes[0, r].set_title(rf"$A_{{{r + 1}}}B_{{{r + 1}}}^{{\top}}$", fontsize=40)

    cbar = plt.colorbar(im, ax=axes[0, R], aspect=6, fraction=0.7, location="left")
    cbar.set_ticks([0.0, 1.0])
    cbar.ax.tick_params(labelsize=30)
    axes[0, R].axis("off")

    # bottom row: polarization ellipses
    for r in range(R):
        x, y, psi, chi = stokes_to_ellipse(C[:, r])
        handedness = "LH" if np.tan(chi) > 0 else "RH"

        ax = axes[1, r]
        ax.plot(x, y)
        ax.set_title(
            rf"$c_{{{r + 1}}}$:  $\psi = {np.degrees(psi):.1f}^\circ$"
            "\n"
            rf"$\chi = {np.degrees(chi):.1f}^\circ$ ({handedness})",
            fontsize=30,
        )
        ax.set_aspect("equal", "box")
        ax.grid(True, linestyle="--", linewidth=0.5, alpha=0.5)
        ax.set_xlim(-1.1, 1.1)
        ax.set_ylim(-1.1, 1.1)
        ax.tick_params(
            axis="both",
            which="both",
            bottom=False,
            top=False,
            left=False,
            right=False,
            labelbottom=False,
            labelleft=False,
        )
    axes[1, R].axis("off")

    plt.show()
