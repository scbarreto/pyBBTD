import numpy as np
import matplotlib.pyplot as plt
from palettable.cmocean.sequential import Ice_15


def stokes_to_ellipse(S, n_points=200):
    """
    Convert a Stokes vector to a polarization ellipse in the transverse plane.

    Computes the orientation angle :math:`\\psi` and ellipticity angle
    :math:`\\chi` from a single Stokes vector, then traces out the
    corresponding normalised polarization ellipse.

    Parameters
    ----------
    S : array-like
        Stokes vector ``[S0, S1, S2, S3]``.
    n_points : int, optional
        Number of points used to sample the ellipse (default: 200).

    Returns
    -------
    x : np.ndarray
        Horizontal coordinates of the ellipse, shape ``(n_points,)``.
    y : np.ndarray
        Vertical coordinates of the ellipse, shape ``(n_points,)``.
    psi : float
        Orientation angle in radians.
    chi : float
        Ellipticity angle in radians.  The sign of ``tan(chi)``
        encodes the handedness (positive = left-handed,
        negative = right-handed).
    """
    S0, S1, S2, S3 = S

    # Orientation (psi) and ellipticity angle (chi)
    psi = 0.5 * np.arctan2(S2, S1)  # radians
    chi = 0.5 * np.arctan2(S3, np.sqrt(S1**2 + S2**2))

    # Semi-axes (unnormalized)
    a = np.cos(chi)  # major-ish
    b = np.sin(chi)  # minor-ish, carries handedness sign

    # Parametric ellipse in principal frame
    t = np.linspace(0, 2 * np.pi, n_points)
    xp = a * np.cos(t)
    yp = b * np.sin(t)

    # Rotate by psi into lab frame
    cpsi = np.cos(psi)
    spsi = np.sin(psi)
    x = xp * cpsi - yp * spsi
    y = xp * spsi + yp * cpsi

    # Normalize ellipse size for visualization
    rmax = np.sqrt(x**2 + y**2).max()
    if rmax > 0:
        x = x / rmax
        y = y / rmax

    return x, y, psi, chi


def plot_stokes_terms(outA, outB, outC, R, L, cmap=None):
    """
    Visualise the spatial maps and polarization ellipses of a Stokes-BTD.

    For each component *r* the function displays:

    * **Top row** — the rank-*L* spatial reconstruction
      :math:`A_r B_r^{\\top}`, min–max normalised, as a heatmap.
    * **Bottom row** — the polarization ellipse derived from the
      Stokes vector :math:`c_r`, annotated with orientation
      :math:`\\psi`, ellipticity :math:`\\chi`, and handedness.

    Parameters
    ----------
    outA : np.ndarray
        Spatial factor matrix of shape ``(N1, R*L)``.
        Columns are grouped in contiguous blocks of *L* for each
        component.
    outB : np.ndarray
        Spatial factor matrix of shape ``(N2, R*L)``, with the same
        block structure as *outA*.
    outC : np.ndarray
        Stokes matrix of shape ``(4, R)``.  Each column is a Stokes
        vector :math:`[S_0, S_1, S_2, S_3]^{\\top}`.
    R : int
        Number of components (block terms).
    L : int
        Column rank shared by every component in *outA* / *outB*.
    cmap : matplotlib.colors.Colormap, optional
        Colormap for the heatmaps.  Defaults to
        ``palettable.cmocean.sequential.Ice_15`` when available,
        otherwise ``plt.cm.viridis``.

    Returns
    -------
    fig : matplotlib.figure.Figure
        The created figure.
    axes : np.ndarray
        Array of axes with shape ``(2, R+1)``.  The last column
        holds the shared colorbar (top) and an empty slot (bottom).
    """

    # ---------- Safety / sanity checks ----------
    # Check outC
    if outC.ndim != 2 or outC.shape[0] != 4:
        raise ValueError("outC must have shape (4, R) with rows [S0,S1,S2,S3].")
    if outC.shape[1] != R:
        raise ValueError(f"outC has {outC.shape[1]} columns, but R={R}.")

    # Check outA, outB
    if outA.ndim != 2 or outB.ndim != 2:
        raise ValueError("outA and outB must be 2D arrays.")

    N1, colsA = outA.shape
    N2, colsB = outB.shape

    if colsA != R * L:
        raise ValueError(f"outA has {colsA} columns, expected R*L = {R * L}.")
    if colsB != R * L:
        raise ValueError(f"outB has {colsB} columns, expected R*L = {R * L}.")

    # Set a colormap
    if cmap is None:
        try:
            cmap = Ice_15.mpl_colormap
        except NameError:
            cmap = plt.cm.viridis

    # ---------- Figure layout ----------
    # R heatmaps in the top row, plus one extra column for the colorbar.
    # Bottom row: R ellipses + one empty slot.
    width_ratio = np.ones(R)
    width_ratio = np.append(width_ratio, 0.35)  # skinny column for colorbar

    fig, axes = plt.subplots(
        2,
        R + 1,
        figsize=(3.5 * (2 + 2 * R), 3.5 * 3),
        gridspec_kw={
            "width_ratios": width_ratio,
            "height_ratios": [1, 0.35],
        },
    )

    last_im = None

    result = np.zeros((R, N1, N2))

    # ---------- Top row: A_r B_r^T heatmaps ----------
    for r in range(R):
        # Slice out the r-th block of L columns
        A_block = outA[:, r * L : (r + 1) * L]  # shape (Na, L)
        B_block = outB[:, r * L : (r + 1) * L]  # shape (Nb, L)

        result[r] = A_block @ B_block.T

        # Min-max normalize for display so contrast is comparable
        block = result[r]
        block_min = block.min()
        block_max = block.max()
        denom = block_max - block_min
        if denom == 0:
            # Avoid division by zero: flat patch
            normalized_block = np.zeros_like(block)
        else:
            normalized_block = (block - block_min) / denom

        # Show the normalized block
        ax_heat = axes[0, r]
        im = ax_heat.imshow(normalized_block, cmap=cmap, origin="upper")
        last_im = im

        # No ticks, just the visual texture
        ax_heat.axis("off")

        ax_heat.set_title(
            rf"$A_{{{r + 1}}}B_{{{r + 1}}}^{{\top}}$",
            fontsize=40,
        )

    # Add a shared colorbar in the last column of the top row
    cax = axes[0, R]
    if last_im is not None:
        cbar = plt.colorbar(
            last_im,
            ax=cax,
            aspect=6,
            fraction=0.7,
            location="left",
        )
        cbar.set_ticks([0.0, 1.0])
        cbar.ax.tick_params(labelsize=30)

    cax.axis("off")

    # ---------- Bottom row: polarization ellipses from Stokes vectors ----------
    for idx in range(R):
        ax_ellipse = axes[1, idx]

        S = outC[:, idx]  # [S0, S1, S2, S3]
        x, y, psi, chi = stokes_to_ellipse(S)

        # Draw the ellipse trace
        ax_ellipse.plot(x, y)

        # Compute orientation / ellipticity
        S0, S1, S2, S3 = S

        # Signed ellipticity ratio (minor/major). Sign gives handedness.
        ellipticity_ratio = np.tan(chi)
        handed = "LH" if ellipticity_ratio > 0 else "RH"

        # Format subplot title
        ax_ellipse.set_title(
            rf"$c_{{{idx + 1}}}$:  "
            rf"$\psi = {np.degrees(psi):.1f}^\circ$"
            "\n"
            rf"$\chi = {np.degrees(chi):.1f}^\circ$ "
            rf"({handed})",
            fontsize=30,
        )

        # Fixed aspect and clean grid
        ax_ellipse.set_aspect("equal", "box")
        ax_ellipse.grid(True, linestyle="--", linewidth=0.5, alpha=0.5)
        ax_ellipse.set_xlim(-1.1, 1.1)
        ax_ellipse.set_ylim(-1.1, 1.1)

        # Strip ticks and labels to emphasize shape only
        ax_ellipse.tick_params(
            axis="both",
            which="both",
            bottom=False,
            top=False,
            left=False,
            right=False,
            labelbottom=False,
            labelleft=False,
        )
        ax_ellipse.set_xlabel("")
        ax_ellipse.set_ylabel("")

    # The final (R+1)-th axis in the bottom row is empty
    axes[1, R].axis("off")

    plt.tight_layout()
    plt.show()
