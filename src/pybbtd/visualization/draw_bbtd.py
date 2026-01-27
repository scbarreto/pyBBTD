import numpy as np
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec


def plot_BBTD_cov_terms(
    outA, outB, outC, R, L1, L2, cmap_spatial=None, cmap_cov="seismic"
):
    """
    Visualize the spatial maps and covariance matrices of a BBTD model.

    For each component *r* the function displays:

    * **Top row** — the non-negative spatial maps
      :math:`\\max(0,\\; A_r B_r^{\\top})` as a heatmap.
    * **Bottom row** — the normalized sample covariance
      :math:`C_r C_r^{H} / \\|C_r C_r^{H}\\|_{\\infty}` (real part).

    Parameters
    ----------
    outA : np.ndarray
        Spatial factor matrix of shape ``(J, R*L1)``.
        Columns are grouped in contiguous blocks of *L1* for each
        component.
    outB : np.ndarray
        Spatial factor matrix of shape ``(M, R*L1)``, with the same
        block structure as *outA*.
    outC : np.ndarray
        Spectral factor matrix of shape ``(K, R*L2)``.
        Columns are grouped in contiguous blocks of *L2* for each
        component.
    R : int
        Number of components (block terms).
    L1 : int
        Spatial rank shared by every component in *outA* / *outB*.
    L2 : int
        Spectral rank shared by every component in *outC*.
    cmap_spatial : matplotlib.colors.Colormap or str, optional
        Colormap for the spatial heatmaps.  Defaults to the
        Matplotlib default colormap when ``None``.
    cmap_cov : matplotlib.colors.Colormap or str, optional
        Colormap for the covariance matrices (default: ``"seismic"``).

    Returns
    -------
    None
        The figure is displayed via ``plt.show()``.
    """

    # --- Compute spatial maps ---
    J, M = outA.shape[0], outB.shape[0]
    outS = np.zeros((R, J, M))

    for r in range(R):
        outS[r] = np.maximum(
            0, outA[:, r * L1 : (r + 1) * L1] @ outB[:, r * L1 : (r + 1) * L1].T
        )

    spatial_vmax = 0.5 * outS.max()

    # --- GridSpec: last column reserved for colorbars ---
    fig = plt.figure(figsize=(4 * R + 2, 8))
    gs = GridSpec(2, R + 1, width_ratios=[1] * R + [0.05], wspace=0.15, hspace=0.25)

    axes = np.empty((2, R), dtype=object)

    for r in range(R):
        axes[0, r] = fig.add_subplot(gs[0, r])
        axes[1, r] = fig.add_subplot(gs[1, r])

    cax_spatial = fig.add_subplot(gs[0, -1])
    cax_cov = fig.add_subplot(gs[1, -1])

    # --- Plot ---
    for r in range(R):
        # Spatial map
        im_s = axes[0, r].imshow(outS[r], cmap=cmap_spatial, vmin=0, vmax=spatial_vmax)
        axes[0, r].set_title(f"Spatial r={r}")
        axes[0, r].axis("off")

        # Covariance
        C = outC[:, r * L2 : (r + 1) * L2]
        cov = C @ C.T.conj()

        max_abs = np.max(np.abs(cov))
        cov_norm = cov / max_abs if max_abs > 0 else cov

        im_c = axes[1, r].imshow(cov_norm.real, cmap=cmap_cov, vmin=-1, vmax=1)
        axes[1, r].set_title(f"Covariance r={r}")
        axes[1, r].axis("off")

    # --- Colorbars (fixed position on the right) ---
    plt.colorbar(im_s, cax=cax_spatial)
    plt.colorbar(im_c, cax=cax_cov)

    plt.show()
