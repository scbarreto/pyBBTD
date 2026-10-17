"""Plots for the covariance BBTD model."""

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.gridspec import GridSpec

from pybbtd.btd import spatial_maps


def plot_bbtd_cov_terms(A, B, C, R, L1, L2, cmap_spatial=None, cmap_cov="seismic"):
    """Show the spatial map and covariance matrix of each BBTD term.

    For each term *r*, the top row shows the non-negative spatial map
    :math:`\\max(0, \\mathbf{A}_r \\mathbf{B}_r^{\\top})` and the bottom row
    shows the real part of the covariance matrix
    :math:`\\mathbf{C}_r \\mathbf{C}_r^{H}`, scaled so its largest entry has
    magnitude 1.

    Parameters
    ----------
    A : np.ndarray
        Spatial factor of shape ``(I, R * L1)``; the columns of term *r* are
        the *r*-th block of ``L1`` columns.
    B : np.ndarray
        Spatial factor of shape ``(J, R * L1)``, with the same blocks as A.
    C : np.ndarray
        Spectral factor of shape ``(K, R * L2)``; the columns of term *r* are
        the *r*-th block of ``L2`` columns.
    R : int
        Number of terms.
    L1 : int
        Rank of the spatial maps.
    L2 : int
        Rank of the covariance matrices.
    cmap_spatial : str or matplotlib.colors.Colormap, optional
        Colormap of the spatial maps (Matplotlib default when ``None``).
    cmap_cov : str or matplotlib.colors.Colormap, default "seismic"
        Colormap of the covariance matrices.
    """
    maps = np.maximum(0, spatial_maps(A, B, L1))
    spatial_vmax = 0.5 * maps.max()

    # last column reserved for the colorbars
    fig = plt.figure(figsize=(4 * R + 2, 8))
    gs = GridSpec(2, R + 1, width_ratios=[1] * R + [0.05], wspace=0.15, hspace=0.25)

    for r in range(R):
        ax_map = fig.add_subplot(gs[0, r])
        im_map = ax_map.imshow(maps[r], cmap=cmap_spatial, vmin=0, vmax=spatial_vmax)
        ax_map.set_title(f"Spatial r={r}")
        ax_map.axis("off")

        C_r = C[:, r * L2 : (r + 1) * L2]
        cov = C_r @ C_r.T.conj()
        max_abs = np.max(np.abs(cov))
        cov_norm = cov / max_abs if max_abs > 0 else cov

        ax_cov = fig.add_subplot(gs[1, r])
        im_cov = ax_cov.imshow(cov_norm.real, cmap=cmap_cov, vmin=-1, vmax=1)
        ax_cov.set_title(f"Covariance r={r}")
        ax_cov.axis("off")

    fig.colorbar(im_map, cax=fig.add_subplot(gs[0, -1]))
    fig.colorbar(im_cov, cax=fig.add_subplot(gs[1, -1]))

    plt.show()
