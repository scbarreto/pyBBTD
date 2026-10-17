"""Building blocks of the K-means initializations.

The K-means initializations assume each pixel belongs mostly to one term:

1. cluster the pixels by their signature (Stokes vector, covariance, ...);
2. turn each cluster into a spatial map, and factor it with NMF to get
   non-negative blocks of A and B;
3. use the cluster centers to initialize the signature factor.
"""

import numpy as np
from sklearn.cluster import KMeans
from tensorly import unfold


def cluster_pixels(T, R):
    """Cluster the pixels of a third-order tensor with K-means.

    Parameters
    ----------
    T : np.ndarray
        Real tensor of shape ``(I, J, N)``; each pixel is a vector of length N.
    R : int
        Number of clusters.

    Returns
    -------
    labels : np.ndarray
        Cluster of each pixel, shape ``(I * J,)``, in the order of
        ``unfold(T, 2).T``.
    centers : np.ndarray
        Cluster centers, one per column, shape ``(N, R)``.
    """
    kmeans = KMeans(n_clusters=R, random_state=0, n_init="auto")
    kmeans.fit(unfold(T, 2).T)
    return kmeans.labels_, kmeans.cluster_centers_.T


def cluster_maps(labels, R, shape, random_weights):
    """Turn cluster labels into R spatial maps of the given shape.

    Pixels outside cluster ``r`` are zero in map ``r``. Pixels inside it are
    one, or a uniform random weight when ``random_weights`` is True.

    Returns
    -------
    np.ndarray
        Array of shape ``(R, *shape)``.
    """
    Rterms = np.zeros((R, len(labels)))
    for idx in range(len(labels)):
        Rterms[labels[idx], idx] = np.random.rand() if random_weights else 1
    return Rterms.reshape(R, *shape)


def nmf_factors(maps, nmf):
    """Factor each map with NMF and concatenate the blocks.

    Parameters
    ----------
    maps : np.ndarray
        Spatial maps of shape ``(R, I, J)``.
    nmf : sklearn.decomposition.NMF
        NMF estimator whose ``n_components`` is the block rank L.

    Returns
    -------
    A : np.ndarray
        Shape ``(I, R * L)``.
    B : np.ndarray
        Shape ``(J, R * L)``.
    """
    W, H = [], []
    for spatial_map in maps:
        W.append(nmf.fit_transform(spatial_map))
        H.append(nmf.components_.T)
    return np.concatenate(W, axis=1), np.concatenate(H, axis=1)


def match_norm(A, B, T, T_init):
    """Scale A and B so the initial tensor has the same norm as the data."""
    scale = (np.linalg.norm(T) / np.linalg.norm(T_init)) ** (1 / 2)
    return A * scale, B * scale
