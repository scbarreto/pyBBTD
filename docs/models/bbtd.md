# BBTD

The **block-block term decomposition** (BBTD) in rank-(L1, L1, L2, L2) terms splits a
fourth-order tensor, such as an image with a matrix at each pixel, into R terms. Each term
is a spatial map of rank L1 times a matrix of rank L2.

**Use it for** images where each pixel holds a matrix. For covariance images, the ADMM
solver keeps the spatial maps non-negative and every term's matrix a covariance matrix.

```{image} ../images/model_bbtd.png
:alt: A fourth-order covariance imaging tensor written as a sum of R spatial maps S_r of rank L1, each times one covariance matrix of rank L2
```

## The model

For a tensor $\mathcal{T}$ of shape (I, J, K, M):

$$
\mathcal{T} \approx \sum_{r=1}^{R} \left(\mathbf{A}_r \mathbf{B}_r^{\top}\right) \circ \left(\mathbf{C}_r \mathbf{D}_r^{\top}\right)
$$

- $\mathbf{A}_r \mathbf{B}_r^{\top}$ (I × J) is the spatial map of term r, with rank $L_1$;
- $\mathbf{C}_r \mathbf{D}_r^{\top}$ (K × M) is the matrix of term r, with rank $L_2$.

For covariance images, K = M and $\mathbf{D}_r = \mathbf{C}_r^*$, so the matrix of term r
is the covariance matrix $\mathbf{C}_r \mathbf{C}_r^{H}$.

## Parameters

| Parameter | Meaning |
|---|---|
| `dims` | shape (I, J, K, M) of the tensor |
| `R` | number of terms |
| `L1` | rank of the spatial maps |
| `L2` | rank of the matrices |

## Fitting

{meth}`BBTD.fit <pybbtd.bbtd.BBTD.fit>` offers two algorithms:

| `algorithm` | Solver | Constraints | `init` |
|---|---|---|---|
| `"ALS"` (default) | {func}`~pybbtd.solvers.bbtd_als.fit_bbtd_als` | none | `"random"`, `"svd"` |
| `"ADMM"` | {func}`~pybbtd.solvers.bbtd_cov_admm.fit_bbtd_cov_admm` | non-negative maps, $\mathbf{D} = \mathbf{C}^*$ | `"random"`, `"svd"`, `"kmeans"` |

```python
from pybbtd import BBTD
from pybbtd.visualization import plot_bbtd_cov_terms

model = BBTD((40, 40, 4, 4), R=3, L1=2, L2=2)
model.fit(data, algorithm="ADMM", init="kmeans", max_iter=1000)
A, B, C, D = model.factors
plot_bbtd_cov_terms(A, B, C, R=3, L1=2, L2=2)
```

**Tutorials:** [one matrix per pixel (ALS)](../tutorials/bbtd-als.ipynb) and
[covariance images (ADMM)](../tutorials/bbtd-admm.ipynb).
**API:** {class}`pybbtd.bbtd.BBTD`.
