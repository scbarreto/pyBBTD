# Cov-LL1

**Cov-LL1** is a BTD in rank-(L, L, 1) terms for covariance images, where every pixel holds
a K × K covariance matrix. Each term is a non-negative spatial map times one covariance
matrix.

**Use it for** images where each pixel holds a covariance matrix (Hermitian and positive
semidefinite) and the scene is made of a few components, each with its own covariance.

```{image} ../images/model_covll1.png
:alt: A third-order covariance imaging tensor written as a sum of R spatial maps S_r, each times one vectorized covariance matrix
```

## The model

Each pixel's matrix is stored as a vector of K² numbers, so the data form a tensor
$\mathcal{X}$ of shape (I, J, K²):

$$
\mathcal{X} \approx \sum_{r=1}^{R} \left(\mathbf{A}_r \mathbf{B}_r^{\top}\right) \circ \operatorname{vec}(\boldsymbol{\Sigma}_r)
$$

- $\mathbf{A}_r \mathbf{B}_r^{\top}$ is the spatial map of term r, with rank $L_1$ and
  non-negative factors;
- $\boldsymbol{\Sigma}_r$ is the K × K covariance matrix of term r, kept positive
  semidefinite during the fit.

## Parameters

| Parameter | Meaning |
|---|---|
| `dims` | shape (I, J, K²) of the tensor; the last dimension must be a perfect square |
| `R` | number of terms |
| `L1` | rank of the spatial maps |
| `L2` | rank of the covariance matrices, used only by {meth}`~pybbtd.covll1.CovLL1.generate_covll1_tensor` to draw synthetic data |

## Fitting

{meth}`CovLL1.fit <pybbtd.covll1.CovLL1.fit>` uses AO-ADMM
({func}`~pybbtd.solvers.covll1_admm.fit_covll1_admm`), with a random or K-means start
(`init="random"` or `init="kmeans"`).

```python
from pybbtd import CovLL1

model = CovLL1((40, 40, 9), R=3, L1=2, L2=2)
model.fit(data, init="kmeans", max_iter=2000, max_admm=50)
A, B, C = model.factors  # C[:, r].reshape(3, 3) is the covariance matrix of term r
```

To constrain the rank of the covariance matrices as well, use the fourth-order
[BBTD](bbtd.md) model.

**Tutorial:** [Cov-LL1: covariance images](../tutorials/covll1.ipynb).
**API:** {class}`pybbtd.covll1.CovLL1`.
