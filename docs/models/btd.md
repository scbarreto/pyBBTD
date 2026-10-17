# BTD

The **block term decomposition** (BTD) in rank-(L, L, 1) terms splits a third-order tensor,
such as a multichannel image, into R terms. Each term is a spatial map of rank L times a
vector: the term's spectrum, or more generally its signature across the third dimension.

**Use it for** images with K numbers per pixel (for example K spectral channels), when the
scene is made of a few components, each with its own layout and its own signature.

## The model

For a tensor $\mathcal{T}$ of shape (I, J, K):

$$
\mathcal{T} \approx \sum_{r=1}^{R} \left(\mathbf{A}_r \mathbf{B}_r^{\top}\right) \circ \mathbf{c}_r
$$

- $\mathbf{A}_r$ (I × L) and $\mathbf{B}_r$ (J × L) give the spatial map
  $\mathbf{A}_r \mathbf{B}_r^{\top}$, a matrix of rank L;
- $\mathbf{c}_r$ (length K) is the signature of term r;
- $\circ$ is the outer product: pixel (i, j) of term r holds the map value at (i, j)
  times $\mathbf{c}_r$.

The values can have any sign. The factors are stored as `A` (I × R·L), `B` (J × R·L) and
`C` (K × R), with the columns of term r in the r-th block.

## Parameters

| Parameter | Meaning |
|---|---|
| `dims` | shape (I, J, K) of the tensor |
| `R` | number of terms |
| `L` | rank of the spatial maps: one integer for all terms, or one per term |
| `block_mode` | which dimension holds the signature: `"LL1"` (third, default), `"L1L"` (second) or `"1LL"` (first) |

## Fitting

{meth}`BTD.fit <pybbtd.btd.BTD.fit>` uses alternating least squares
({func}`~pybbtd.solvers.btd_als.fit_btd_als`), with a random or SVD-based start
(`init="random"` or `init="svd"`).

```python
from pybbtd import BTD

model = BTD((80, 100, 4), R=3, L=5)
model.fit(data, init="svd", max_iter=3000, rel_tol=1e-10)
A, B, C = model.factors
```

**Tutorial:** [BTD: separate a multichannel image](../tutorials/btd.ipynb).
**API:** {class}`pybbtd.btd.BTD`.
