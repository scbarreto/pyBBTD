# Stokes

The **Stokes** model is a BTD in rank-(L, L, 1) terms for polarization images. Every pixel
holds a Stokes vector, and each term is a non-negative spatial map times one physically
valid Stokes vector.

**Use it for** polarization images, to find regions that share the same polarization
state.

## Stokes vectors

The Stokes vector $[S_0, S_1, S_2, S_3]$ describes the light at one pixel:

- $S_0$ is the total intensity;
- $S_1$ compares horizontal and vertical linear polarization;
- $S_2$ compares linear polarization at +45° and −45°;
- $S_3$ measures circular polarization; its sign gives the handedness.

A Stokes vector is physically valid when $S_0 \geq \sqrt{S_1^2 + S_2^2 + S_3^2}$: the
polarized part cannot carry more power than the light itself. Equality means the light is
fully polarized. {func}`~pybbtd.stokes.is_valid_stokes` checks one vector and
{func}`~pybbtd.stokes.validate_stokes_tensor` checks every pixel of an image.

## The model

For a tensor $\mathcal{T}$ of shape (I, J, 4):

$$
\mathcal{T} \approx \sum_{r=1}^{R} \left(\mathbf{A}_r \mathbf{B}_r^{\top}\right) \circ \mathbf{s}_r
$$

with non-negative spatial factors and valid Stokes vectors $\mathbf{s}_r$, stored as the
columns of `C` (4 × R).

## Parameters

| Parameter | Meaning |
|---|---|
| `spatial_dims` | image size (I, J); the tensor has shape (I, J, 4) |
| `R` | number of terms (polarization states) |
| `L` | rank of the spatial maps |

## Fitting

{meth}`Stokes.fit <pybbtd.stokes.Stokes.fit>` uses AO-ADMM
({func}`~pybbtd.solvers.stokes_admm.fit_stokes_admm`), with a random or K-means start.

```python
from pybbtd import Stokes
from pybbtd.visualization import plot_stokes_terms

model = Stokes((160, 160), R=5, L=25)
model.fit(data, init="kmeans", max_iter=5000, max_admm=2)
A, B, C = model.factors
plot_stokes_terms(A, B, C, R=5, L=25)
```

**Tutorials:** [a synthetic polarization image](../tutorials/stokes-synthetic.ipynb) and
[the BBTD board](../tutorials/stokes-bbtd-board.ipynb).
**API:** {class}`pybbtd.stokes.Stokes`.
