# Choosing a model

All models split an image into **R terms**. Each term is a spatial map times a signature;
the models differ in what the signature is, which depends on what each pixel holds.

| Each pixel holds | Example | Model | Tensor shape | Fitting algorithm |
|---|---|---|---|---|
| K numbers | spectral channels | [BTD](btd.md) | (I, J, K) | ALS |
| a K × K covariance matrix, as K² numbers | covariance imaging | [Cov-LL1](covll1.md) | (I, J, K²) | AO-ADMM |
| a Stokes vector | polarization imaging | [Stokes](stokes.md) | (I, J, 4) | AO-ADMM |
| a K × M matrix | any matrix-valued image | [BBTD](bbtd.md) | (I, J, K, M) | ALS |
| a K × K covariance matrix | covariance imaging | [BBTD](bbtd.md) with `algorithm="ADMM"` | (I, J, K, K) | AO-ADMM |

Cov-LL1 and BBTD can both describe covariance images. BBTD keeps each pixel's matrix as a
matrix and controls the rank of each term's covariance matrix; Cov-LL1 works with
vectorized matrices and does not constrain their rank.

## The same steps for every model

```python
from pybbtd import Stokes  # or BTD, CovLL1, BBTD

model = Stokes(spatial_dims, R=5, L=25)  # 1. describe the model
model.check_uniqueness()                 # 2. are the terms guaranteed to be unique?
model.fit(data)                          # 3. estimate the terms

model.factors    # the factor matrices [A, B, C] (or [A, B, C, D] for BBTD)
model.tensor     # the tensor rebuilt from the factors
model.fit_error  # the error at each iteration
```

The spatial map of term r is `A_r @ B_r.T`, where `A_r` and `B_r` are its block of L
columns in `A` and `B`; {func}`pybbtd.btd.spatial_maps` returns all of them.

The choice of R and L is discussed in [Ranks and uniqueness](../guides/ranks-and-uniqueness.md),
and the fitting options in [Initialization and convergence](../guides/fitting.md).
