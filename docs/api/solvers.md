# Solvers

Each model's `fit` method calls one of these functions; they can also be called directly
with a model and a data tensor.

| Model | `algorithm` | Solver | Initialization |
|---|---|---|---|
| BTD | `"ALS"` | {func}`~pybbtd.solvers.btd_als.fit_btd_als` | {func}`~pybbtd.solvers.btd_als.init_btd_factors` |
| Cov-LL1 | `"ADMM"` | {func}`~pybbtd.solvers.covll1_admm.fit_covll1_admm` | {func}`~pybbtd.solvers.covll1_admm.init_covll1_factors` |
| Stokes | `"ADMM"` | {func}`~pybbtd.solvers.stokes_admm.fit_stokes_admm` | {func}`~pybbtd.solvers.stokes_admm.init_stokes_factors` |
| BBTD | `"ALS"` | {func}`~pybbtd.solvers.bbtd_als.fit_bbtd_als` | {func}`~pybbtd.solvers.bbtd_als.init_bbtd_factors` |
| BBTD | `"ADMM"` | {func}`~pybbtd.solvers.bbtd_cov_admm.fit_bbtd_cov_admm` | {func}`~pybbtd.solvers.bbtd_cov_admm.init_bbtd_cov_factors` |

## BTD: ALS

```{automodule} pybbtd.solvers.btd_als
:members: fit_btd_als, init_btd_factors
```

## Cov-LL1: AO-ADMM

```{automodule} pybbtd.solvers.covll1_admm
:members: fit_covll1_admm, init_covll1_factors
```

## Stokes: AO-ADMM

```{automodule} pybbtd.solvers.stokes_admm
:members: fit_stokes_admm, init_stokes_factors
```

## BBTD: ALS

```{automodule} pybbtd.solvers.bbtd_als
:members: fit_bbtd_als, init_bbtd_factors
```

## BBTD: AO-ADMM for covariance images

```{automodule} pybbtd.solvers.bbtd_cov_admm
:members: fit_bbtd_cov_admm, init_bbtd_cov_factors
```
