# Initialization and convergence

## Starting points

Each solver starts from initial factors, chosen with the `init` argument of `fit`:

| `init` | Available for | Starting point |
|---|---|---|
| `"random"` | all models | random factors (non-negative where the model requires it) |
| `"svd"` | BTD, BBTD | leading singular vectors of the tensor unfoldings |
| `"kmeans"` | Cov-LL1, Stokes, BBTD with ADMM | pixels grouped by their signature with K-means; each group gives a spatial map (factored with NMF) and its center gives the signature |

K-means works well when most pixels belong to one term. The draws use NumPy's global
random generator, so `np.random.seed(...)` before `fit` makes a run reproducible.

## Several starts

ALS and AO-ADMM can stop at a poorer solution than the best one. Fitting from a few starts
and keeping the run with the lowest final error is a simple safeguard:

```python
best = None
for _ in range(5):
    model.fit(data, init="random")
    if best is None or model.fit_error[-1] < best[0]:
        best = (model.fit_error[-1], model.factors)
```

## When a fit stops

A fit stops at the first of three events: the error stops decreasing (`rel_tol`), the
error is small enough (`abs_tol`), or `max_iter` iterations are done. In the last case a
`RuntimeWarning` reports it, since the solver may not have converged.

The error measure and the exact rules depend on the solver. In this table, $e_k$ is
`fit_error[k]` and $\hat{\mathcal{T}}$ the fitted tensor:

| Solver | `fit_error` | Stops when | Defaults (`max_iter`, `rel_tol`, `abs_tol`) |
|---|---|---|---|
| BTD, ALS | $\lVert\mathcal{T}-\hat{\mathcal{T}}\rVert^2$ | $\lvert e_k-e_{k-1}\rvert / e_k$ < `rel_tol` or $e_k / \lVert\mathcal{T}\rVert$ < `abs_tol` | 1000, 1e-3, 1e-8 |
| BBTD, ALS | $\lVert\mathcal{T}-\hat{\mathcal{T}}\rVert / \lVert\mathcal{T}\rVert$ | $\lvert e_k-e_{k-1}\rvert / e_{k-1}$ < `rel_tol` or $e_k$ < `abs_tol` | 1000, 1e-3, 1e-8 |
| BBTD, ADMM | $\lVert\mathcal{T}-\hat{\mathcal{T}}\rVert / \lVert\mathcal{T}\rVert$ | $\lvert e_k-e_{k-1}\rvert$ divided by $e_k$ or by $e_{k-1}$ < `rel_tol`, or $e_k$ < `abs_tol` | 1000, 1e-6, 1e-12 |
| Stokes, ADMM | $\lVert\mathcal{T}-\hat{\mathcal{T}}\rVert^2$ | $\lvert e_k-e_{k-1}\rvert / e_k$ < `rel_tol` or $e_k / \lVert\mathcal{T}\rVert$ < `abs_tol` | 1000, 1e-10, 1e-10 |
| Cov-LL1, ADMM | $\lVert\mathcal{T}-\hat{\mathcal{T}}\rVert^2 / \lVert\mathcal{T}\rVert^2$ | $\lvert e_k-e_{k-1}\rvert / e_k$ < `rel_tol` or $e_k / \lVert\mathcal{T}\rVert$ < `abs_tol` | 1000, 1e-10, 1e-10 |

`fit_error[0]` is the error of the starting point, except for BBTD with ADMM, whose list
starts after the first iteration.

## ADMM settings

The AO-ADMM solvers update one factor at a time with a few inner ADMM iterations, which
enforce the constraints.

| Argument | Meaning |
|---|---|
| `rho` | initial penalty of the inner ADMM loops; it is adapted during the fit |
| `max_admm` | inner iterations per factor update (default 1; 50 for BBTD) |
| `admm_tol` | tolerance of the inner loops |
| `gamma` | BBTD only: weight that ties each map $\mathbf{A}_r\mathbf{B}_r^\top$ to its non-negative part |

More inner iterations make each update closer to the exact constrained solution, at a
higher cost per outer iteration.

## Reading the fit error

{func}`~pybbtd.visualization.plot_error` plots `model.fit_error` on a log scale, and can
compare several runs:

```python
from pybbtd.visualization import plot_error

plot_error([run_1_error, run_2_error], labels=["random init", "SVD init"])
```

A curve that drops then levels off has converged. A curve still decreasing at the end
needs a larger `max_iter`. Two runs that level off at different values ended at different
solutions; keep the lower one.

## Progress messages

The solvers report convergence and progress through Python's `logging` module, silent by
default. To show the messages:

```python
import logging

logging.basicConfig(level=logging.INFO, format="%(name)s: %(message)s")
```
