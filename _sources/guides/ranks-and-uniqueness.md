# Ranks and uniqueness

## Choosing R and L

**R** is the number of terms. When the scene is known (five regions, three materials),
use that number. Otherwise, fit a few values of R and compare the final fit errors and the
recovered maps.

**L** (or **L1** for BBTD) is the rank of each spatial map, which limits how complex its
shape can be. A rank-1 map is a column profile times a row profile, such as a blob or a
rectangle aligned with the image axes. Shapes made of several such pieces, or letters,
need a higher rank: the [BBTD board tutorial](../tutorials/stokes-bbtd-board.ipynb) uses
L = 25 for 160 × 160 images of letters.

**L2** (BBTD only) is the rank of each term's matrix. A covariance matrix of rank L2 is
built from L2 independent signals.

## Uniqueness

A decomposition is *essentially unique* when no other set of terms explains the data
equally well, apart from listing the same terms in another order or rescaling a map and
its signature in opposite ways. Only then can the recovered terms be read as the true
underlying components.

pyBBTD checks known **sufficient** conditions when a model is created, and warns when
they do not hold:

```python
from pybbtd import BTD

BTD((60, 80, 6), R=3, L=2).check_uniqueness()  # True
BTD((5, 5, 5), R=3, L=3)  # UserWarning: Cannot guarantee uniqueness ...
```

A warning does not mean the result is wrong: the conditions are sufficient, not
necessary. It means the guarantee does not apply.

| Model | Check | Condition |
|---|---|---|
| BTD, Cov-LL1, Stokes | {func}`~pybbtd.uniqueness.check_uniqueness_ll1` | five algebraic conditions on (I, J, K, R, L); one is enough |
| BBTD | {func}`~pybbtd.uniqueness.check_uniqueness_bbtd` | either of the two third-order BTDs obtained by merging modes 1–2 or modes 3–4 is unique |

For BTD with `block_mode="L1L"` or `"1LL"`, the dimensions are reordered so the
signature's dimension comes last before the check. When L differs between terms, the check
is not available and `check_uniqueness()` returns `None`.

## Uniqueness and finding the solution

Uniqueness says the answer is well defined; it does not guarantee that a solver finds it.
ALS and AO-ADMM can stop at a poorer solution from some starting points, which
[Initialization and convergence](fitting.md) covers.
