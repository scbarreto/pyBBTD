# API reference

The four models are available from the top-level package:

```python
from pybbtd import BBTD, BTD, CovLL1, Stokes
```

| Module | Contents |
|---|---|
| [Models](models.md) | {class}`~pybbtd.btd.BTD`, {class}`~pybbtd.covll1.CovLL1`, {class}`~pybbtd.stokes.Stokes`, {class}`~pybbtd.bbtd.BBTD` and their helpers |
| [Solvers](solvers.md) | the fitting algorithms called by each model's `fit` |
| [Uniqueness](uniqueness.md) | sufficient conditions for a unique decomposition |
| [Visualization](visualization.md) | plots of fit errors and fitted terms |
| [Datasets](datasets.md) | example data used in the tutorials |

```{toctree}
:hidden:

models
solvers
uniqueness
visualization
datasets
```
