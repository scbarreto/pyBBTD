# Installation

pyBBTD needs Python 3.11 or newer. Install it from GitHub with pip:

```bash
pip install git+https://github.com/scbarreto/pyBBTD
```

This also installs its dependencies: NumPy, SciPy, TensorLy, scikit-learn, Matplotlib and
palettable.

To run the tutorials in Jupyter, install the `notebooks` extra as well:

```bash
pip install "pybbtd[notebooks] @ git+https://github.com/scbarreto/pyBBTD"
```

Check the installation:

```python
import pybbtd

print(pybbtd.__version__)
```

To work on pyBBTD itself, see [Contributing](../contributing.md).
