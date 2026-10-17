# pyBBTD

**Split images that have several numbers per pixel into a few parts you can interpret.**

Many instruments record more than one number at each pixel: several spectral channels, the
polarization of light, or a whole covariance matrix. pyBBTD explains such an image as a sum
of a few **terms**. Each term is a **spatial map**, showing where it is present, times a
**signature**, describing what it puts in each pixel.

```{figure} images/bbtd-board-terms.png
:alt: Five spatial maps (the background and the letters B, B, T, D) above five polarization ellipses

A polarization image of the letters B B T D, split by pyBBTD into five terms: each letter
and the background, with its polarization drawn below it. See the
[Quickstart](getting-started/quickstart.ipynb).
```

## Where to start

::::{grid} 1 2 2 2
:gutter: 3

:::{grid-item-card} New to tensors?
:link: getting-started/tensors
:link-type: doc

What a tensor is and what decomposing one means, with a real image and no prior knowledge.
:::

:::{grid-item-card} Quickstart
:link: getting-started/quickstart
:link-type: doc

Install pyBBTD and decompose an image in a few lines.
:::

:::{grid-item-card} Choosing a model
:link: models/index
:link-type: doc

Which of the four models matches your data.
:::

:::{grid-item-card} Tutorials
:link: tutorials/btd
:link-type: doc

Step-by-step examples for every model, with real outputs.
:::
::::

## Install

```bash
pip install git+https://github.com/scbarreto/pyBBTD
```

## The models

| Model | Data at each pixel | Tensor shape |
|---|---|---|
| [BTD](models/btd.md) | several numbers (channels) | (I, J, K) |
| [Cov-LL1](models/covll1.md) | a covariance matrix, stored as a vector | (I, J, K²) |
| [Stokes](models/stokes.md) | a Stokes vector (polarization) | (I, J, 4) |
| [BBTD](models/bbtd.md) | a matrix, or a covariance matrix | (I, J, K, M) |

```{toctree}
:hidden:
:caption: Getting started

getting-started/tensors
getting-started/installation
getting-started/quickstart
```

```{toctree}
:hidden:
:caption: Models

models/index
models/btd
models/covll1
models/stokes
models/bbtd
```

```{toctree}
:hidden:
:caption: Tutorials

tutorials/btd
tutorials/covll1
tutorials/stokes-synthetic
tutorials/stokes-bbtd-board
tutorials/bbtd-als
tutorials/bbtd-admm
```

```{toctree}
:hidden:
:caption: Guides

guides/ranks-and-uniqueness
guides/fitting
```

```{toctree}
:hidden:
:caption: Reference

api/index
glossary
references
```

```{toctree}
:hidden:
:caption: Development

contributing
```
