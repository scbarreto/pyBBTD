# pyBBTD

A python package for computing Block-block terms tensor decompositions.
Check the [documentation and examples](https://scbarreto.github.io/pyBBTD/)

## basic instructions

Packaging uses `uv` : [see here for detailed installation instructions](https://docs.astral.sh/uv/#installation)

## after cloning repo

```
curl -Ls https://astral.sh/uv/install.sh | sh
uv venv
uv sync --all-extras
```

## before commit + push

```
uv run ruff check
uv run ruff format
```

## testing

```
uv run pytest
```

## running coverage locally

pytest --cov=pybbtd --cov-report=html tests/
