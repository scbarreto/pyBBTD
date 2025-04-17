# pyBBTD

A python package for computing Block-block terms tensor decompositions.

## basic instructions

Packaging uses `uv` : [see here for detailed installation instructions](https://docs.astral.sh/uv/#installation)

## after cloning repo

```
curl -Ls https://astral.sh/uv/install.sh | sh
uv venv
source .venv/bin/activate
uv pip install ipykernel
uv pip install -e .
```

## before commit + push

```
uv run ruff check
uv run ruff format
```

## testing

```
uv run pytest
uv run ruff format
```

pytest --cov=pybbtd --cov-report=html tests/
