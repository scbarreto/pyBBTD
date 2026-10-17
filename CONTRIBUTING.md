# Contributing

## Setup

The project uses [uv](https://docs.astral.sh/uv/) for packaging
([installation instructions](https://docs.astral.sh/uv/#installation)). After cloning the
repository:

```bash
curl -LsSf https://astral.sh/uv/install.sh | sh  # if uv is not installed yet
uv sync --all-extras
uv run pre-commit install  # optional: lint and format on every commit
```

## Before committing

```bash
uv run ruff check
uv run ruff format
```

## Tests

```bash
uv run pytest
```

Warnings are turned into errors, so tests that expect a warning declare it with
`pytest.warns` or a `filterwarnings` mark. Coverage report:

```bash
uv run pytest --cov=pybbtd --cov-report=html
```

The report is written to `htmlcov/index.html`.

## Documentation

The site is built with Sphinx from `docs/`. Pages are Markdown (MyST), and the getting
started pages and tutorials are Jupyter notebooks shown with their saved outputs.

```bash
uv run --extra docs sphinx-build -W --keep-going -b html docs docs/_build/html
```

After changing a notebook or the code it uses, run the notebooks again so the saved
outputs come from a real execution:

```bash
uv run --extra notebooks python scripts/run_notebooks.py           # run and save all
uv run --extra notebooks python scripts/run_notebooks.py --check   # run without saving
```

The figure at the top of `README.md` and of the documentation home page,
`docs/images/bbtd-board-terms.png`, is the output of the Quickstart notebook.

## Continuous integration

GitHub Actions has two workflows:

- **Tests** (`.github/workflows/tests.yml`): the linter, the tests on Python 3.11 to 3.13
  and a run of the documentation notebooks. It runs only when code, tests or notebooks
  change.
- **Docs** (`.github/workflows/docs.yml`): builds the site on pull requests and, after a
  merge to `main`, publishes it to GitHub Pages without waiting for the tests.
