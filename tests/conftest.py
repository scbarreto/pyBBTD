import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pytest  # noqa: E402


@pytest.fixture(autouse=True)
def seed_global_rng():
    """Seed NumPy's global RNG, which the package draws from, before each test.

    Every test is then reproducible on its own, whatever order tests run in.
    """
    np.random.seed(12345)


@pytest.fixture(autouse=True)
def no_figure_windows(monkeypatch):
    """Make plt.show() a no-op and close the figures after each test."""
    monkeypatch.setattr(plt, "show", lambda: None)
    yield
    plt.close("all")
