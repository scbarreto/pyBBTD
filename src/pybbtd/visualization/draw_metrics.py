"""Convergence plots."""

from collections.abc import Sequence

import matplotlib.pyplot as plt
import numpy as np


def plot_error(
    errors: Sequence[float] | Sequence[Sequence[float]],
    labels: str | Sequence[str] | None = None,
    xlabel: str = "Iteration Number",
    ylabel: str = "Fit Error",
    title: str | None = None,
    semilogy: bool = True,
    grid: bool = True,
):
    """Plot one or several fit-error curves.

    Parameters
    ----------
    errors : array-like or list of array-like
        One curve (e.g. ``model.fit_error``) or a list of curves.
    labels : str or list of str, optional
        Label of each curve.
    xlabel, ylabel : str, optional
        Axis labels (LaTeX supported).
    title : str, optional
        Figure title.
    semilogy : bool, default True
        Plot the errors on a logarithmic scale.
    grid : bool, default True
        Show grid lines.
    """
    fig, ax = plt.subplots(figsize=(6, 4))
    plot = ax.semilogy if semilogy else ax.plot

    if isinstance(errors[0], (list, np.ndarray)):
        for i, err in enumerate(errors):
            label = labels[i] if labels and i < len(labels) else f"Curve {i + 1}"
            plot(err, label=label)
    else:
        plot(errors, label=labels)

    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    if title:
        ax.set_title(title)
    if labels:
        ax.legend()
    if grid:
        ax.grid(True, which="both", linestyle="--", linewidth=0.5, alpha=0.7)

    fig.tight_layout()
    plt.show()
