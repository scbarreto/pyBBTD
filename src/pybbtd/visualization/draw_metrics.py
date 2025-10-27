import matplotlib.pyplot as plt
import numpy as np
from typing import Union, Sequence


def plot_error(
    errors: Union[Sequence[float], Sequence[Sequence[float]]],
    labels: Union[str, Sequence[str]] = None,
    xlabel: str = r"Iteration Number",
    ylabel: str = r"Fit Error",
    title: str = None,
    semilogy: bool = True,
    grid: bool = True,
):
    """
    Plots one or multiple convergence/error curves

    Parameters
    ----------
    errors : array-like or list of array-like
        One or multiple sequences of error values.
    labels : str or list of str, optional
        Label(s) for the plotted curves.
    xlabel, ylabel : str, optional
        Axis labels (LaTeX supported).
    title : str, optional
        Figure title.
    semilogy : bool, optional
        If True, plot errors on a log scale.
    grid : bool, optional
        If True, display gridlines.
    """

    fig, ax = plt.subplots(figsize=(6, 4))

    # Handle single or multiple error curves
    if isinstance(errors[0], (list, np.ndarray)):
        for i, err in enumerate(errors):
            label = labels[i] if labels and i < len(labels) else f"Curve {i + 1}"
            if semilogy:
                ax.semilogy(err, label=label)
            else:
                ax.plot(err, label=label)
    else:
        if semilogy:
            ax.semilogy(errors, label=labels)
        else:
            ax.plot(errors, label=labels)

    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    if title:
        ax.set_title(title)

    if labels:
        ax.legend()
    if grid:
        ax.grid(True, which="both", linestyle="--", linewidth=0.5, alpha=0.7)

    plt.tight_layout()
    plt.show()
