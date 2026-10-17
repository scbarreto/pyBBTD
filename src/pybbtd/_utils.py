"""Input checks and warning helpers shared by the models and solvers."""

import os
import sys
import warnings
from numbers import Integral

import numpy as np

_ORDER_NAMES = {2: "two", 3: "three", 4: "four"}
_PACKAGE_DIR = os.path.dirname(os.path.abspath(__file__)) + os.sep


def check_positive_int(value, name):
    """Return ``value`` as an ``int``, or raise if it is not a positive integer."""
    if isinstance(value, bool) or not isinstance(value, Integral) or value <= 0:
        raise ValueError(f"{name} should be a positive integer.")
    return int(value)


def check_dims(dims, order):
    """Return ``dims`` as a tuple of ``order`` positive integers, or raise."""
    if (
        not isinstance(dims, (list, tuple))
        or len(dims) != order
        or not all(
            isinstance(d, Integral) and not isinstance(d, bool) and d > 0 for d in dims
        )
    ):
        raise ValueError(
            f"dims should be a list or tuple of {_ORDER_NAMES[order]} positive integers."
        )
    return tuple(int(d) for d in dims)


def check_solver_inputs(model, model_class, T):
    """Check that a solver received a model of the right class and a matching tensor."""
    if not isinstance(model, model_class):
        raise TypeError(
            f"model must be an instance of the {model_class.__name__} class."
        )
    if not isinstance(T, np.ndarray):
        raise TypeError("T must be a numpy array.")
    if T.shape != model.dims:
        raise ValueError(
            f"T's dimensions ({T.shape}) do not match model.dims ({model.dims})."
        )


def unknown_option_error(kind, value, options):
    """Build the error raised when ``value`` is not one of ``options``."""
    choices = ", ".join(f"'{option}'" for option in options)
    return ValueError(f"Unknown {kind} '{value}'. Choose from: {choices}.")


def warn_user(message, category=UserWarning):
    """Issue a warning that points at the first line of code outside pybbtd."""
    frame = sys._getframe(1)
    stacklevel = 2
    while frame is not None and frame.f_code.co_filename.startswith(_PACKAGE_DIR):
        frame = frame.f_back
        stacklevel += 1
    warnings.warn(message, category, stacklevel=stacklevel)
