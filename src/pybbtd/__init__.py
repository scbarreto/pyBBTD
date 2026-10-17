"""pyBBTD: block term tensor decompositions for imaging data.

The four models are available at the top level::

    from pybbtd import BTD, BBTD, CovLL1, Stokes
"""

from importlib.metadata import PackageNotFoundError, version

from pybbtd.bbtd import BBTD
from pybbtd.btd import BTD
from pybbtd.covll1 import CovLL1
from pybbtd.stokes import Stokes
from pybbtd.uniqueness import check_uniqueness_bbtd, check_uniqueness_ll1

try:
    __version__ = version("pybbtd")
except PackageNotFoundError:  # running from a source tree that is not installed
    __version__ = "unknown"

__all__ = [
    "BBTD",
    "BTD",
    "CovLL1",
    "Stokes",
    "__version__",
    "check_uniqueness_bbtd",
    "check_uniqueness_ll1",
]
