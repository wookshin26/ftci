"""Function tensor cross interpolation and portable tensor-network functions."""

from .basis import Basis
from .model import Model
from .engine import FunctionProblem, fit

__version__ = "0.1.0"
__all__ = ["Basis", "Model", "FunctionProblem", "fit"]
