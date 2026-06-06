""" Shorthand Plotting functions """

from . import _decorations, _core_plot, _plot
from ._decorations import *
from ._core_plot import *
from ._plot import *

__all__ = _decorations.__all__.copy()
__all__ += _core_plot.__all__.copy()
__all__ += _plot.__all__.copy()
