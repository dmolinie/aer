""" Graphical User Interface """

from . import _funcanimation, _gui_tk
from ._funcanimation import *
from ._gui_tk import *

__all__ = _funcanimation.__all__.copy()
__all__ += _gui_tk.__all__.copy()
