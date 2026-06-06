""" Shorthand to handle Aidge models """

from . import _aidge_tk, _aidge_io
from ._aidge_tk import *
from ._aidge_io import *

__all__ = _aidge_tk.__all__.copy()
__all__ += _aidge_io.__all__.copy()
