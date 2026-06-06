""" Audio Event Recognition (AER) Project """

__version__ = '1.0'

from importlib.util import find_spec as _find_spec
from importlib import import_module as _import_module


#__submodules__ = {
#    'tools', 'spectrum', 'filters', 'data_tk', 'accuracy',
#    'layers', 'models', 'models_tk', 'onnx', 'datasets',
#    'aidge', 'plot', 'display', 'gui'
#    }
#__all__ = list(__submodules__)


# Modules that require only Python standard libraries
__submodules__ = ['tools']

# Modules that require third-part libraries that are listed as
# "dependencies" in the '.toml' installation file
__submodules__ += [
    'accuracy',     # numpy
    'datasets',     # numpy & scipy
    'data_tk',      # numpy
    'filters',      # numpy
    'plot',         # numpy & matplotlib
    'display',      # numpy & matplotlib
    'spectrum'      # numpy
]

# Modules that require additional third-part libraries that are listed
# as optional dependencies ("extras") in the '.toml' installation file
__opt_submodules__ = {
#    # Base submodules
#    'accuracy': ['numpy'],
#    'datasets': ['numpy', 'scipy'],
#    'data_tk': ['numpy'],
#    'filters': ['numpy'],
#    'plot': ['numpy', 'matplotlib'],
#    'spectrum': ['numpy'],
    # Advanced submodules
    # (consider that `numpy` & `matplotlib` are installed)
    'layers': ['torch'],
    'models': ['torch'],
    'onnx': ['onnxruntime'],
    'models_tk': ['torch'],
    'gui': ['sounddevice', 'torch'],
    'aidge': ['requests', 'aidge_core', 'aidge_onnx',
              'aidge_backend_cpu', 'aidge_export_cpp'],
}

# For any optional submodule, check if its dependencies are installed;
# if so, add this submodule to the list of the available submodules
for _module in __opt_submodules__.items():
    if all(_find_spec(_mod) is not None for _mod in _module[1]):
        __submodules__.append(_module[0])

# Add the installed submodules to the `__all__` set
__all__ = __submodules__


def __getattr__(attr):
    """ Return the correct module from its name, if it exists """
    # pylint: disable=C0415, R0402, R0911, R0912
#    # Short version based on importlib (a little less efficient)
#    if name in __submodules__:
#        return _import_module(f"aer.{name}")
    # Submodules
    if attr == 'tools':
        import aer.tools as tools
        return tools
    if attr == 'spectrum':
        import aer.spectrum as spectrum
        return spectrum
    if attr == 'filters':
        import aer.filters as filters
        return filters
    if attr == 'data_tk':
        import aer.data_tk as data_tk
        return data_tk
    if attr == 'accuracy':
        import aer.accuracy as accuracy
        return accuracy
    if attr == 'layers':
        import aer.layers as layers
        return layers
    if attr == 'models':
        import aer.models as models
        return models
    if attr == 'models_tk':
        import aer.models_tk as models_tk
        return models_tk
    if attr == 'onnx':
        import aer.onnx as onnx
        return onnx
    # Load and handle datasets
    if attr == 'datasets':
        import aer.datasets as datasets
        return datasets
    # Aidge functions
    if attr == 'aidge':
        import aer.aidge as aidge
        return aidge
    # Plotting functions and Graphical User Interface
    if attr == 'plot':
        import aer.plot as plot
        return plot
    if attr == 'display':
        import aer.display as display
        return display
    if attr == 'gui':
        import aer.gui as gui
        return gui
    raise AttributeError(f"module {__name__!r} has no attribute {attr!r}")

def __dir__():
    """ Add the modules of `submodules` to the list of callable variables"""
    public_symbols = globals().keys() | __submodules__
    return list(public_symbols)
