""" Audio (1D) signal filtering tools """
#pylint: disable=import-outside-toplevel, consider-using-from-import

__submodules__ = {'mivia_loader'}
__all__ = list(__submodules__)

def __getattr__(attr):
    """ Return the correct module from its name, if it exists """
    if attr == 'mivia_loader':
        import aer.datasets.mivia_loader as mivia_loader
        return mivia_loader
    raise AttributeError(f"module {__name__!r} has no attribute {attr!r}")

def __dir__():
    """ Add the modules of 'submodules' to the list of callable variables"""
    public_symbols = globals().keys() | __submodules__
    return list(public_symbols)
