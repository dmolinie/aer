""" Simple frequency-domain filters

Authors: Dylan MOLINIE
Company: CEA Paris-Saclay, Gif-sur-Yvette, France
Contact: dylan.molinie@cea.fr
Date: February 2025
Last revised: March 2026

License: GPLv3
"""

__all__ = [
    'extrema', 'extrema_nd', 'rescale', 'sub_bands',
    'hzscale', 'melscale', 'barkscale',
    'hz2mel', 'mel2hz', 'hz2bark', 'bark2hz']

import numpy as np


##############################################################################
##                         Data Limits & Rescaling                          ##
##############################################################################

#-------------------------   Data Extremal Values   -------------------------#
def _extrema(data, axis=None):
    """ Return the min & max of 'data' along the given axis """
    return (np.amin(data, axis), np.amax(data, axis))

def extrema(data, axis=None):
    """ Return the min and max values within the data

    Parameters
    ----------
    data : array_like
        Data of extremal values to find.
    [OPT] axis : int (or None)
        Axis along which to compute the extremal values.
            :Default: None (all components' min and max).

    Returns
    -------
    mini, maxi : 2-tuple of floats or of np.ndarrays
        The min and max values within the data.

    Examples
    --------
    >>> import numpy as np

    >>> extrema(np.linspace(-5, +5, 5))
    (-5.0, 5.0)
    """
    if data.dtype == complex:                       # Complex values
        ext_real = _extrema(data.real, axis)
        ext_imag = _extrema(data.imag, axis)
        return (ext_real[0] + 1j*ext_imag[0], ext_real[1] + 1j*ext_imag[1])
    return _extrema(data.real, axis)
#----------------------------------------------------------------------------#

#---------------------   Min & Max of a Set of Arrays   ---------------------#
def _min_cplx(val1, val2):
    """ Retrieve the minimum real & imag parts between two complex values
    For two complex values `val1` and `val2`, find the minimum `mr` btw
    their real parts and the minimum `mj` btw their imaginary parts, and
    return the complex z with `mr` as real part and `mj` as imag part.
    If the values are real, directly return the minimum between them. """
    # If the values are real, return the min between them
    if val1.imag == val2.imag == 0:
        return min(val1, val2)
    # If the values are complex, return the min between their real parts,
    # and the min between their imaginary parts
    return min(val1.real, val2.real) + 1j*min(val1.imag, val2.imag)

def _max_cplx(val1, val2):
    """ Retrieve the maximum real & imag parts between two complex values
    For two complex values `val1` and `val2`, find the maximum `Mr` btw
    their real parts and the maximum `Mj` btw their imaginary parts, and
    return the complex z with `Mr` as real part and `Mj` as imag part.
    If the values are real, directly return the maximum between them. """
    # If the values are real, return the max between them
    if val1.imag == val2.imag == 0:
        return max(val1, val2)
    # If the values are complex, return the max between their real parts,
    # and the max between their imaginary parts
    return max(val1.real, val2.real) + 1j*max(val1.imag, val2.imag)

def extrema_nd(data):
    """ Get the min & max values among several arrays

    If `data` is an np.ndarray, directly return its min and max values;
    if it a list of such arrays, compute these two quantities for any
    array and return the true min & max among all.

    If `data` is complex-valued, find the min and max for both the real
    and imaginary parts, and return two complex values, the first whose
    real and imaginary parts are the minimum among that of any array of
    `data`, and the second one for the maximum values.

    It is somehow a two-step search: first, find the min and max among
    all the real parts of any array of `data`; second, do the same for
    all their imaginary parts. Finally, return the complex formed by the
    min real & imag, and the complex formed by the max real & imag.

    If `data` is real-valued, do so only for the real part, and return
    the real-valued minimum and maximum among all the arrays, if any.

    Parameters
    ----------
    data : np.ndarray, or list of np.ndarrays
        The data for with to find the min & max values; can be either
        real- or complex-valued.

    Returns
    -------
    min, max : same type as the values of `data`
        The minimum and maximum values among all the arrays of `data`.
        If `data` is real-valued, min & max also are real values; else,
        they are two complexes, whose real part is the min/max value
        among all the real parts of the values of `data`, and also the
        same for the imaginary parts.

    Examples
    --------
    >>> import numpy as np

    # Generate semi-random data
    >>> rng = np.random.default_rng(seed=0)
    >>> data = np.arange(1., 11.) + 1j*np.arange(1., 11.)
    >>> rng.shuffle(data.real)
    >>> rng.shuffle(data.imag)

    # Min & max of a unique data array
    >>> extrema_nd(data)
    ((1+1j), (10+10j))

    # Min & max between several data arrays
    >>> extrema_nd([data, data+5j, data-3.])
    ((-2+1j), (10+15j))
    """
    # If tuple of arrays, search for the true min & max among all arrays
    if isinstance(data, (tuple, list)):
        lims = extrema(data[0])                 # Last seen min & max
        # Compare the initial limits to that of any other array
        for arr in data[1:]:
            # If `arr` is itself a tuple of arrays, call the func. recursively
            if isinstance(arr, (tuple, list)):
                lcur = extrema_nd(arr)          # Recursive call
            else:
                lcur = extrema(arr)             # Standard call
            # Update the min & max values
            lims = (_min_cplx(lims[0], lcur[0]), _max_cplx(lims[1], lcur[1]))
        return lims
    # Else, simply return the min & max of the array
    return extrema(data)
#----------------------------------------------------------------------------#

#------------------------   Database Normalization   ------------------------#
def rescale(data, limits=None, bounds=(0., 1.), axis=None, dtype=float):
    r""" Data rescaling between `bounds`

    Take the data to rescale (e.g. to normalize), its original `limits`
    (i.e. the lower and upper bounds of the interval within which the
    data in their current scale exist) and the `bounds` of the interval
    within which to rescale the data. This can be formalized as:
        data \in [limits[0]; limits[1]]
    which are linearly rescaled to exist inside [bounds[0]; bounds[1]].

    The rescaling is linear, and is expressed as:
        data_norm = (M2-m2) * (data-m1) / (M1-m1) + m2
    where (m1, M1) = `limits` and (m2, M2) = `bounds`, `m` referring to
    the lower bounds, and `M` to the upper bounds.

    If `limits` is not provided, it is the min and max along the speci-
    fied `axis` which are used as lower and upper limits; if `bounds` is
    not provided, the data are rescaled between 0 and 1.

    N.B: any operations are performed using Python native `float` type;
        then, the `limits`, `bounds` and (rescaled) `data` arrays are
        cast into `dtype` type copied arrays and returned.

    N.B.: this function is designed to function with 1D data only.

    Parameters
    ----------
    data : np.ndarray
        The data to rescale.
    [OPT] limits : 2-tuple
        The lower & upper bounds of the interval within which the data
        actually live; if not provided, use the min & max of the data.
        Essentially aims to save computation if already computed prior.
            :Default: None (get the data min and max)
    [OPT] bounds : 2-tuple
        The lower & upper bounds of the interval into which to rescale
        the data, i.e. the new min & max; if not provided, use 0 and 1.
            :Default: (0., 1.)
    [OPT] axis : int
        The axis along which to get the min & max of the data; used only
        if `limits` is not provided.
            :Default: -1
    [OPT] dtype : data type
        The data type to use to format the different arrays.
            :Default: float

    Returns
    -------
    limits : 2-tuple
        The min and max values within the data. If `limits` was provided
        as argument, return it directly; else, return the min and max of
        `data` along `axis`.
    bounds : 2-tuple
        The bounds of the interval within which the data are rescaled.
    data : np.ndarray
        The rescaled data.

    Examples
    --------
    >>> import numpy as np

    >>> data = np.random.random((10, 2))
    >>> lims, bnds, data_norm = rescale(data, (0., 1.), (-1., 1.))
    """

    # TODO Return `data` first?
    # TODO Reverse `limits` and `bounds` (`bounds` first)?

    # New bounds for the rescaled data
    bounds = np.array(bounds, float)

    # If not provided, get the data extremal values
    if limits is None:
        limits = np.array(extrema(data, axis), float)
    else:
        limits = np.array(limits, float)

    # If the limits & bounds are the same, do nothing
    if np.all(limits == bounds):
        return limits, bounds, data

    # Operate the normalization between the upper and lower bounds
    coef = (bounds[1] - bounds[0]) / (limits[1] - limits[0])
    data = coef*(data-limits[0]) + bounds[0]

    return limits.astype(dtype), bounds.astype(dtype), data.astype(dtype)
#----------------------------------------------------------------------------#

##############################################################################



##############################################################################
##                          Representation Scales                           ##
##############################################################################

#----------------------   Generate Frequency Scales   -----------------------#
def hzscale(min_freq, max_freq, samples):
    """ Build a set of `samples` linearly spaced samples in the Hertz
    scale spanning from `min_freq` to `max_freq` Hz frequencies """
    return np.linspace(min_freq, max_freq, samples)

def melscale(min_freq, max_freq, samples):
    """ Build a set of `samples` linearly spaced samples in the Mel
    scale spanning from `min_freq` to `max_freq` Hz frequencies """
    return np.linspace(hz2mel(min_freq), hz2mel(max_freq), samples)

def barkscale(min_freq, max_freq, samples):
    """ Build a set of `samples` linearly spaced samples in the Bark
    scale spanning from `min_freq` to `max_freq` Hz frequencies """
    return np.linspace(hz2bark(min_freq), hz2bark(max_freq), samples)
#----------------------------------------------------------------------------#

#-------------------------   Melody-Hertz Scales   --------------------------#
def hz2mel(freqs):
    """ Convert Hertz frequency into Mel frequency

    For a Hz frequency f, its Mel frequency m is defined as:
        m = 1127*ln(1+f/700) = 2595*log10(1+f/700)

    Note: the np.log (ln) function is faster than the np.log10 one,
          whence the np.log-based implementation here.

    Parameters
    ----------
    freqs : scalar or np.ndarray
        The Hertz frequencies.

    Returns
    -------
    mels : same as the `freqs` parameter
        The frequencies rescaled to the Melody scale.

    Examples
    --------
    >>> import numpy as np

    >>> hz2mel(np.linspace(0., 100., 101))
    array([  0.        ,   1.60885109,   ..., 149.08024828, 150.48987949])
    """
    if isinstance(freqs, (list, tuple)):
        freqs = np.array(freqs)
    return 1127. * np.log(1. + freqs/700.)

def mel2hz(mels):
    """ Convert Mel frequency into Hertz frequency

    For a Mel frequency m, its Hertz frequency f is defined as:
        f = 700*(exp(m/1127)-1) = 700*(10^(m/2595)-1)

    Note: the np.exp-based variant is around 1.5-2 times faster than
          the np.pow-based variant, whence the np.exp implementation.

    Parameters
    ----------
    mels : scalar or np.ndarray
        The Mel frequencies.

    Returns
    -------
    freqs : same as the `mels` parameter
        The frequencies rescaled to the Hertz scale.

    Examples
    --------
    >>> import numpy as np

    >>> mel2hz(np.linspace(0., 100., 101))
    array([ 0.        ,  0.62139366,  ..., 64.27232213, 64.95077066])
    """
    if isinstance(mels, (list, tuple)):
        mels = np.array(mels)
    return 700. * (np.exp(mels/1127.) - 1.)
#----------------------------------------------------------------------------#

#---------------------------   Bark-Hertz Scales  ---------------------------#
def hz2bark(freqs):
    """ Convert Hertz frequency into Bark frequency

    Wrt the Traunmüller's definition (the simplest Bark scale analytic
    expression), the Bark frequency b of a Hz frequency f is given by:
        b = 26.81 * f / (1960 + f) - 0.53

    Note: the above implementation is upper-bounded by 26.28.

    Parameters
    ----------
    freqs : scalar or np.ndarray
        The Hertz frequencies.

    Returns
    -------
    barks : same as the `freqs` parameter
        The frequencies rescaled to the Bark scale.

    Examples
    --------
    >>> import numpy as np

    >>> hz2bark(np.linspace(0., 100., 101))
    array([-0.53      , -0.5163284 , ...,  0.75906751, 0.77145631])
    """
    if isinstance(freqs, (list, tuple)):
        freqs = np.array(freqs)
    return 26.81 * freqs / (1960. + freqs) - 0.53

def bark2hz(barks):
    """ Convert Bark frequency into Hertz frequency

    For a Bark frequency b, its Hertz frequency f is defined as:
        f = 1960 * alpha / (26.81 - alpha), with alpha = b + 0.53

    Note: with the implementation of the `hz2bark` function, a bark
          frequency should never be equal nor greater than 26.28.

    Parameters
    ----------
    mels : scalar or np.ndarray
        The Bark frequencies.

    Returns
    -------
    freqs : same as the `barks` parameter
        The frequencies rescaled to the Hertz scale.

    Examples
    --------
    >>> import numpy as np

    >>> bark2hz(np.linspace(0., 100., 101))
    array([ 3.95281583e+01,  1.18623418e+02,  ..., -2.67279978e+03])
    """
    if isinstance(barks, (list, tuple)):
        barks = np.array(barks)
    return (1038.8 + 1960. * barks) / (26.28 - barks)
#----------------------------------------------------------------------------#

##############################################################################



##############################################################################
##                             Signal Sub-Bands                             ##
##############################################################################

#---------------------------   Signal Sub-Bands   ---------------------------#
def _sub_bands(bandwidth, nb_bands=10, overlap=True):
    """ Split a bandwidth interval into several sub-bands

    Take the bandwidth interval and build `nb_bands` sub-bands from it.
    Make these sub-bands overlap in their center if `overlap` is True,
    or just stack these sub-bands side-by-side else.

    Parameters
    ----------
    bandwidth : 2-float array_like
        The overall bandwidth, providing the two extremal frequencies,
        supposed to be organized as: bandwidth = [fmin, fmax].
    [OPT] nb_bands : int
        The number of sub-bands to extract from the total bandwidth.
            :Default: 10
    [OPT] overlap : bool
        If the sub-bands should overlap (in their center) or not.
            :Default: True

    Returns
    -------
    bands : np.ndarray of shape (nb_bands, 2)
        The linearly spaced sub-bands.

    Examples
    --------
    >>> _sub_bands([0, 60], 3, True)
    [(0.0, 30.0), (15.0, 45.0), (30.0, 60.0)]

    >>> _sub_bands([0, 60], 3, False)
    [(0.0, 20.0), (20.0, 40.0), (40.0, 60.0)]
    """

    # Displacing offset
    off = 1 if not overlap else 2

    # Compute the bands bounds
    bands = np.linspace(*bandwidth, nb_bands+off, dtype=float)

    # Build the sub-bands (every pair of two values separated by `off`)
    return np.array([(bands[i], bands[i+off]) for i in range(len(bands)-off)])
#----------------------------------------------------------------------------#

#---------------------------   Signal Sub-Bands   ---------------------------#
def sub_bands(bandwidth, nb_bands=10, overlap=True, scale='hz'):
    """ Split a bandwidth interval into sub-bands on a given scale

    Take a scale (Hertz, Melody or Bark), project the bandwidth bounding
    frequencies onto this scale, pass this so-projected bandwidth to the
    `_sub_bands` function, and finally project the bandwidth back to the
    Hertz scale. Thus, if `scale` it not `Hertz`, the frequencies are
    linearly spaced in `scale`, but not in the Hertz scale.
    See `_sub_bands` function for details.

    Parameters
    ----------
    bandwidth : 2-float array_like
        The overall bandwidth, providing the two extremal frequencies.
    [OPT] nb_bands : int
        The number of sub-bands to extract from the total bandwidth.
            :Default: 10
    [OPT] overlap : bool
        If the sub-bands should overlap (in their center) or not.
            :Default: True
    [OPT] scale : string (case insensitive)
        The scale to use; valid options are: Melody ('mel' or 'melody'),
        Bark ('bk' or 'bark') and Hertz (anything else).
            :Default: 'hz' (Hertz scale)

    Returns
    -------
    bands : np.ndarray of shape (nb_bands, 2)
        The sub-bands linearly spaced over `scale`.

    Examples
    --------
    >>> sub_bands((0, 12), 3, True, 'hz')
    array([[ 0.,  6.], [ 3.,  9.], [ 6., 12.]])

    >>> sub_bands((0, 12), 3, True, 'mel')
    array([[ 0.   ,  5.975], [ 2.981,  8.981], [ 5.975, 12.   ]])

    >>> sub_bands((0, 12), 3, True, 'bark').round(3)
    array([[ 0.   ,  5.982], [ 2.986,  8.986], [ 5.982, 12.   ]])
    """
    if scale.lower() in ('mel', 'melody'):              # Mel scale
        return mel2hz(_sub_bands(hz2mel(bandwidth), nb_bands, overlap))
    if scale.lower() in ('bk', 'bark'):                 # Bark scale
        return bark2hz(_sub_bands(hz2bark(bandwidth), nb_bands, overlap))
    return _sub_bands(bandwidth, nb_bands, overlap)     # Hertz scale
#----------------------------------------------------------------------------#

##############################################################################
