""" Simple time- & frequency-domain filters

Authors: Dylan MOLINIE
Company: CEA Paris-Saclay, Gif-sur-Yvette, France
Contact: dylan.molinie@cea.fr
Date: January 2025
Last revised: March 2026

License: GPLv3
"""

__all__ = [
    'flip', 'convolve',
    'rectangular_time', 'triangular_time', 'gammatone_time',
    'rectangular_freq', 'triangular_freq']

import numpy as np


##############################################################################
##                             Filtering Tools                              ##
##############################################################################

#--------------------------   Mirror Data Vector   --------------------------#
def flip(vals, center=None, even=True):
    """ Mirror a vector and stack the mirrored and original data

    Take a vector `vals`, mirror it (flip) and stack both side-by-side.
    Place the mirrored vector first, such that [mirror, vals]. If `even`
    is False, consider the mirror to be anti-symmetric (f(-x) = -f(x)),
    and use `-mirror` instead of `mirror` (i.e. inverse sign of `vals`).
    If `center` is a scalar or an array of scalars, add it in-between
    the mirrored and original data, such that [mirror, center, vals];
    if None, add no value in-between, such that [mirror, vals].

    Parameters
    ----------
    vals : array_like
        The data to be mirrored (flip).
    [OPT] center : scalar, array of scalars, None
        The value(s) to add in-between the mirrored and original data.
            :Default: None (add no data)
    [OPT] even : bool
        If `True`, consider the function is even, and do not change the
        sign of `vals`; change their sign (`-mirror`) else.
            :Default: True (no sign change)

    Returns
    -------
    mirror : array_like
        The mirrored and original data stacked together.

    Examples
    --------
    >>> flip([1, 2, 3])
    array([3, 2, 1, 1, 2, 3])

    >>> flip([1, 2, 3], 0)
    array([3, 2, 1, 0, 1, 2, 3])

    >>> flip([2, 3], [-1, 0, +1], False)
    array([-3, -2, -1,  0,  1,  2,  3])
    """

    # Mirror the data
    mirror = np.flip(vals, 0) if even else -np.flip(vals, 0)

    # Concatenate the mirrored and original data
    if center is None:
        return np.hstack([mirror, vals])            # Do not center data
    return np.hstack([mirror, center, vals])        # Center the data
#----------------------------------------------------------------------------#

#------------------   Time-Domain Convolution Filtering   -------------------#
def convolve(data, filt):
    """ Operate the convolution two vectors

    Use the `np.convolve` function to operate the convolution between
    the two input vectors, which are assumed to be the same length N.
    Use the `full` convolution mode, which results in a 2N-long vector;
    return only the second half of the convolution vector, and divide
    its components by N/2 to normalize it.

    Parameters
    ----------
    data : np.ndarray
        The first vector, for instance the signal data samples.
    filt : np.ndarray
        The second vector, for instance the time response of a filter.

    Returns
    -------
    conv : np.ndarray
        The convolution between the two input vector data.

    Examples
    --------
    >>> import numpy as np

    >>> conv = convolve(np.linspace(-5, +5, 10), np.linspace(-1, +10, 10))
    """
    beg, end = abs(len(filt) - len(data)), max(len(data), len(filt))
    return (2./len(filt)) * np.convolve(data, filt, mode='full')[beg:end]
#----------------------------------------------------------------------------#

##############################################################################



##############################################################################
##                           Time-Domain Filters                            ##
##############################################################################

#--------------------   Time-Domain Rectangular Filter   --------------------#
def _gate(tstp, fmin, fmax):
    """ Rectangle-shape filter core -- See `rectangular_time` """
    # Compute the sinc filters
    # 2pi*f * sinc(2pi*ft) = sin(2pi*ft)/t
    return (np.sin(fmax*tstp) - np.sin(fmin*tstp)) / ((fmax-fmin) * tstp)
#    # Do not add pi in sinc, since np.sinc(x) = np.sin(np.pi*x)/(np.pi*x)
#    return (fmax*np.sinc(2.*fmax*tstp) - fmin*np.sinc(2.*fmin*tstp)) / (fmax-fmin)

def rectangular_time(tstp, bandwidth, sym=True, center=None, rescale=True):
    """ Rectangular-shaped filter time impulse response

    Take a set of timestamps and the gate filter bandwidth (lower and
    upper cutoff frequencies), and compute the temporal response of it,
    defined as the difference between two sinc functions:
        h(t) = 2pi*(fmax*sinc(2pi*fmax*tstp) - fmin*sinc(2pi*fmin*tstp))
             = (sin(2pi*fmax*tstp) - sin(2pi*fmin*tstp)) / tstp
    The filter response is finally normalized by dividing it by its max,
    i.e. h'(t) = h(t) / h(0) = h(t) / (2pi * (fmax - fmin))

    If `sym`, consider the timestamps `tstp` as only half of the inputs,
    and, since sinc is an even function, mirror the filter response to
    these inputs backwards; stack the two vectors as [mirror, outputs].
    If `center` is a number, add this value in-between the two vectors;
    if `None`, just stack both side-by-side.

    If `rescale`, assuming that `tstp` is organized in ascending order,
    make them start from 0 by subtracting the first stamp to any other.

    The vector returned aims to be convolved with the temporal represen-
    tation of the signal (see `convolve` function).

    Parameters
    ----------
    tstp : np.ndarray
        The timestamps to filter with the gate filter.
    bandwidth : 2-tuple of ints or floats
        The filter lower and upper cutoff frequencies as (flow, fupp).
    [OPT] sym : bool
        If True, mirror backwards the filter response to `tstp`, and
        stack both vectors as [mirror, response].
            :Default: True
    [OPT] center : int, float or None
        If not None, the scalar value to add in-between the mirror and
        the response; only used if `sym` is True; if None, no value is
        added. Ensures the response parity (odd or even cardinal).
            :Default: None (no value added)
    [OPT] rescale : bool
        If True, make the timestamps `tstp` start from 0, by subtracting
        the 1st vector component to them; leave them unchanged else.
            :Default: True (rescale the timestamp)

    Returns
    -------
    gate : np.ndarray
        The amplitudes of the gate filter's time impulse response.

    Examples
    --------
    >>> import numpy as np

    >>> tstp = np.linspace(100., 500., 200)
    >>> gate = rectangular_time(tstp, (150., 300.))
    """

    # Rescale the timestamps (start them from 0) if `rescale`
    if rescale and tstp[0] != 0.:
        tstp -= tstp[0]

    # Avoid null values (if x --> 0+, sinc(x) ~ 1 - x^2/3! ~ 1)
    tstp[tstp == 0] = 1e-6

    # Compute the rectangular filters
    gate = _gate(tstp, 2.*np.pi*bandwidth[0], 2.*np.pi*bandwidth[1])

    # Mirror the values if `sym` or return them as they are else
    return flip(gate, center, True) if sym else gate
#----------------------------------------------------------------------------#

#--------------------   Time-Domain Triangular Filter   ---------------------#
def _triangle(tstp, fmin, fmax):
    """ Triangle-shape filter core -- See `triangular_time` """
    # Filter characteristic frequencies
    fcnt = 0.5 * (fmax + fmin)
    flw = fmax - fcnt                   # Lower frequency (lower cutoff)
    fmid = fcnt + fmin                  # Center frequency (triangle peak)
    fup = fmax + fcnt                   # Upper frequency (upper cutoff)

    # Compute the sinc filters
    # 2pi*f * sinc(2pi*ft) = sin(2pi*ft)/t
    return (np.sin(flw*tstp) * np.sin(fup*tstp)
            - np.sin(flw*tstp) * np.sin(fmid*tstp)) / (flw*(fup-fmid)*tstp**2)
#    # Do not add pi in sinc, since np.sinc(x) = np.sin(np.pi*x)/(np.pi*x)
#    return (fup * np.sinc(flw*tstp) * np.sinc(fup*tstp)
#            - fmid * np.sinc(flw*tstp) * np.sinc(fmid*tstp)) / (fup-fmid)

def triangular_time(tstp, bandwidth, sym=True, center=None, rescale=True):
    """ Triangular-shaped filter time impulse response

    Take a set of timestamps and the lower and upper cutoff frequencies
    of the triangular filter, and compute the temporal response of it,
    defined as the difference between two half-parallelogram filters,
    obtained by computing the products of two sinc functions:
        h(t) = para(tstp, fcnt, fmax) - para(tstp, fmin, fcnt)
    where fcnt = (fmax+fmin)/2, and the `para` function can be given by:
        para(t, x, y) = (y-x)*sinc(pi*(y-x)*t) * (y+x)*sinc(pi*(y+x)*t)
    with x = fmin, y = fmax and t = tstp.

    Although the above solution works fine, the current implementation
    directly fuses both `para` functions into one expression.

    The filter response is finally normalized by dividing it by its max,
    i.e. h'(t) = h(t) / h(0) = h(t) / flw*(fup-fmid)

    If `sym`, consider the timestamps `tstp` as only half of the inputs,
    and, since sinc is an even function, mirror the filter response to
    these inputs backwards; stack the two vectors as [mirror, outputs].
    If `center` is a number, add this value in-between the two vectors;
    if `None`, just stack both side-by-side.

    If `rescale`, assuming that `tstp` is organized in ascending order,
    make them start from 0 by subtracting the first stamp to any other.

    The vector returned aims to be convolved with the temporal represen-
    tation of the signal (see `convolve` function).

    Parameters
    ----------
    tstp : np.ndarray
        The timestamps to filter with the gate filter.
    bandwidth : 2-tuple of ints or floats
        The filter lower and upper cutoff frequencies as (flow, fupp).
    [OPT] sym : bool
        If True, mirror backwards the filter response to `tstp`, and
        stack both vectors as [mirror, response].
            :Default: True
    [OPT] center : int, float or None
        If not None, the scalar value to add in-between the mirror and
        the response; only used if `sym` is True; if None, no value is
        added. Ensures the response parity (odd or even cardinal).
            :Default: None (no value added)
    [OPT] rescale : bool
        If True, make the timestamps `tstp` start from 0, by subtracting
        the 1st vector component to them; leave them unchanged else.
            :Default: True (rescale the timestamp)

    Returns
    -------
    trig : np.ndarray
        The amplitudes of the triangle filter's time impulse response.

    Examples
    --------
    >>> import numpy as np

    >>> tstp = np.linspace(100., 500., 200)
    >>> trig = triangular_time(tstp, (150., 300.))
    """

    # Rescale the timestamps (start them from 0) if `rescale`
    if rescale and tstp[0] != 0.:
        tstp -= tstp[0]

    # Avoid null values (if x --> 0+, sinc(x) ~ 1 - x^2/3! ~ 1)
    tstp[tstp == 0] = 1e-6

    # Compute the triangular filters
    trig = _triangle(tstp, np.pi*bandwidth[0], np.pi*bandwidth[1])

    # Mirror the values if `sym` or return them as they are else
    return flip(trig, center) if sym else trig
#----------------------------------------------------------------------------#

#---------------------   Time-Domain Gammatone Filter   ---------------------#
def gammatone_time(tstp, bandwidth, rescale=True, order=2):
    """ Gammatone filter time impulse response

    Take a set of timestamps and the lower and upper cutoff frequencies
    of the gammatone filter, and compute the temporal response of it,
    defined as:
        h(t) = A * t^(n-1) * exp(-2pi*b*t) * cos(2pi*f*t)
    where t is time, A is the filter amplitude, b is its bandwidth, f is
    the central frequency, and n is the filter order.

    In the current implementation, the following constants are used:
      - A = 1
      - b = `fmax` - `fmin`
      - f = (`fmax` + `fmin`) / 2
    Time `t` and order `n` are dynamic arguments (`tstp` and `order`).

    The vector returned aims to be convolved with the temporal represen-
    tation of the signal (see `convolve` function).

    Parameters
    ----------
    tstp : np.ndarray
        The timestamps to filter with the gammatone filter.
    bandwidth : 2-tuple of ints or floats
        The filter lower and upper cutoff frequencies as (flow, fupp).
    [OPT] order : int
        Filter order (the power of `t` is `order`-1).
            :Default: 2
    [OPT] rescale : bool
        If True, make the timestamps `tstp` start from 0, by subtracting
        the 1st vector component to them; leave them unchanged else.
            :Default: True (rescale the timestamp)

    Returns
    -------
    gamma : np.ndarray
        The amplitudes of the gammatone filter's time impulse response.

    Examples
    --------
    >>> import numpy as np

    >>> tstp = np.linspace(100., 500., 200)
    >>> gamma = gammatone_time(tstp, (150., 300.), order=2)
    """

    # Rescale the timestamps (start them from 0) if `rescale`
    if rescale and tstp[0] != 0.:
        tstp -= tstp[0]

    # Filter characteristics
    twopi = 2.*np.pi                            # Double of Pi as a constant
    bdwd = bandwidth[1] - bandwidth[0]          # Bandwidth length
    fcnt = 0.5 * (bandwidth[1] + bandwidth[0])  # Center frequency (peak)

    # Build the filter
    if order == 1:
        return np.exp(-twopi*bdwd*tstp) * np.cos(twopi*fcnt*tstp)
    return tstp**(order-1) * np.exp(-twopi*bdwd*tstp) * np.cos(twopi*fcnt*tstp)
#----------------------------------------------------------------------------#

##############################################################################



##############################################################################
##                         Frequency-Domain Filters                         ##
##############################################################################

#--------------------   Freq-Domain Rectangular Filter   --------------------#
def rectangular_freq(freq, fmin, fmax):
    """ Frequency-domain rectangular-shaped filter

    For a set of frequencies, build the spectral response of a rectangle
    filter, which is a filter whose response to any frequency inside the
    bandwidth is one, and that to the other frequencies is zero.

    The vector returned aims to be multiplied by the spectral represen-
    tation of the signal, i.e. its (F)FT.

    Parameters
    ----------
    freq : np.ndarray
        The frequencies to filter with the gate filter.
    fmin : int or float
        The filter lower cutoff frequency.
    fmax : int or float
        The filter upper cutoff frequency.

    Returns
    -------
    gate : np.ndarray
        The amplitudes of the freq-response of the gate filter.

    Examples
    --------
    >>> import numpy as np

    >>> freq = np.linspace(100., 500., 200)
    >>> gate = rectangular_freq(freq, 200., 300.)
    """

    # Instantiate and fill the gate vector
    gate = np.zeros_like(freq, float)
    gate[np.argwhere((fmin <= freq) & (freq <= fmax))] = 1.0

    return gate
#----------------------------------------------------------------------------#

#--------------------   Freq-Domain Triangular Filter   ---------------------#
def triangular_freq(freq, fmin, fmax):
    """ Frequency-domain triangular-shaped filter

    For a set of frequencies, build the spectral response of a triangle
    filter, which is a filter whose response to any frequency inside the
    bandwidth follows a triangular function centered around the mean of
    the bandwidth, and that to the other frequencies is zero.

    The vector returned aims to be multiplied by the spectral represen-
    tation of the signal, i.e. its (F)FT.

    Parameters
    ----------
    freq : np.ndarray
        The frequencies to filter with the triangular filter.
    fmin : int or float
        The filter lower cutoff frequency.
    fmax : int or float
        The filter upper cutoff frequency.

    Returns
    -------
    trig : np.ndarray
        The amplitudes of the freq-response of the triangle filter.

    Examples
    --------
    >>> import numpy as np

    >>> freq = np.linspace(100., 500., 200)
    >>> trig = triangular_freq(freq, 200., 300.)
    """

    # Center of the bandwidth
    fcnt = 0.5 * (fmax + fmin)

    trig = np.zeros_like(freq, float)

    # Left side of the triangle (lower frequencies)
    pos = np.argwhere((fmin <= freq) & (freq <= fcnt))
    trig[pos] =  (freq[pos] - fmin) / (fcnt - fmin)

    # Right side of the triangle (upper frequencies)
    pos = np.argwhere((fcnt < freq) & (freq <= fmax))
    trig[pos] = (freq[pos] - fmax) / (fcnt - fmax)

    return trig
#----------------------------------------------------------------------------#

##############################################################################
