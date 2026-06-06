""" Windowing functions

Authors: Dylan MOLINIE
Company: CEA Paris-Saclay, Gif-sur-Yvette, France
Contact: dylan.molinie@cea.fr
Date: January 2025
Last revised: March 2026

License: GPLv3
"""

__all__ = ['cosine_sum', 'hann', 'hamming', 'blackman']

import numpy as np


##############################################################################
##                           Windowing Functions                            ##
##############################################################################

#--------------------------   Cosine-Sum Window   ---------------------------#
def cosine_sum(nb_points, coefs, dtype='float64'):
    r""" General Cosine-sum window

    Compute and return the general cosine-sum window in the `nb_points`-
    long interval [-pi; +pi], which is defined as:
        \forall n in I, w(n) = \sum_{k} (-1)^k a_k cos(2pi*k*n/N)
                             = a0 + \sum_{k>1} (-1)^k a_k cos(2pi*k*n/N)
    where I is the definition domain ([-pi; +pi] here), n is the current
    point, and N is the total number of points (`nb_points` here). The
    total number of cosines to add is defined by the number of elements
    contained in the `coefs` array like argument.

    Parameters
    ----------
    nb_points : int
        The total number of points of the window; in practice, the vari-
        able `n` takes every value in [-pi, +pi] / N.
    coefs : array_like
        The set of coefficients {ai}, weighting the different cosines.
    [OPT] dtype : str
        The format of the data; must be numpy array compliant.
            :Default: 'float64'

    Returns
    -------
    win : np.ndarray
        The `nb_points`-long Cosine-sum window computed over [-pi; +pi].

    Examples
    --------
    >>> import numpy as np

    >>> cosine_sum(7, [0.5, 0.5])                       # Hann
    array([0.  , 0.25, 0.75, 1.  , 0.75, 0.25, 0.  ])

    >>> cosine_sum(7, [0.54, 0.46])                     # Hamming
    array([0.08, 0.31, 0.77, 1.  , 0.77, 0.31, 0.08])

    >>> cosine_sum(7, [0.42, 0.5, 0.08])                # Blackman
    array([-0.  ,  0.13,  0.63,  1.  ,  0.63,  0.13, -0.  ])
    """
    # Generate the points between -pi and +pi to have the window centered
    # in 0, and thus in the middle of the interval [-pi; +pi]
    points = np.linspace(-np.pi, np.pi, nb_points, dtype=dtype)
    win = np.full(nb_points, coefs[0])
    for k, coef in enumerate(coefs[1:], 1):
        win += coef * np.cos(k * points)
    return win
#----------------------------------------------------------------------------#

#-----------------------------   Hann Window   ------------------------------#
def hann(nb_points, dtype='float64'):
    r""" Hann window function

    Compute and return the Hann window in the `nb_points`-long interval
    [0; +2pi], which is defined as:
        \forall n in I, w(n) = 0.5 * (1 - cos(2pi * n/N))
                             = sin^2(pi * n/N)
    where I is the definition domain ([0; +2pi] here), n is the current
    point, and N is the total number of points (`nb_points` here).

    Parameters
    ----------
    nb_points : int
        The total number of points of the window; in practice, the vari-
        able `n` takes every value in [-pi, +pi] / N.
    [OPT] dtype : str
        The format of the data; must be numpy array compliant.
            :Default: 'float64'

    Returns
    -------
    win : np.ndarray
        The `nb_points`-long Hann window computed over [-pi; +pi].

    Examples
    --------
    >>> hann(7)
    array([0.  , 0.25, 0.75, 1.  , 0.75, 0.25, 0.  ])
    """
    # sin: [0; pi] ; cos: [0; 2pi]
    points = np.linspace(0., np.pi, nb_points, dtype=dtype)
    return np.sin(points)**2
#----------------------------------------------------------------------------#

#----------------------------   Hamming Window   ----------------------------#
def hamming(nb_points, dtype='float64'):
    r""" Hamming window function

    Compute and return the Hamming window in the `nb_points`-long inter-
    val [0; +2pi], which is defined as:
        \forall n in I, w(n) = 25/46 - 21/46 * cos(2pi * n/N)
                             ~ 0.54 - 0.46 * cos(2pi * n/N)
    where I is the definition domain ([0; +2pi] here), n is the current
    point, and N is the total number of points (`nb_points` here).

    Parameters
    ----------
    nb_points : int
        The total number of points of the window; in practice, the vari-
        able `n` takes every value in [0; +2pi] / N.
    [OPT] dtype : str
        The format of the data; must be numpy array compliant.
            :Default: 'float64'

    Returns
    -------
    win : np.ndarray
        The `nb_points`-long Hamming window computed over [0; +2pi].

    Examples
    --------
    >>> hamming(7)
    array([0.08, 0.31, 0.77, 1.  , 0.77, 0.31, 0.08])
    """
    points = np.linspace(0., 2.*np.pi, nb_points, dtype=dtype)
    return 0.54 - 0.46*np.cos(points)
#----------------------------------------------------------------------------#

#---------------------------   Blackman Window   ----------------------------#
def blackman(nb_points, alpha=0.16, dtype='float64'):
    r""" Blackman window function

    Compute and return the Blackman window in the `nb_points`-long inter-
    val [0; +2pi], which is defined as:
        \forall n in I, w(n) = a0 - a1*cos(2pi * n/N) + a2*cos(2pi * n/N)
        with a0 = (1-alpha)/2, a1 = 1/2 and a2 = alpha/2
    where I is the definition domain ([0; +2pi] here), n is the current
    point, and N is the total number of points (`nb_points` here). Typi-
    cally, alpha is set to 0.16.

    Parameters
    ----------
    nb_points : int
        The total number of points of the window; in practice, the vari-
        able `n` takes every value in [0; +2pi] / N.
    [OPT] alpha : float
        The `offset` to compute the coefficients a0, a1 and a2. Alpha
        is often set to 0.16 to approximate the exact Blackman function.
            :Default: 0.16
    [OPT] dtype : str
        The format of the data; must be numpy array compliant.
            :Default: 'float64'

    Returns
    -------
    win : np.ndarray
        The `nb_points`-long Blackman window computed over [0; +2pi].

    Examples
    --------
    >>> blackman(7)
    array([-0.  ,  0.13,  0.63,  1.  ,  0.63,  0.13, -0.  ])
    """
    coefs = 0.5*np.array((1.-alpha, 1., alpha))
    points = np.linspace(0., 2.*np.pi, nb_points, dtype=dtype)
    return coefs[0] - coefs[1]*np.cos(points) + coefs[2]*np.cos(2.*points)
#----------------------------------------------------------------------------#

##############################################################################
