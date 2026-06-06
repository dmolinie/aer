""" Cepstral transformation

Authors: Dylan MOLINIE
Company: CEA Paris-Saclay, Gif-sur-Yvette, France
Contact: dylan.molinie@cea.fr
Date: January 2025
Last revised: March 2026

License: GPLv3
"""

__all__ = [
    'Cepstrum', 'cepstrum', 'ceps_quef', 'ceps_sfft', 'cepstral_power']

import numpy as np
from numpy.fft import ifft, fftshift

from aer.spectrum._spectrum import Spectrum


##############################################################################
##                          Cepstrum Toolkit Class                          ##
##############################################################################

class Cepstrum(Spectrum):
    """ Cepstrum class

    This class is a continuation of the `Spectrum` class, providing addi-
    tional tools relating to cepstral analysis.

    Attributes
    ----------
    fftparams : dict, getter & setter
        The FFT parameters.
    freq : float, getter & setter
        The spectrum frequencies.
    sfft : np.ndarray, getter & setter
        The spectrum components.
    quef : np.ndarray, getter & setter
        The cepstrum quefrencies.
    ceps : np.ndarray, getter & setter
        The cepstrum components.

    Constructor
    -----------
    __init__(**fftparams)

    Magic Methods
    -------------
    __getitem__(pos)
        Get the cepstrum component at pos --> value = ceps[pos].
    __setitem__(pos, value)
        Set the cepstrum component at pos --> ceps[pos] = value.
    __iter__()
        Iterate on the cepstrum components --> for i in ceps: [...].
    __len__()
        Length of the cepstrum --> len(ceps).
    __repr__()
        Display the cepstrum --> repr(ceps).
    __str__()
        Print the cepstrum --> print(ceps).

    Methods
    -------
    spectrum(data, onesided=False, shift=True)
        Compute the FFT of an input signal.
    spectral_power()
        Spectral power density estimation (periodogram).
    cepstrum(**ifftparams)
        Cepstrum transformation of a given signal from its FT.
    cepstral_power()
        Cepstral power density estimation.

    Examples
    --------
    >>> import numpy as np

    # Generate dummy data
    >>> frate = 200.
    >>> data = np.arange(0., 100., 1./frate, dtype=float)

    # Build the cepstrogram
    >>> cepst = Cepstrum(axis=-1)
    >>> cepst.spectrum(data)
    >>> cepst.freq *= frate
    >>> cepst.cepstrum()
    >>> cxx = cepst.cepstral_power()
    """

    #---------------------------   Constructor   ----------------------------#
    def __init__(self, **fftparams):
        """ Instantiate a Cepstrum object (constructor)

        Other Parameters
        ----------------
        **fftparams : inline keyword arguments, optional
            The NumPy FFT parameters (see `np.fft.fft`).

        Examples
        --------
        # Default parameters
        >>> cepst = Cepstrum()

        # With parameters for the NumPy's FFT
        >>> cepst = Cepstrum(n=100, norm='backward')
        """
        super().__init__(**fftparams)
        self._ceps = np.empty(0, complex)           # Cepstrum components
        self._quef = np.empty(0, float)             # Quefrencies
    #------------------------------------------------------------------------#

    #-------------------   Magic Methods and Properties   -------------------#
    def __getitem__(self, pos):
        """ Get the cepstrum value at pos """
        return self._ceps[pos]

    def __setitem__(self, pos, value):
        """ Set the cepstrum value at pos """
        self._ceps[pos] = value

    def __iter__(self):
        """ Iterate on the cepstrum """
        return iter(self._ceps)

    def __len__(self):
        """ Length of the cepstrum """
        return len(self._ceps)

    def __repr__(self):
        """ Display the cepstrum """
        return repr(self._ceps)

    def __str__(self):
        """ Print the cepstrum """
        return "Quefrencies:\n" + str(self._quef)\
               + "\nCepstrum components:\n" + str(self._ceps)

    @property
    def quef(self):
        """ Get the cepstrum quefrencies """
        return self._quef

    @quef.setter
    def quef(self, quef):
        """ Set the cepstrum quefrencies """
        self._quef = quef

    @property
    def ceps(self):
        """ Get the cepstrum components """
        return self._ceps

    @ceps.setter
    def ceps(self, ceps):
        """ Set the cepstrum components """
        self._ceps = ceps
    #------------------------------------------------------------------------#

    #---------------------   Cepstral Transformation   ----------------------#
    def cepstrum(self, onesided=True, shift=True, **ifftparams):
        """ Cepstrum transformation of a given signal from its FT

        Compute the cepstrum and set the `ceps` and `quef` attributes;
        the cepstrum is defined as:
            C{x} = F^{-1} { log( |F{x}| ) }
        This implementation uses the natural logarithm (base e).

        Note: the quefrencies q are linked to frequencies f by: f=fs/q,
              with fs the sampling frequency.

        See `cepstrum` function in the same module for details.

        Parameters
        ----------
        [OPT] onesided : bool
            Return both positive and negative quefrencies if set to False;
            return only the positive ones if set to True.
                :Default: True
        [OPT] shift : bool
            Reorganize the quefrencies and their components if set to True
            (-N -> +N-1, with N the length of `sfft`); leave them in their
            original order else. Used only if not `onesided`.
                :Default: True

        Other Parameters
        ----------------
        **ifftparams : inline keyword arguments, optional
            The parameters of the IFT function (see `numpy.fft.ifft`).

        Returns
        -------
        None : set the `quef` and `ceps` attributes directly.

        Examples
        --------
        >>> import numpy as np

        >>> data = np.sin(2*np.pi*5.*np.linspace(0., 1.0, 100))

        >>> cepst = Cepstrum(axis=-1, norm=None, n=100)
        >>> cepst.spectrum(data)
        >>> cepst.spectrum(data, onesided=False, shift=False)
        >>> cepst.cepstrum(n=100, norm='backward')
        """
        self._quef, self._ceps = cepstrum(
            self._sfft, onesided, shift, **ifftparams)
    #------------------------------------------------------------------------#

    #----------------------   Power Cepstral Density   ----------------------#
    def cepstral_power(self):
        """ Cepstral power density estimation

        Compute and return the cepstral power density, defined as:
            Cxx = |F^{-1}{log(|F{x}|^2)}|^2 = 4|C{x}|^2

        See `cepstral_power` function in the same module for details.

        Parameters
        ----------
        Nothing, only use `self`.

        Returns
        -------
        cxx : np.ndarray
            The power cepstral density estimate.

        Examples
        --------
        >>> import numpy as np

        >>> data = np.sin(2*np.pi*5.*np.linspace(0., 1.0, 100))

        >>> cepst = Cepstrum(axis=-1, norm=None)
        >>> cepst.spectrum(data)
        >>> cepst.cepstrum(n=100, norm='backward')
        >>> cxx = cepst.cepstral_power()
        """

        # Check if nonempty cepstrum (if 'cepstrum' has been run before)
        if len(self._ceps) == 0:
            print("No cepstrum, please run `cepstrum` method first")
            return np.empty(0)

        # Return the estimated power cepstral density
        return cepstral_power(self._ceps)
    #------------------------------------------------------------------------#

##############################################################################



##############################################################################
##                          Cepstrum as Functions                           ##
##############################################################################

#-------------------------   Cepstrum Frequencies   -------------------------#
def ceps_quef(samples, onesided=True, shift=True):
    """ Build the frequency scale for an FFT

    Take the nb of samples of a signal FFT & build its quefrency scale.
    If `onesided` is True, return only the positive quefrencies; else,
    return both positive & negative ones. If `shift` is set to True &
    `onesided` to False, shift the positive and negative quefrencies;
    ignore `shift` if `onesided` is True.

    See the `cepstrum` function for more details.

    Parameters
    ----------
    samples : int
        The number of samples in the FFT of the signal.
    [OPT] onesided : bool
        Return both positive and negative quefrencies if set to False;
        return only the positive ones if set to True.
            :Default: True
    [OPT] shift : bool
        Reorganize the quefrencies and their components if set to True
        (-N -> +N-1, with N the length of `sfft`); leave them in their
        original order else. Used only if not `onesided`.
            :Default: True

    Returns
    -------
    quef : np.ndarray
        The quefrencies as the reciprocal of the indices.

    Examples
    --------
    >>> import numpy as np

    >>> data = np.sin(2*np.pi*5.*np.linspace(0., 1.0, 100))

    >>> sfft = spec_sfft(data, onesided=False, shift=False)

    >>> ceps_quef(len(sfft), True)
    array([ 0, 1,  2, ...,  47,  48,  49])

    >>> ceps_quef(len(sfft), False, True)
    array([-50, -49, -48, ...,  47,  48,  49])
    """

    # Build the quefrencies
    span = samples // 2         # Euclidean division quotient
    remd = samples % 2          # Euclidean division remainder
    if remd == 0:               # Even number of samples
        quef = np.hstack((np.arange(0, span), np.arange(-span, 0)))
    else:                       # Odd number of samples
        quef = np.hstack((np.arange(0, span), span, np.arange(-span, 0)))

    # Return the quefrencies wrt the options
    if onesided:
        return quef[:span+remd]
    return fftshift(quef) if shift else quef
#----------------------------------------------------------------------------#

#-------------------------   Cepstrum Components   --------------------------#
def ceps_sfft(sfft, onesided=True, shift=True, **ifftparams):
    """ Compute the FFT components of the data provided as input

    Take the spectrum (FFT) of a signal and build its cepstrum, defined
    as: C{x} = F^{-1} {log(|F{x}|)}. If `onesided` is True, return only
    the components associated with the positive quefrencies; else, return
    that of both the pos. & neg. ones. If `shift` is True and `onesided`
    is False, shift the two sets of components; ignore `shift` else.

    Note that the cepstrum requires the terms associated with both neg.
    & pos. frequencies (i.e. the FFT should not be `onesided`).

    See the `cepstrum` function for more details.

    Parameters
    ----------
    sfft : array_like
        The freq-domain FT samples for which to compute the cepstrum.
    [OPT] onesided : bool
        Return both positive and negative quefrencies if set to False;
        return only the positive ones if set to True.
            :Default: True
    [OPT] shift : bool
        Reorganize the quefrencies and their components if set to True
        (-N -> +N-1, with N the length of `sfft`); leave them in their
        original order else. Used only if not `onesided`.
            :Default: True

    Other Parameters
    ----------------
    **ifftparams : inline keyword arguments, optional
        The parameters of the IFT function (see `numpy.fft.ifft`).

    Returns
    -------
    ceps : np.ndarray
        The cepstrum of the signal.

    Examples
    --------
    >>> import numpy as np

    >>> data = np.sin(2*np.pi*5.*np.linspace(0., 1.0, 100))

    >>> sfft = spec_sfft(data, onesided=False, shift=False)
    >>> ceps_sfft(sfft, False, True)
    array([-0.35383994-1.14916407e-18j, -0.35140556-5.00829198e-16j, ...,
           -0.3443317 -1.65747429e-16j, -0.35140556+4.98509860e-16j])
    """
    # Compute the components of the cepstrum
    ceps = ifft(np.log(np.abs(sfft)), **ifftparams)
    # Return the cepstrum components wrt the options
    if onesided:
        return ceps[:len(sfft)//2 + len(sfft)%2]
    return fftshift(ceps) if shift else ceps
#----------------------------------------------------------------------------#

#-----------------------   Cepstral Transformation   ------------------------#
def cepstrum(sfft, onesided=True, shift=True, **ifftparams):
    """ Cepstrum transformation of a given signal from its FT

    Given the spectrum (i.e. Fourier Transform) of a signal, compute its
    cepstrum, defined as:
        C{x} = F^{-1} { log( |F{x}| ) }
    This implementation uses the natural logarithm (base e).

    Important: the IFT requires the terms associated with both negative
               and positive frequencies (i.e. the `onesided` should not
               be used when computing the FFT).

    Note: the indexes cepstrum are called quefrencies q; they are linked
          to frequencies f by: f = fs/q, with fs the sampling frequency.
          The cepstrum can be plotted against its quefrencies (indexes),
          or against fs/q to draw a parallel with the spectrum.

    The cepstrum can also be computed using the `acoustics` module:
        ceps = acoustics.cepstrum.real_cepstrum(data)

    Parameters
    ----------
    sfft : array_like
        The freq-domain FT samples for which to compute the cepstrum.
    [OPT] onesided : bool
        Return both positive and negative quefrencies if set to False;
        return only the positive ones if set to True.
            :Default: True
    [OPT] shift : bool
        Reorganize the quefrencies and their components if set to True
        (-N -> +N-1, with N the length of `sfft`); leave them in their
        original order else. Used only if not `onesided`.
            :Default: True

    Other Parameters
    ----------------
    **ifftparams : inline keyword arguments, optional
        The parameters of the IFT function (see `numpy.fft.ifft`).

    Returns
    -------
    quef : np.ndarray
        The quefrencies as the reciprocal of the indices.
    ceps : np.ndarray
        The cepstrum of the signal.

    Examples
    --------
    >>> import numpy as np

    >>> data = np.sin(2*np.pi*5.*np.linspace(0., 1.0, 100))

    >>> sfft = spec_sfft(data, onesided=False, shift=False)
    >>> quef, ceps = cepstrum(sfft)
    """
    # Compute and return the quefrencies and their associated components
    return ceps_quef(len(sfft), onesided, shift),\
           ceps_sfft(sfft, onesided, shift, **ifftparams)
#----------------------------------------------------------------------------#

#------------------------   Power Cepstral Density   ------------------------#
def cepstral_power(ceps):
    """ Cepstral power density

    Compute the cepstral power density, defined as:
        Cxx = |F^{-1}{log(|F{x}|^2)}|^2 = 4|C{x}|^2

    Note: to link the cepstral indexes to regular frequencies, see the
          note in the `cepstrum` function.

    Parameters
    ----------
    ceps : np.ndarray
        The cepstrum of the signal.

    Returns
    -------
    cxx : np.ndarray
        The signal power cepstral density estimate.

    Examples
    --------
    >>> import numpy as np

    >>> data = np.sin(2*np.pi*5.*np.linspace(0., 1.0, 100))

    >>> sfft = spec_sfft(data, onesided=False, shift=False)
    >>> ceps = ceps_sfft(sfft)
    >>> cxx = cepstral_power(ceps)
    """
    return 4. * np.abs(ceps)**2
#----------------------------------------------------------------------------#

##############################################################################
