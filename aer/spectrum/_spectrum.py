""" Spectral transformation

Authors: Dylan MOLINIE
Company: CEA Paris-Saclay, Gif-sur-Yvette, France
Contact: dylan.molinie@cea.fr
Date: January 2025
Last revised: March 2026

License: GPLv3
"""

__all__ = [
    'onesidedfft', 'twosidedfft',
    'Spectrum', 'spectrum', 'spec_freq', 'spec_sfft', 'spectral_power']

import numpy as np
from numpy.fft import fft, rfft, fftfreq, rfftfreq, fftshift


##############################################################################
##                              Miscellaneous                               ##
##############################################################################

#---------------------   2-sided FFT --> 1-sided FFT   ----------------------#
def onesidedfft(freq, sfft):
    """ Given a two-sided FFT, return the corresponding one-sided FFT

    Parameters
    ----------
    freq : array_like
        The 2-sided frequency scale.
    sfft : array_like
        The 2-sided FFT components.

    Returns
    -------
    freq_os : np.ndarray
        The 1-sided frequency scale (positive frequencies only).
    sfft_os : np.ndarray
        The 1-sided Fourier transformation (pos. freqs components).

    Examples
    --------
    >>> import numpy as np

    >>> data = np.sin(2*np.pi*5.*np.linspace(0., 1.0, 100))

    >>> freq, sfft = spectrum(data, onesided=False, shift=False)
    >>> freq_os, sfft_os = onesidedfft(freq, sfft)
    """

    lgt = len(freq) // 2                    # Freq scale center (FT window)

    # If the fft is shifted (neg freqs -> pos freqs)
    if freq[0] < 0:
        if len(freq) % 2 == 1:
            return freq[lgt:], sfft[lgt:]               # Odd nb of samples
        return np.hstack((freq[lgt:], -freq[0])),\
               np.hstack((sfft[lgt:], sfft[0].conj()))  # Even nb of samples

    # If the fft is naturally organized (pos freqs -> neg freqs)
    if len(freq) % 2 == 0:
        freq[lgt] = -freq[lgt]              # freq[lgt] is negative
        sfft[lgt] = sfft[lgt].conj()        # Conjugate of sfft[lgt]

    return freq[:lgt+1], sfft[:lgt+1]       # Return the pos. components
#----------------------------------------------------------------------------#

#---------------------   1-sided FFT -> 2-sided FFT   ----------------------#
def twosidedfft(freq, sfft, shift=False, odd=False):
    """ Rebuild the whole FFT (neg & pos freqs) from a one-sided FFT

    Parameters
    ----------
    freq : array_like
        The 1-sided frequency scale.
    sfft : array_like
        The 1-sided FFT components.
    [OPT] shift : bool
        If the rebuilt 2-sided frequency scale and FFT components should
        be organized in increasing frequency order.
            :Default: False (natural FFT)
    [OPT] odd : bool
        If the rebuilt 2-sided frequency scale and FFT components should
        contain an odd or even number of samples.
            :Default: False (even)

    Returns
    -------
    freq_ts : np.ndarray
        The rebuilt 2-sided frequency scale.
    sfft_ts : np.ndarray
        The rebuilt 2-sided Fourier transformation.

    Examples
    --------
    >>> import numpy as np

    >>> data = np.sin(2*np.pi*5.*np.linspace(0., 1.0, 100))

    >>> freq, sfft = spectrum(data, onesided=True)
    >>> freq_ts, sfft_ts = twosidedfft(freq, sfft, True)
    >>> freq_ts, sfft_ts = twosidedfft(freq, sfft, False, False)
    """

    # If odd nb of samples, mirror all values; do not double last if even
    end = [-1, None][odd]

    # If the fft soulbd should be shifted (neg freqs -> pos freqs)
    if shift:
        return np.hstack((-freq[-1:0:-1], freq[:end])),\
               np.hstack((sfft[-1:0:-1].conj(), sfft[:end]))

    # If the fft should be naturally organized (pos freqs -> neg freqs)
    return np.hstack((freq[:end], -freq[-1:0:-1])),\
           np.hstack((sfft[:end], sfft[-1:0:-1].conj()))
#----------------------------------------------------------------------------#

##############################################################################



##############################################################################
##                          Spectrum Toolkit Class                          ##
##############################################################################

class Spectrum():
    """ Spectrum class

    This class embeds the NumPy FFT into a slightly more elaborated wrap-
    per to represent the spectrum and periodogram of an input signal.

    Attributes
    ----------
    fftparams : dict, getter & setter
        The FFT parameters.
    freq : float, getter & setter
        The spectrum frequencies.
    sfft : np.ndarray, getter & setter
        The spectrum components.

    Constructor
    -----------
    __init__(**fftparams)

    Magic Methods
    -------------
    getitem(pos)
        Get the spectrum component at pos --> value = spec[pos].
    setitem(pos, value)
        Set the spectrum component at pos --> spec[pos] = value.
    iter()
        Iterate on the spectrum components --> for i in spec: [...].
    len()
        Length of the spectrum --> len(spec).
    repr()
        Display the spectrum --> repr(spec).
    str()
        Print the spectrum --> print(spec).

    Methods
    -------
    spectrum(data, onesided=False, shift=True)
        Compute the FFT of an input signal.
    spectral_power()
        Spectral power density estimation (periodogram).

    Examples
    --------
    >>> import numpy as np

    # Generate dummy data
    >>> frate = 200.
    >>> data = np.arange(0., 100., 1./frate, dtype=float)

    # Fourier Transform
    >>> freq, sfft = spectrum(data, onesided=False, shift=False)
    >>> freq *= frate

    # Signal spectrum
    >>> freqs, spec = onesidedfft(freq, sfft)

    # Build the spectrogram
    >>> spect = Spectrum(axis=-1)
    >>> spect.spectrum(data, onesided=True)
    >>> spect.freq *= frate
    >>> pxx = spect.spectral_power()
    """

    #---------------------------   Constructor   ----------------------------#
    def __init__(self, **fftparams):
        """ Instantiate a Spectrum object (constructor)

        Other Parameters
        ----------------
        **fftparams : inline keyword arguments, optional
            The NumPy FFT parameters (see `np.fft.fft`).

        Examples
        --------
        # Default parameters
        >>> spect = Spectrum()

        # With parameters for the NumPy's FFT
        >>> spect = Spectrum(n=100, norm='backward')
        """
        self._sfft = np.empty(0, complex)           # Spectrum components
        self._freq = np.empty(0, float)             # Frequencies
        self._fftparams = fftparams                 # FFT params
    #------------------------------------------------------------------------#

    #-------------------   Magic Methods and Properties   -------------------#
    def __getitem__(self, pos):
        """ Get the spectrum value at pos """
        return self._sfft[pos]

    def __setitem__(self, pos, value):
        """ Set the spectrum value at pos """
        self._sfft[pos] = value

    def __iter__(self):
        """ Iterate on the spectrum """
        return iter(self._sfft)

    def __len__(self):
        """ Length of the spectrum """
        return len(self._sfft)

    def __repr__(self):
        """ Display the spectrum """
        return repr(self._sfft)

    def __str__(self):
        """ Print the spectrum """
        return "Frequencies:\n" + str(self._freq)\
               + "\nSpectrum components:\n" + str(self._sfft)

    @property
    def fftparams(self):
        """ Get the FFT parameters """
        return self._fftparams

    @fftparams.setter
    def fftparams(self, **fftparams):
        """ Set the FFT parameters """
        self._fftparams = fftparams

    @property
    def freq(self):
        """ Get the spectrum frequencies """
        return self._freq

    @freq.setter
    def freq(self, freq):
        """ Set the spectrum frequencies """
        self._freq = freq

    @property
    def sfft(self):
        """ Get the spectrum components """
        return self._sfft

    @sfft.setter
    def sfft(self, sfft):
        """ Set the spectrum components """
        self._sfft = sfft
    #------------------------------------------------------------------------#

    #---------------------   Spectral Transformation   ----------------------#
    def spectrum(self, data, onesided=False, shift=True, frate=1.0):
        """ Compute the FFT of an input signal

        Take a data vector, apply the FFT to it and set the `freq` and
        `sfft` attributes as the frequency scale and the FFT components.

        See `spectrum` function in the same module for details.

        Parameters
        ----------
        data : array_like
            The data samples to be Fourier Transformed.
        [OPT] onesided : bool
            Use both positive and negative frequencies if False; use
            only the positive frequencies if True.
                :Default: False
        [OPT] shift : bool
            Reorganize the frequencies and their components if True
            (-0.5 -> +0.5-eps); leave them in their original order else
            (0 -> +0.5-eps -> -0.5 -> -eps). Used only if not onesided.
                :Default: True
        [OPT] frate : float
            The sampling rate to rescale the frequencies from -frate/2
            to +frate/2(-eps/2); set to 1.0 to keep normalized scale.
                :Default: 1.0

        Returns
        -------
        None : set the `freq` and `sfft` attributes directly.

        Examples
        --------
        >>> import numpy as np

        >>> data = np.sin(2*np.pi*5.*np.linspace(0., 1.0, 100))

        >>> spect = Spectrum(axis=-1, norm=None, n=100)
        >>> spect.spectrum(data, True, frate=100)
        """
        self._freq, self._sfft = spectrum(
            data, onesided, shift, frate, **self._fftparams)
    #------------------------------------------------------------------------#

    #----------------------   Power Spectral Density   ----------------------#
    def spectral_power(self):
        """ Spectral power density estimation (periodogram)

        Compute and return the power spectral density (periodogram),
        defined as the squared FT modulus divided by the nb of samples:
            Pxx = |F{x}|^2 / N

        See `spectral_power` function in the same module for details.

        Parameters
        ----------
        Nothing, only use `self`.

        Returns
        -------
        pxx : np.ndarray
            The power spectral density estimate (periodogram).

        Examples
        --------
        >>> import numpy as np

        >>> data = np.sin(2*np.pi*5.*np.linspace(0., 1.0, 100))

        >>> spect = Spectrum(axis=-1, norm=None)
        >>> spect.spectrum(data)
        >>> pxx = spect.spectral_power()
        """

        # Check if nonempty spectrum (if `spectrum` has been run before)
        if len(self._sfft) == 0:
            print("No spectrum, please run `spectrum` method first")
            return np.empty(0)

        # Return the estimated power spectral density (periodogram)
        return spectral_power(self._sfft)
    #------------------------------------------------------------------------#

##############################################################################



##############################################################################
##                          Spectrum as Functions                           ##
##############################################################################

#-------------------------   Spectrum Frequencies   -------------------------#
def spec_freq(samples, onesided=False, shift=True, frate=1.):
    """ Build the frequency scale for an FFT

    Take the number of samples of the signal and generate the frequency
    scale using the NumPy's `fftfreq` function. If `onesided` is True,
    return only the positive frequencies; else, return both positive &
    negative ones. If `shift` is True & `onesided` is False, shift the
    positive and negative frequencies; ignore `shift` if `onesided` is
    True. Finally, multiply the frequencies by `frate` if it is not 1.

    See the `spectrum` function for more details.

    Parameters
    ----------
    samples : int
        The number of samples in the data signal.
    [OPT] onesided : bool
        Return both positive and negative frequencies if set to False;
        return only the positive frequencies if set to True.
            :Default: False
    [OPT] shift : bool
        Reorganize the frequencies if set to True (-0.5 -> +0.5-eps);
        leave them unchanged else (0 -> +0.5-eps -> -0.5 -> -eps).
        Used only if not `onesided`.
            :Default: True
    [OPT] frate : float
        The sampling rate; used to rescale the frequencies from -frate/2
        to +frate/2(-eps/2); set to 1.0 to leave the normalized scale.
            :Default: 1.0

    Returns
    -------
    freq : np.ndarray
        The frequency scale.

    Examples
    --------
    >>> import numpy as np

    >>> data = np.sin(2*np.pi*5.*np.linspace(0., 1.0, 100))

    >>> spec_freq(len(data), True)
    array([0.  , 0.01, 0.02, ..., 0.48, 0.49, 0.5 ])
    >>> len(_)
    51

    >>> spec_freq(len(data), False, True)
    array([-0.5 , -0.49, -0.48, ..., 0.47,  0.48, 0.49])
    >>> len(_)
    100
    """
    # Return only the positive frequencies
    if onesided:
        freq = rfftfreq(samples)
    # Return both the positive and negative frequencies
    else:
        freq = fftshift(fftfreq(samples)) if shift else fftfreq(samples)
    # Rescale the frequencies
    return freq if frate == 1.0 else frate*freq
#----------------------------------------------------------------------------#

#-------------------------   Spectrum Components   --------------------------#
def spec_sfft(data, onesided=False, shift=True, **fftparams):
    """ Compute the FFT components of the data provided as input

    Apply the NumPy's `fft` function to the data to Fourier Transform.
    If `onesided` is True, return only the components associated with
    the positive frequencies; else, return that of both the positive &
    negative ones. If `shift` is True and `onesided` is False, shift
    the two sets of components; ignore `shift` if `onesided` is True.

    See the `spectrum` function for more details.

    Parameters
    ----------
    data : array_like
        The data samples to be Fourier Transformed.
    [OPT] onesided : bool
        Return both positive and negative frequencies if set to False;
        return only the positive frequencies if set to True.
            :Default: False
    [OPT] shift : bool
        Reorganize the frequencies and their components if set to True
        (-0.5 -> +0.5-eps); leave them in their original order else
        (0 -> +0.5-eps -> -0.5 -> -eps). Used only if not `onesided`.
            :Default: True

    Other Parameters
    ----------------
    **fftparams : inline keyword arguments, optional
        The parameters of the FFT function (see `np.fft.fft`).

    Returns
    -------
    sfft : np.ndarray
        The signal Fourier transformation.

    Examples
    --------
    >>> import numpy as np

    >>> data = np.sin(2*np.pi*5.*np.linspace(0., 1.0, 100))

    >>> spec_sfft(data, True)
    array([-1.065814e-14+0.000000e+00j,  6.419757e-03-2.042800e-01j, ...,
           -1.600154e-01+5.028685e-03j, -1.600113e-01+0.000000e+00j])
    >>> len(_)
    51

    >>> spec_sfft(data, False, True, axis=-1, norm=None)
    array([-1.600113e-01+8.326673e-17j, -1.600154e-01-5.028685e-03j, ...,
           -1.600275e-01+1.006808e-02j, -1.600154e-01+5.028685e-03j])
    >>> len(_)
    100
    """
    # Return only the components associated with the positive frequencies
    if onesided:
        return rfft(data, **fftparams)
    # Return both the positive and negative frequencies
    return fftshift(fft(data, **fftparams)) if shift else fft(data, **fftparams)
#----------------------------------------------------------------------------#

#-----------------------   Spectrum Freqs & Compos   ------------------------#
def spectrum(data, onesided=False, shift=True, frate=1., **fftparams):
    """ Compute the FFT of an input signal

    Take the signal TS data, apply the FFT to them and return the trans-
    formed frequency-domain signal and its frequency scale.

    The frequency scale ranges from -0.5 to +0.5-eps, where eps is the
    normalized frequency step. In practice, this scale should be multi-
    plied by the sampling frequency afterwards. Note that the rescaled
    frequencies will range between about -fs/2 and +fs/2. Consequently,
    only the frequencies contained inside this interval can be studied.

    If the signal is real-valued, the FFT is mirrored in the negative
    frequencies: F(-f) = F(+f). Therefore, there may be no need to con-
    sider both sides. Use parameter `onesided` to specify if only the
    positive freqs should be returned, or both neg. and pos. should.

    N.B.: if the whole spectrum is returned, it will comprise as many
        samples as the original data; else, it comprises only half+1
        samples (the positive half of the spectrum plus the zero).

    - x[0] contains the zero frequency term,
    - x[1:n//2] contains the positive-frequency terms,
    - x[n//2+1:] contains the negative-frequency terms, in increasing
          order starting from the most negative frequency.

    Parameters
    ----------
    data : array_like
        The data samples to be Fourier Transformed.
    [OPT] onesided : bool
        Return both positive and negative frequencies if set to False;
        return only the positive frequencies if set to True.
            :Default: False
    [OPT] shift : bool
        Shift the components associated with the pos. & neg. frequencies
        if True; leave them unchanged else. Used only if not `onesided`.
            :Default: True
    [OPT] frate : float
        The sampling rate; used to rescale the frequencies from -frate/2
        to +frate/2(-eps/2); set to 1.0 to leave the normalized scale.
            :Default: 1.0

    Other Parameters
    ----------------
    **fftparams : inline keyword arguments, optional
        The parameters of the FFT function (see `np.fft.fft`).

    Returns
    -------
    freq : np.ndarray
        The normalized frequency scale (-0.5 to +0.5-eps).
    sfft : np.ndarray
        The signal Fourier transformation.

    Examples
    --------
    >>> import numpy as np

    >>> data = np.sin(2*np.pi*5.*np.linspace(0., 1.0, 100))

    >>> freq, sfft = spectrum(data, True)
    >>> freq, sfft = spectrum(data, True, axis=-1, norm=None)
    >>> freq, sfft = spectrum(data, True, n=500, norm='forward')
    """
    # Nb of samples for the FFT
    if 'n' not in fftparams:                        # If `n` not provided,
        fftparams['n'] = data.shape[-1]             # add it to the dict
    # Compute and return the frequencies and their associated FFT components
    return spec_freq(fftparams['n'], onesided, shift, frate),\
           spec_sfft(data, onesided, shift, **fftparams)
#----------------------------------------------------------------------------#

#------------------------   Power Spectral Density   ------------------------#
def spectral_power(sfft):
    """ Spectral Power Density estimation (periodogram)

    Take the Fourier Transform of a signal, and compute its periodogram,
    defined as the squared FT modulus, divided by the number of samples:
        Pxx = |F{x}|^2 / N

    Note: one may have a look at the scipy.signal.periodogram function.

    Parameters
    ----------
    sfft : array_like
        The freq-domain FT samples for which to compute the periodogram.

    Returns
    -------
    pxx : np.ndarray
        The signal power spectral density estimate (periodogram).

    Examples
    --------
    >>> import numpy as np

    >>> data = np.sin(2*np.pi*5.*np.linspace(0., 1.0, 100))

    >>> pxx = spectral_power(spec_sfft(data, True))
    """
    return np.abs(sfft)**2 / len(sfft)
#----------------------------------------------------------------------------#

##############################################################################
