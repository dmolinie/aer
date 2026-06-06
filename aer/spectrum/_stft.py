""" Short Time Fourier Transform (STFT)

Authors: Dylan MOLINIE
Company: CEA Paris-Saclay, Gif-sur-Yvette, France
Contact: dylan.molinie@cea.fr
Date: January 2025
Last revised: March 2026

License: GPLv3
"""

__all__ = ['ShortTimeFT', 'stft_chunks', 'stft_time', 'spectrogram']

import numpy as np
from numpy.fft import fft, rfft, fftfreq, fftshift, rfftfreq

from aer import data_tk as dtk


##############################################################################
##                       Short-Time Fourier Transform                       ##
##############################################################################

class ShortTimeFT():
    """ Short-Time Fourier Transform class

    This class locally implements the STFT, as the SciPy ShortTimeFFT
    class, but also faster but more restrictive.

    Attributes
    ----------
    stftparams : dict, getter & setter
        The STFT parameters.
    freq : np.ndarray, getter & setter
        The STFT frequencies.
    stft : np.ndarray, getter & setter
        The STFT components.
    specs : dict, getter & setter
        The specifications.

    Constructor
    -----------
    __init__(**stftparams)

    Magic Methods
    -------------
    __getitem__(pos)
        Get the STFT component at pos --> value = sft[pos].
    __setitem__(pos, value)
        Set the STFT component at pos --> sft[pos] = value.
    __iter__()
        Iterate on the STFT components --> for i in sft: [...].
    __len__()
        Length of the STFT --> len(sft).
    __repr__()
        Display the STFT --> repr(sft).
    __str__()
        Print the STFT --> print(sft).

    Methods
    -------
    stft_chunks(chunks)
        Apply the STFT to a set of signal chunks.
    stft_time(data, window, hop_size)
        Build the Short-Time Fourier Transform.
    spectrogram()
        Compute the spectrogram amplitude of the STFT.

    Examples
    --------
    >>> import numpy as np
    >>> from scipy.signal.windows import gaussian

    # Generate dummy data
    >>> frate = 200.
    >>> data = np.arange(0., 100., 1./frate, dtype=float)

    # STFT parameters
    >>> win_size = 50                               # Vertical resolution (frequency)
    >>> hop_size = 10                               # Horizontal resolution (time)

    # Build the STFT
    >>> stf = ShortTimeFT()                         # Instantiate the STFT
    >>> win = gaussian(win_size, std=8, sym=True)   # Symmetric Gaussian window
    >>> stf.stft_time(data, win, 10)                # Build the STFT
    >>> pxx = stf.spectrogram()                     # Build the spectrogram
    """

    #---------------------------   Constructor   ----------------------------#
    def __init__(self, **stftparams):
        """ Instantiate an STFT object (constructor)

        Other Parameters
        ----------------
        **stftparams : inline keyword arguments, optional
            The STFT parameters, as the NumPy FFT parameters, and the
            `onesided` and `shift` args. (cf. `fourier.spectrum`).

        Examples
        --------
        # Default parameters
        >>> stf = ShortTimeFT()

        # With parameters for the NumPy's FFT
        >>> stf = ShortTimeFT(n=100, norm='backward')
        """
        self._stftparams = stftparams                       # STFT parameters
        self._freq = np.empty(0, complex)                   # STFT frequencies
        self._stft = np.empty(0, complex)                   # STFT components
        self._specs = {'hop_size': 0, 'win_size': 0, 'samples': 0}    # Specs
    #------------------------------------------------------------------------#

    #-------------------   Magic Methods and Properties   -------------------#
    def __getitem__(self, pos):
        """ Get the STFT value at pos """
        return self._stft[pos]

    def __setitem__(self, pos, value):
        """ Set the STFT value at pos """
        self._stft[pos] = value

    def __iter__(self):
        """ Iterate on the STFT """
        return iter(self._stft)

    def __len__(self):
        """ Length of the STFT """
        return len(self._stft)

    def __repr__(self):
        """ Display the STFT """
        return repr(self._stft)

    def __str__(self):
        """ Print the STFT """
        return "Frequencies:\n" + str(self._freq)\
               + "\nSTFT components:\n" + str(self._stft)

    @property
    def stftparams(self):
        """ Get the STFT parameters """
        return self._stftparams

    @stftparams.setter
    def stftparams(self, stftparams):
        """ Set the STFT parameters """
        self._stftparams = stftparams

    @property
    def freq(self):
        """ Get the STFT frequencies """
        return self._freq

    @freq.setter
    def freq(self, freq):
        """ Set the STFT frequencies """
        self._freq = freq

    @property
    def stft(self):
        """ Get the STFT components """
        return self._stft

    @stft.setter
    def stft(self, stf):
        """ Set the STFT components """
        self._stft = stf

    @property
    def specs(self):
        """ Get the specifications """
        return self._specs

    @specs.setter
    def specs(self, specs):
        """ Set the specifications """
        self._specs = specs
    #------------------------------------------------------------------------#

    #---------------------------   Chunks STFT   ----------------------------#
    def stft_chunks(self, chunks):
        """ Apply the STFT to a set of signal chunks

        Provided a set of data chunks, apply the FFT to each. Set both
        the frequency scale (`freq`) and the STFT (`stft`) attributes.

        See `stft_chunks` function in the same module for details.

        Parameters
        ----------
        chunks : array_like
            The input data chunks (slices).

        Returns
        -------
        None : set the `freq` and `stft` attributes directly.

        Examples
        --------
        >>> import numpy as np
        >>> import aer.data_tk as dtk

        >>> data = np.sin(2*np.pi*5.*np.linspace(0., 1.0, 1000))  # 5 Hz sine

        >>> chunks = dtk.build_chunks(data, 50, 50)
        >>> stf = ShortTimeFT()
        >>> stf.stft_chunks(chunks)
        """
        self._freq, self._stft = stft_chunks(chunks, **self.stftparams)
    #------------------------------------------------------------------------#

    #-------------------   Short-Time Fourier Transform   -------------------#
    def stft_time(self, data, window, hop_size):
        """ Compute the Short-Time Fourier Transform

        Take a data vector (signal), slice it into chunks, multiply them
        by a masking `window` (as argument) and apply the FFT to every
        so-masked signal slice. Set both the normalized frequency scale
        (`freq`) and the short-time Fourier transformed chunks (`stft`).

        The `window` argument must be an array, whose length serves as
        chunk width; it also serves as default FFT resolution.

        See `stft_time` function in the same module for details.

        Parameters
        ----------
        data : np.ndarray
            The data samples to which apply the STFT.
        window : np.ndarray
            The masking window for the STFT. Its width serves as chunk size,
            and is the default FFT resolution (use `n` arg. to modify it).
            The window's size is the spectrogram vertical resolution (freq).
        hop_size : int
            The hop size in samples, i.e. the window offset size.
            The hop's size is the spectrogram horizontal resolution (time).

        Returns
        -------
        None : set the `freq` and `stft` attributes directly.

        Examples
        --------
        >>> import numpy as np
        >>> from scipy.signal.windows import gaussian

        >>> data = np.sin(2*np.pi*5.*np.linspace(0., 1.0, 100))  # 5 Hz sine

        >>> stf = ShortTimeFT(n=50, onesided=False, shift=False)
        >>> stf.stft_time(data, gaussian(100, std=8, sym=True), 10)
        """

        # Save the specifications into the `_specs` attribute
        self._specs = {'hop_size': hop_size,
                       'win_size': len(window),
                       'samples': len(data)}

        # Pad data
        data = dtk.pad(data, dtk.pad_offset(len(data), len(window), hop_size))

        # Build the signal chunks (slices) and mask them with the window
        chunks = dtk.build_chunks(data, len(window), hop_size) * window

        # Build the chunks FT and return the frequency scale and the STFT
        self._freq, self._stft = stft_chunks(chunks, **self.stftparams)
    #------------------------------------------------------------------------#

    #---------------------------   Spectrogram   ----------------------------#
    def spectrogram(self):
        """ Build the spectrogram amplitude of an STFT

        The STFT is the modulus of the STFT, and the resulting 2D vector
        is transposed to get the time as column and frequency as row.

        See `spectrogram` function in the same module for details.

        Parameters
        ----------
        Nothing, only use `self`.

        Returns
        -------
        spec : 2D np.ndarray
            The transposed absolute value (modulus) of the STFT.

        Examples
        --------
        >>> import numpy as np
        >>> from scipy.signal.windows import gaussian

        >>> data = np.sin(2*np.pi*5.*np.linspace(0., 1.0, 100))  # 5 Hz sine

        >>> stf = ShortTimeFT(n=50, onesided=False, shift=False)
        >>> stf.stft_time(data, gaussian(100, std=8, sym=True), 10)
        >>> spec = stf.spectrogram()
        """
        return spectrogram(self._stft)
    #------------------------------------------------------------------------#

##############################################################################



##############################################################################
##                            STFT as Functions                             ##
##############################################################################

def _fftshift(*args, **kwargs):
    """ Used as a reference to dynamically switch between functions """
    return fftshift(fft(*args, **kwargs))

#-----------------------------   Chunks STFT   ------------------------------#
def stft_chunks(chunks, onesided=True, shift=True, **fftparams):
    """ Apply the STFT to a set of signal chunks

    Provided a set of signal chunks, build the STFT by applying the FFT
    to each chunk. Return the (normalized) frequency scale and the STFT.

    Parameters
    ----------
    chunks : array_like
        The input data chunks (slices).
    [OPT] onesided : bool
        Return both positive and negative frequencies if set to False;
        return only the positive frequencies if set to True.
            :Default: True
    [OPT] shift : bool
        Reorganize the frequencies and their components if set to True
        (-0.5 -> +0.5-eps); leave them in their original order else
        (0 -> +0.5-eps -> -0.5 -> -eps). Used only if not `onesided`.
            :Default: True

    Other Parameters
    ----------------
    **fftparams : inline keyword arguments, optional
        The different keyword arguments for the (r)fft functions.
        See `numpy.fft.(r)fft` functions for details.

    Returns
    -------
    freq : np.ndarray
        The normalized frequency scale (should be multiplied by the
        sampling frequency).
    stft : 2D np.ndarray
        The vector of the STFTs (one per chunk).

    Examples
    --------
    >>> import numpy as np
    >>> import aer.data_tk as dtk

    >>> data = np.sin(2*np.pi*5.*np.linspace(0., 1.0, 1000))  # 5 Hz sine

    >>> chunks = dtk.build_chunks(data, 50, 50)
    >>> freq, stft = stft_chunks(chunks, onesided=True, axis=-1)
    >>> freq, stft = stft_chunks(chunks, onesided=False, shift=True, n=100)
    """

    # Nb of samples for the FFT
    if 'n' not in fftparams:                        # If `n` provided,
        fftparams['n'] = chunks.shape[-1]           # add it to the dict
    n = fftparams['n']                              # Get the nb of samples

    # If `onesided` or `twosided` FFT
    if not onesided:                                # Twosided
        func = _fftshift if shift else fft
        freq = fftshift(fftfreq(n)) if shift else fftfreq(n)
    else:                                           # Onesided
        func = rfft
        freq = rfftfreq(n)

    # Build the STFT by applying the FFT to each chunk
    return freq, np.array([func(chunk, **fftparams) for chunk in chunks])
#----------------------------------------------------------------------------#

#---------------------   Short-Time Fourier Transform   ---------------------#
def stft_time(data, window, hop_size, **stftparams):
    """ Compute the Short-Time Fourier Transform

    Take a given data vector (signal), slice it into chunks, multiply
    them by a masking window (taken as argument) and apply the FFT to
    every so-masked signal slice. Return both the normalized frequency
    scale and the so-built Fourier transformed chunks (the STFTs).

    The `window` argument must be an array, whose length serves as chunk
    width when building them; this quantity also serves as default FFT
    resolution, but it can be modified by specifying another resolution
    using the `n` argument (in `stftparams`). Such windows can be hand-
    crafted, or one can use those of the `scipy.signal.windows` module.

    N.B.: this function is less general than the SciPy ShortTimeFFT one,
        but also around 2-2.5 times faster.
            # SciPy STFT
            from scipy.signal import ShortTimeFFT
            SFT = ShortTimeFFT(window, hop=hop_size, fs=frate, mfft=n)
            Sx = SFT.stft(data)

    Parameters
    ----------
    data : np.ndarray
        The data samples to which apply the STFT.
    window : np.ndarray
        The masking window for the STFT. Its width serves as chunk size,
        and is the default FFT resolution (use `n` arg. to modify it).
        The window's size is the spectrogram vertical resolution (freq).
    hop_size : int
        The hop size in samples, i.e. the window offset size.
        The hop's size is the spectrogram horizontal resolution (time).

    Other Parameters
    ----------------
    **stftparams : inline keyword arguments, optional
        The parameters to pass to the `stft_chunks` function; see this
        function for details.

    Returns
    -------
    freq : np.ndarray
        The normalized frequency scale (-0.5 to +0.5-eps).
    stft : 2D np.ndarray
        The signal STFT (i.e. the chunks FT).

    Examples
    --------
    >>> import numpy as np
    >>> from scipy.signal.windows import gaussian

    >>> data = np.sin(2*np.pi*5.*np.linspace(0., 1.0, 100))  # 5 Hz sine

    >>> win = gaussian(100, std=8, sym=True)
    >>> freq, stft = stft_time(data, win, 10, onesided=False, shift=False)
    """

    # Pad data
    data = dtk.pad(data, dtk.pad_offset(len(data), len(window), hop_size))

    # Build the signal chunks (slices) and mask them with the window
    chunks = dtk.build_chunks(data, len(window), hop_size) * window

    # Build the chunks FT and return the frequency scale and the STFT
    return stft_chunks(chunks, **stftparams)
#----------------------------------------------------------------------------#

#-----------------------------   Spectrogram   ------------------------------#
def spectrogram(stf):
    """ Build the spectrogram amplitude of an STFT

    The STFT is the modulus of the STFT, and the resulting 2D vector is
    transposed so as to get the time as column and frequency as row.

    Parameters
    ----------
    stf : 2D np.ndarray
        The set of STFTs, organized as one cell for one chunk FFT.

    Returns
    -------
    spec : 2D np.ndarray
        The transposed absolute value (modulus) of the STFT.

    Examples
    --------
    >>> import numpy as np
    >>> from scipy.signal.windows import gaussian

    >>> data = np.sin(2*np.pi*5.*np.linspace(0., 1.0, 100))  # 5 Hz sine

    >>> freqs, stft = stft_time(data, gaussian(100, std=8, sym=True), 10, n=200)
    >>> spec = spectrogram(stft)
    """
    return np.abs(stf.T) #**2 * / stft.shape[1]
#----------------------------------------------------------------------------#

##############################################################################
