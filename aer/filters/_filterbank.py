""" Filterbank class

Authors: Dylan MOLINIE
Company: CEA Paris-Saclay, Gif-sur-Yvette, France
Contact: dylan.molinie@cea.fr
Date: February 2025
Last revised: March 2026

License: GPLv3
"""

__all__ = ['FilterBank']

import numpy as np

from aer.data_tk import _data_parser as dps
from aer.data_tk import _scaling as scl
from aer.filters import _filters as flt


##############################################################################
##                          Time-Domain Filtering                           ##
##############################################################################

class FilterBank():
    r""" Filterbank class

    Provide a bank of filters object, which allows to build the bank's
    filters, filter a signal (or signal chunks) and its spectrum.

    Attributes
    ----------
    nb_filters : int, getter & setter
        The number of filters.
    bandwidth : 2-tuple of floats, getter & setter
        The filterbank total bandwidth.
    filter_type : function reference, getter & setter
        The filter type (kernel).
    bands : np.ndarray, getter only (use `build_bands` method as setter)
        The filterbank sub-bands.
    filters : np.ndarray, getter only (use `build_filters` method as setter)
        The bank of filters impulse response.

    Constructor
    -----------
    __init__(nb_filters, bandwidth, filter_type='rectangle')

    Magic Methods
    -------------
    __len__()
        Length of the filterbank --> len(fbank).
    __str__()
        Print the filterbank's specs --> print(fbank).

    Methods
    -------
    build_bands(overlap=True, scale='hz')
        Split the bandwidth attribute into sub-bands on a given scale.
    build_filters(tstp, **fltkws)
        Compute the time impulse response of the filters.
    filter_signal(data)
        Filter the signal data with the filters of a bank.
    filter_signal_norm(data)
        Filter the signal and normalize it; see `filter_signal` method
        and `rescale` function from the `data_tk` module for detail.
    filter_signal_chunks(tstp_chunks, data_chunks, norm=True, acc=True, **fltkws)
        Filter a set of signal chunks.
    spectrum(onesided=True, cmplx=False, **fftkws)
        Build the spectrum of the filters with the NumPy FFT.

    Examples
    --------
    >>> import numpy as np
    >>> import aer.data_tk as dtk

    #--- Generate dummy data
    # Signal Parameters
    >>> frate = 200.                            # Sampling frequency
    >>> tstp = np.arange(0, 2, 1./frate)

    >>> bins = 4
    >>> off = len(tstp) // bins
    >>> amps = 2.5*(rng.random(bins)+0.25)
    >>> freqs = 0.5*frate*rng.random(bins)

    >>> def sinus(amp, freq, tstp):
    ...     return amp * np.sin(2.0 * np.pi * freq * tstp)

    >>> pos = 0
    >>> data = np.empty_like(tstp)
    >>> for amp, freq in zip(amps, freqs):
    ...     data[pos:pos+off] = sinus(amp, freq, tstp[pos:pos+off])
    ...     pos += off

    #--- Filter Bank Frequency Response
    # Build the frequency-domain sub-bands on a given scale
    >>> nb_filters = 11
    >>> bandwidth = (0., 0.5*frate)

    # Compute the filters' temporal impulse response
    >>> fbank = FilterBank(nb_filters, bandwidth, 'rect')
    >>> fbank.build_bands(overlap=True, scale='hz')
    >>> fbank.build_filters(tstp, rescale=True, sym=True)

    # Compute the spectral representation of the filters (FFT)
    >>> filters_fft = fbank.spectrum(onesided=True, cmplx=False)

    #--- FB Time Response -- Full Signal
    # Build the frequency-domain sub-bands on a given scale
    >>> nb_filters = 8
    >>> bandwidth = (0., 0.5*frate)

    # Build the filters time response
    >>> fbank = FilterBank(nb_filters, bandwidth, 'rect')
    >>> fbank.build_bands(overlap=True, scale='hz')
    >>> fbank.build_filters(tstp, sym=True, rescale=True)

    # Filter the signal by convolving it with the filters' time response
    >>> signals = 200.*fbank.filter_signal(data)

    #--- FB Time Response -- Signal Chunks
    # Parameters
    >>> nb_filters = 8              # Number of filters of the bank
    >>> win_size = 50               # Vertical resolution (frequency)
    >>> hop_size = 10               # Horizontal resolution (time)
    >>> bandwidth = (0., 0.5*frate) # Filterbank bandwidth

    # Time and data chunks (consecutive overlapping segments)
    >>> tstp_chunks = dtk.build_chunks(tstp, win_size, hop_size)
    >>> data_chunks = dtk.build_chunks(data, win_size, hop_size)

    # Filter the signal chunks with the bank filters
    >>> fbank = FilterBank(nb_filters, bandwidth, 'rect')
    >>> fbank.build_bands(overlap=True, scale='hz')
    >>> data_resp = fbank.filter_signal_chunks(
    ...     tstp_chunks, data_chunks, norm=True, sym=True, rescale=True)
    """

    #---------------------------   Constructor   ----------------------------#
    def __init__(self, nb_filters, bandwidth, filter_type='rectangle'):
        """ Instantiate a FilterBank object (constructor)

        Parameters
        ----------
        nb_filters : int
            Number of filters of the bank.
        bandwidth : 2-tuple of ints
            The total bandwidth of the bank of filters, organized as
            (min_freq, max_freq), typically in Hertz. This bandwidth
            is cut into `nb_filters` regularly spaced intervals, one
            per filter of the bank.
        [OPT] filter_type : str
            The type of the filters, among:
                {'rectangle', 'triangle', 'gammatone'}.
                :Default: 'rectangle'

        Examples
        --------
        # 10 rectangle filters with a total bandwidth between 50 and 500 Hz
        >>> fbank = FilterBank(10, (50, 500), 'rectangle')

        # 100 gammatone filters with a total bandwidth between 0 and 1000 Hz
        >>> fbank = FilterBank(100, (0, 1000), 'gammatone')
        """
        self._nbfilters = nb_filters
        self._bandwidth = bandwidth
        self._ftype = self.__set_ftype(filter_type)
        self._bands = None
        self._filters = None
    #------------------------------------------------------------------------#

    #-------------------   Magic Methods and Properties   -------------------#
    def __len__(self):
        """ Length of the filterbank (number of filters) """
        return self._nbfilters

    def __str__(self):
        """ Print the filterbank's specs """
        return f"Filterbank composed of {self._nbfilters} "\
               + f"'{self.__get_ftype()}' filters, covering the bandwidth "\
               + f"{self._bandwidth[0]}-{self._bandwidth[1]} Hz"

    @property
    def nb_filters(self):
        """ Get the number of filters """
        return self._nbfilters

    @nb_filters.setter
    def nb_filters(self, nb_filters):
        """ Set the number of filters """
        self._nbfilters = nb_filters

    @property
    def bandwidth(self):
        """ Get the bandwidth """
        return self._bandwidth

    @bandwidth.setter
    def bandwidth(self, bandwidth):
        """ Set the bandwidth """
        self._bandwidth = bandwidth

    @property
    def filter_type(self):
        """ Get the filter type (function) """
        return self._ftype

    @filter_type.setter
    def filter_type(self, filter_type):
        """ Set the filter type (function) """
        self._ftype = self.__set_ftype(filter_type)

    @property
    def bands(self):
        """ Get the filterbank sub-bands """
        return self._bands

    @property
    def filters(self):
        """ Get the bank filters impulse response """
        return self._filters
    #------------------------------------------------------------------------#

    #--------------------------   Filter Kernel   ---------------------------#
    def __get_ftype(self):
        """ Get the literal name from filter kernel function """
        if self._ftype is flt.gammatone_time:
            return 'gammatone'
        if self._ftype is flt.triangular_time:
            return 'triangular'
        return 'rectangular'

    def __set_ftype(self, filter_type):
        """ Set the filter kernel function from literal name """
        if filter_type in ('gamma', 'gammatone'):
            return flt.gammatone_time
        if filter_type in ('tg', 'trig', 'triangle', 'triangular'):
            return flt.triangular_time
        return flt.rectangular_time
    #------------------------------------------------------------------------#

    #-----------------------   Frequency Sub-Bands   ------------------------#
    def build_bands(self, overlap=True, scale='hz'):
        """ Split the bandwidth attribute into sub-bands on a given scale

        Take a scale (Hertz, Melody, Bark), linearly split the frequency
        interval contained in the `bandwidth` attribute on this scale,
        and project back these frequency boundaries onto the Hz scale.

        See the `sub_bands` function in `data_tk` package.

        Parameters
        ----------
        [OPT] overlap : bool
            If the sub-bands should overlap (in their center) or not.
                :Default: True
        [OPT] scale : string (case insensitive)
            The scale to use; valid options are: Melody ('mel' or 'melody'),
            Bark ('bk' or 'bark') and Hertz (anything else).
                :Default: 'hz' (Hertz scale)

        Returns
        -------
        None : directly save the sub-bands in the `bands` attribute.

        Examples
        --------
        >>> fbank = FilterBank(4, (10, 30), 'rect')
        >>> fbank.build_bands(overlap=False, scale='hz')
        >>> print(fbank.bands.T)
        [[10. 15. 20. 25.]
         [15. 20. 25. 30.]]
        """
        self._bands = scl.sub_bands(self._bandwidth, self._nbfilters, overlap, scale)
    #------------------------------------------------------------------------#

    #---------------------   Filters Impulse Response   ---------------------#
    def build_filters(self, tstp, **fltkws):
        """ Compute the time impulse response of the filters

        Take a set of timestamps and build the bank filters time impulse
        response for each sub-band of the `bands` attribute.

        The sub-bands are assumed to be set; as such, make sure to call
        the `build_bands` method prior using the present one.

        See the original filtering functions for details.

        Parameters
        ----------
        tstp : np.ndarray
            The timestamps to filter with the gate filter.

        Other Parameters
        ----------------
        **fltkws : inline keyword arguments, optional
            The parameters of the filtering function, if any. See the
            corresponding function in the `filters` module for details.

        Returns
        -------
        None : directly save the filters time impulse response in the
            `filters` attribute, as a 2D np.ndarray.

        Examples
        --------
        >>> import numpy as np

        >>> fbank = FilterBank(4, (10, 30), 'rect')
        >>> fbank.build_bands(overlap=False, scale='hz')
        >>> fbank.build_filters(np.linspace(0, 2, 600))
        """

        # Check if the 'bands' have been built
        if self._bands is None:
            print("No sub-bands built; please run 'build_bands' first")
            return

        # Compute the impulse response of the filters
        self._filters = np.array(
            [self._ftype(tstp, bd, **fltkws) for bd in self._bands], float)
#        self._filters = np.empty((self._nbfilters, len(tstp)), float)
#        for i, band in enumerate(self._bands):
#            self._filters[i] = self._ftype(tstp, band, **fltkws)
    #------------------------------------------------------------------------#

    #-------------------------   Signal Filtering   -------------------------#
    def filter_signal(self, data):
        """ Filter the signal data with the filters of a bank

        Convolve a set of data samples with the impulse response of each
        filter of the bank, so as to build the signal filtered versions.

        See `convolve` function from `filters` module for details.

        N.B.: the function assumes that the `filters` have been built
            prior; use `build_filters` function is not.

        Parameters
        ----------
        data : 1D np.ndarray
            The TS data to filter.

        Returns
        -------
        signal : 2D np.ndarray
            The amplitudes of signal data filtered by each bank filter.

        Examples
        --------
        >>> import numpy as np

        >>> tstp = np.linspace(0., 10., 1000)
        >>> freq = np.array([100., 200.]) * 2*np.pi
        >>> data = np.sin(tstp*freq[0]) + np.sin(tstp*freq[1])

        >>> fbank = FilterBank(4, (10, 30), 'rect')
        >>> fbank.build_bands(overlap=False, scale='hz')
        >>> fbank.build_filters(np.linspace(0, 2, 600))

        >>> signals = fbank.filter_signal(data)
        """

        # Check if the 'filters' have been built
        if self._filters is None:
            print("No filters built; please run `build_filters` first")
            return None

        # Filter the signal data with the bank filters
        signals = np.empty((self._nbfilters, len(data)), float)
        for i, filt in enumerate(self._filters):
            signals[i] = flt.convolve(data, filt)
        return signals

    def filter_signal_norm(self, data):
        """ Filter the signal and normalize it; see `filter_signal` method
            and `rescale` function from the `data_tk` module for detail """
        return scl.rescale(self.filter_signal(data), None)[0]
    #------------------------------------------------------------------------#

    #-------------------------   Chunks Filtering   -------------------------#
    def _fchunk_acc(self, tstp_chunks, data_chunks, norm=True, **fltkws):
        """ Filter the chunks and accumulate the responses """

        # Retrieve the chunks parameters
        win_size = np.shape(tstp_chunks)[1]
        hop_size = dps.get_hop_size(tstp_chunks)
        vec_size = win_size + hop_size*(len(tstp_chunks)-1)

        # Cumulative response (though the overlapping windows, if so)
        data_resp = np.zeros((self.nb_filters, vec_size), float)    # Response
        data_norm = np.zeros(vec_size, float)                       # Norm

        # Select the filtering function (that applies normalization or not)
        sfunc = self.filter_signal_norm if norm else self.filter_signal

        # Filter every signal chunk and accumulate the overlapping responses
        pos = 0
        for tsp, dat in zip(tstp_chunks, data_chunks):
            self.build_filters(tsp, **fltkws)             # Build the bands
            data_resp[:, pos:pos+win_size] += sfunc(dat)  # Filters response
            data_norm[pos:pos+win_size] += 1.0            # Cumulative norm
            pos += hop_size                               # Next chunk

        return data_resp * np.reciprocal(data_norm)       # Accumulation norm

    def _fchunk_noacc(self, tstp_chunks, data_chunks, norm=True, **fltkws):
        """ Filter the chunks and stack the responses in a matrix """

        # Cumulative response (though the overlapping windows, if so)
        shape = np.shape(tstp_chunks)
        data_resp = np.empty((shape[0], self.nb_filters, shape[1]), float)

        # Select the filtering function (that applies normalization or not)
        sfunc = self.filter_signal_norm if norm else self.filter_signal

        # Filter every signal chunk and store them in `data_resp`
        for i, (tsp, dat) in enumerate(zip(tstp_chunks, data_chunks)):
            self.build_filters(tsp, **fltkws)             # Build the bands
            data_resp[i] = sfunc(dat)                     # Filters response

        return data_resp

    def filter_signal_chunks(self,
        tstp_chunks, data_chunks, norm=True, acc=True, **fltkws):
        """ Filter a set of signal chunks

        Take a set of timestamps chunks and a set of data chunks, and
        apply the bank's filters to every pair of timestamps-data chunks
        (assumed to correspond in their position, i.e. `tstp_chunks[i]`
        are the timestamps of the data of `data_chunks[i]`). If `norm`,
        normalize the filters responses for each chunk (normalization
        is operated within a given chunk independently from the others).

        If not `acc`, store the responses of the bank's filters in a
        3D matrix of size `nb_chunks x nb_filters x win_size`, where
        `nb_chunks` is the number of chunks, `nb_filters` is the nb
        of filters and `win_size` is the overlapping window size.
        Otherwise, if `acc`, accumulate the bank's filters responses in
        a unique 2D vector, in which the responses to overlapping data
        (i.e. those present in several, successive chunks) are summed;
        then the summed data are divided (normalized) by the number of
        times they appeared (and thus have been filtered and summed).

        Parameters
        ----------
        tstp_chunks : 2D np.ndarray
            The timestamps chunks, organized as `nb_chunks x chunks`,
            with `chunks` the chunks of size `win_size`.
        data_chunks : 2D np.ndarray
            The data chunks; same as `tstp_chunks` but for data samples.
        [OPT] norm : bool
            If the filters response should be normalized (T) or not (F).
                :Default: True
        [OPT] acc : bool
            If the filters responses should be accumulated in a unique
            vector, or if they should be stacked in a unique matrix.

        Other Parameters
        ----------------
        **fltkws : inline keyword arguments, optional
            The parameters of the filtering function, if any. See the
            corresponding function in the `filters` module for details.

        Returns
        -------
        responses : 2D or 3D np.ndarray, depending on `acc` argument.
            The bank's filters responses to the chunks; if `acc`, the
            overlapping responses (corresponding to overlapping data)
            are averaged, leading to a 2D array; else, all the responses
            are stacked in a unique 3D matrix.

        Examples
        --------
        >>> import numpy as np
        >>> import data_tk as dtk

        # Parameters
        >>> nb_filters = 15         # Number of filters of the bank
        >>> win_size = 50           # Vertical resolution (frequency)
        >>> hop_size = 10           # Horizontal resolution (time)
        >>> freq = 10
        >>> frate = 600

        # Data
        >>> tstp = np.arange(0., 2., 1/frate)
        >>> data = np.sin(2*np.pi*freq*tstp) + np.sin(2*np.pi*5*freq*tstp)
        >>> bandwidth = (0, frate/2)

        # Time and data chunks (consecutive overlapping segments)
        >>> tstp_chunks = dtk.build_chunks(tstp, win_size, hop_size)
        >>> data_chunks = dtk.build_chunks(data, win_size, hop_size)

        # Filter the signal chunks with the bank filters
        >>> fbank = FilterBank(nb_filters, bandwidth, 'rect')
        >>> fbank.build_bands(overlap=True, scale='hz')
        >>> resps = fbank.filter_signal_chunks(tstp_chunks, data_chunks)
        """
        if acc:
            return self._fchunk_acc(tstp_chunks, data_chunks, norm, **fltkws)
        return self._fchunk_noacc(tstp_chunks, data_chunks, norm, **fltkws)
    #------------------------------------------------------------------------#

    #-------------------------   Filters Spectrum   -------------------------#
    def spectrum(self, onesided=True, cmplx=False, **fftkws):
        """ Build the spectrum of the filters with the NumPy FFT

        Compute the FFT of the impulse response of the bank filters.

        N.B.: the function assumes that the `filters` have been built
            prior; use `build_filters` function is not.

        Parameters
        ----------
        [OPT] onesided : bool
            Use both positive and negative frequencies if False; use
            only the positive frequencies if True.
                :Default: True
        [OPT] cmplx : bool
            Keep both the real and imag parts of the FFT if True; keep
            only the real part is False.
                :Default: False (real part only)

        Other Parameters
        ----------------
        **fftkws : inline keyword arguments, optional
            The parameters of the FFT function, if any.

        Returns
        -------
        filters_fft : 2D np.ndarray
            The amplitudes of the filters spectra.

        Examples
        --------
        >>> import numpy as np

        >>> fbank = FilterBank(4, (10, 30), 'rect')
        >>> fbank.build_bands(overlap=False, scale='hz')
        >>> fbank.build_filters(np.linspace(0, 2, 600))
        >>> filters_fft = fbank.spectrum(False, True)
        """

        # Check if the `filters` have been built
        if self._filters is None:
            print("No filters built; please run `build_filters` first")
            return None

        # If `onesided`, use the right FFT (only positive frequencies)
        if onesided:
            fft_fct = np.fft.rfft
            shape = (self._nbfilters, self._filters.shape[1]//2+1)
        else:
            fft_fct = np.fft.fft
            shape = self._filters.shape

        # Keep both real and imag parts of the FFT, or only the real one
        if cmplx:                       # Keep both FFT real and imag parts
            filters_fft = np.empty(shape, complex)
            for i, filt in enumerate(self._filters):
                filters_fft[i] = fft_fct(filt, **fftkws)
        else:                           # Keep only the FFT real part
            filters_fft = np.empty(shape, float)
            for i, filt in enumerate(self._filters):
                filters_fft[i] = fft_fct(filt, **fftkws).real

        return filters_fft
    #------------------------------------------------------------------------#

##############################################################################
