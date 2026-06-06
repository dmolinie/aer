""" Shorthand functions to display results

Authors: Dylan MOLINIE
Company: CEA Paris-Saclay, Gif-sur-Yvette, France
Contact: dylan.molinie@cea.fr
Date: April 2025
Last revised: April 2026

License: GPLv3
"""
# pylint: disable=too-many-arguments, too-many-positional-arguments

__all__ = [
    'freqs2str', 'signal', 'spectrum', 'cepstrum',
    'spectralpower', 'cepstralpower', 'spectrogram',
    'filterbank', 'filterbank_spectrum']

import numpy as np
import matplotlib.pyplot as plt

from aer.tools import check_keys, check_path
from aer.plot._plot import plot
from aer.plot._core_plot import plot_core
from aer.plot._decorations import set_margins, remove_spaces, set_decorations


##############################################################################
##                              Miscellaneous                               ##
##############################################################################

#------------------------   Frequencies Formatting   ------------------------#
def freqs2str(freqs, rnd=1):
    """ Convert a list of scalar frequencies as a literal

    Take a list of scalars, convert them as literals, add the "f_{i} = "
    string before and " Hz" after, and stack them horizontally. Let con-
    sider the list [12.001, 35.308, 9.159], the function will return:
        "f_{0} = 12.001 Hz, f_{1} = 35.308 Hz, f_{2} = 9.159 Hz"
    Notice also that the array of scalars is rounded before conversion;
    the round-off precision can be specified by the `rnd` argument.

    Parameters
    ----------
    freqs : list of scalars
        The frequencies to format as literal.
    [OPT] rnd : int
        The round-off precision.
            :Default: 1

    Returns
    -------
    string : str
        The frequencies formatted and transcribed into literals.

    Examples
    --------
    >>> freqs2str([1.123])
    '$f_{0}$ = 1.1 Hz'

    >>> freqs2str([1.123, 2.456, 3.789])
    '$f_{0}$ = 1.1 Hz, $f_{1}$ = 2.5 Hz, $f_{2}$ = 3.8 Hz'

    >>> freqs2str([1.123456789, 2.456789123, 3.789123456], 3)
    '$f_{0}$ = 1.123 Hz, $f_{1}$ = 2.457 Hz, $f_{2}$ = 3.789 Hz'
    """
    string = ''
    for i, freq in enumerate(np.round(freqs, rnd)):
        string += "$f_{" + str(i) + "}$ = " + str(freq) + " Hz\t"
    return string[:-1]
#----------------------------------------------------------------------------#

##############################################################################



##############################################################################
##                        Signal Plotting Functions                         ##
##############################################################################

def _build_figure(imgparams):
    """ Generate the figure """
    if 'fig_params' in imgparams:
        return plt.figure(**imgparams['fig_params'])
    return plt.figure()

#--------------------------   Time Domain Signal   --------------------------#
def signal(tstp, sig,
    labels=("Time [s]", "Amplitude [a.u.]"),
    title="Input Signal Amplitude Against Time",
    legends=None, rtext=None, **imgparams):
    """ Plot a time-domain signal against time

    Generate a 1-plot figure and plot the `sig` data (y-values) against
    `tstp` (x-values). To this end, pass the so-generated figure and all
    the function's arguments to the `plot` module's `plot` function.

    If `fname` keyword argument is provided, save the figure in a file
    with that name, and with parameters the `save_params` keyword args.

    N.B.: refer to the `plot` module's `plot` function for details.

    Parameters
    ----------
    tstp : array_like
        The signal timestamps (x-values).
    sig : array_like
        The real-valued signal data (y-values).
    [OPT] labels : 2-tuple of strings
        The x-axis & y-axis labels as (x-label, y-label). Use None to skip.
            :Default: ("Time [s]", "Amplitude [a.u.]")
    [OPT] title : string
        The plot title. Set to None to ignore.
            :Default: "Input Signal Amplitude Against Time"
    [OPT] legends : set of strings
        The plot legends, if any.
            :Default: None
    [OPT] rtext : string
        The right y-axis side text, if any.
            :Default: None

    Other Parameters
    ----------------
    **imgparams : inline keyword arguments, optional
        The sets of option keyword arguments for the `plot` module's
        `plot` function; refer to this function for details.

    Returns
    -------
    fig : plt's Figure
        The drawn figure. Display it with `fig.show()`.

    Examples
    --------
    >>> import numpy as np

    # Generate dummy data
    >>> tstp = np.linspace(0., 1.)
    >>> sine = np.sin(6.28*10.*tstp)

    # Plot the signal against time
    >>> fig = signal(tstp, sine,
    ...     fig_params={'figsize': (19.20, 10.80)},
    ...     plot_params={'fmt': '--'})
    >>> fig.show()
    """
    # Generate the figure
    fig = _build_figure(imgparams)
    # Plot the data on the figure
    return plot(fig.gca(), tstp, sig,
        labels=labels, titles=title, legends=legends, rtexts=rtext, **imgparams)
#----------------------------------------------------------------------------#

#-------------------------------   Spectrum   -------------------------------#
def spectrum(freq, sfft,
    labels=("Frequency [Hz]", "Amplitude [a.u.]"),
    title="Signal Spectrum Against Frequency",
    legends=("Real part", "Imaginary part"),
    rtext=None, **imgparams):
    """ Plot the spectrum (Fourier Transform) of a signal

    Generate a 1-plot figure and plot the `sfft` data (y-values) against
    `freq` (x-values). To this end, pass the so-generated figure and all
    the function's arguments to the `plot` module's `plot` function.

    If `fname` keyword argument is provided, save the figure in a file
    with that name, and with parameters the `save_params` keyword args.

    N.B.: refer to the `plot` module's `plot` function for details.

    Parameters
    ----------
    freq : array_like
        The spectrum frequencies (x-values).
    sfft : array_like
        The real- or complex-valued signal spectrum (y-values).
    [OPT] labels : 2-tuple of strings
        The x-axis & y-axis labels as (x-label, y-label). Use None to skip.
            :Default: ("Frequency [Hz]", "Amplitude [a.u.]")
    [OPT] title : string
        The plot title. Set to None to ignore.
            :Default: "Signal Spectrum Against Frequency"
    [OPT] legends : set of strings
        The plot legends, if any.
            :Default: ("Real part", "Imaginary part")
    [OPT] rtext : string
        The right y-axis side text, if any.
            :Default: None

    Other Parameters
    ----------------
    **imgparams : inline keyword arguments, optional
        The sets of option keyword arguments for the `plot` module's
        `plot` function; refer to this function for details.

    Returns
    -------
    fig : plt's Figure
        The drawn figure. Display it with `fig.show()`.

    Examples
    --------
    >>> import numpy as np
    >>> from aer.spectrum import Spectrum

    # Generate a simple sum of a 10 Hz & a 50 Hz sines
    >>> frate = 200     # Frequency rate
    >>> tps = np.linspace(0., 1., frate)
    >>> sig = np.sin(6.28*10.*tps) + np.sin(6.28*50.*tps)

    # Compute the signal spectrum
    >>> spect = Spectrum()
    >>> spect.spectrum(sig, True, frate=frate)

    # Plot the signal spectrum
    >>> fig = spectrum(spect.freq, spect.sfft,
    ...     fig_params={'figsize': (19.20, 10.80)})
    >>> fig.show()
    """
    # Generate the figure
    fig = _build_figure(imgparams)
    # Plot the spectrum on the figure
    return plot(fig.gca(), freq, sfft,
        labels=labels, titles=title, legends=legends, rtexts=rtext, **imgparams)
#----------------------------------------------------------------------------#

#-------------------------------   Cepstrum   -------------------------------#
def cepstrum(quef, ceps,
    labels=(r"Quefrency [$\emptyset$]", "Amplitude [a.u.]"),
    title="Signal Cepstrum Against Quefrency",
    legends=("Real part", "Imaginary part"),
    rtext=None, **imgparams):
    r""" Plot the cepstrum of a signal

    Generate a 1-plot figure and plot the `ceps` data (y-values) against
    `quef` (x-values). To this end, pass the so-generated figure and all
    the function's arguments to the `plot` module's `plot` function.

    If `fname` keyword argument is provided, save the figure in a file
    with that name, and with parameters the `save_params` keyword args.

    N.B.: refer to the `plot` module's `plot` function for details.

    Parameters
    ----------
    quef : array_like
        The cepstrum quefrencies (x-values).
    ceps : array_like
        The real- or complex-valued signal cepstrum (y-values).
    [OPT] labels : 2-tuple of strings
        The x-axis & y-axis labels as (x-label, y-label). Use None to skip.
            :Default: (r"Quefrency [$\emptyset$]", "Amplitude [a.u.]")
    [OPT] title : string
        The plot title. Set to None to ignore.
            :Default: "Signal Cepstrum Against Quefrency"
    [OPT] legends : set of strings
        The plot legends, if any.
            :Default: ("Real part", "Imaginary part")
    [OPT] rtext : string
        The right y-axis side text, if any.
            :Default: None

    Other Parameters
    ----------------
    **imgparams : inline keyword arguments, optional
        The sets of option keyword arguments for the `plot` module's
        `plot` function; refer to this function for details.

    Returns
    -------
    fig : plt's Figure
        The drawn figure. Display it with `fig.show()`.

    Examples
    --------
    >>> import numpy as np
    >>> from aer.spectrum import Cepstrum

    # Generate a simple sum of a 10 Hz & a 50 Hz sines
    >>> frate = 200     # Frequency rate
    >>> tps = np.linspace(0., 1., frate)
    >>> sig = np.sin(6.28*10.*tps) + np.sin(6.28*50.*tps)

    # Compute the signal spectrum & cepstrum
    >>> cepst = Cepstrum()
    >>> cepst.spectrum(sig, True, frate=frate)  # Spectrum
    >>> cepst.cepstrum()                        # Cepstrum

    # Plot the signal cepstrum
    >>> fig = cepstrum(cepst.quef, cepst.ceps,
    ...     fig_params={'figsize': (19.20, 10.80)})
    >>> fig.show()
    """
    # Generate the figure
    fig = _build_figure(imgparams)
    # Plot the cepstrum on the figure
    return plot(fig.gca(), quef, ceps,
        labels=labels, titles=title, legends=legends, rtexts=rtext, **imgparams)
#----------------------------------------------------------------------------#

#----------------------------   Spectrum Power   ----------------------------#
def spectralpower(freq, pxx,
    labels=("Frequency [Hz]", "Power Density [a.u.]"),
    title="Spectral Power Density (Periodogram)",
    legends=None, rtext=None, **imgparams):
    """ Plot the spectral power density (periodogram) of a signal

    Generate a 1-plot figure and plot the `pxx` data (y-values) against
    `freq` (x-values). To this end, pass the so-generated figure and all
    the function's arguments to the `plot` module's `plot` function.

    If `fname` keyword argument is provided, save the figure in a file
    with that name, and with parameters the `save_params` keyword args.

    N.B.: refer to the `plot` module's `plot` function for details.

    Parameters
    ----------
    freq : array_like
        The spectrum frequencies (x-values).
    pxx : array_like
        The real-valued power spectrum (y-values).
    [OPT] labels : 2-tuple of strings
        The x-axis & y-axis labels as (x-label, y-label). Use None to skip.
            :Default: ("Frequency [Hz]", "Power Density [a.u.]")
    [OPT] title : string
        The plot title. Set to None to ignore.
            :Default: "Input Signal Amplitude Against Time"
    [OPT] legends : set of strings
        The plot legends, if any.
            :Default: None
    [OPT] rtext : string
        The right y-axis side text, if any.
            :Default: None

    Other Parameters
    ----------------
    **imgparams : inline keyword arguments, optional
        The sets of option keyword arguments for the `plot` module's
        `plot` function; refer to this function for details.

    Returns
    -------
    fig : plt's Figure
        The drawn figure. Display it with `fig.show()`.

    Examples
    --------
    >>> import numpy as np
    >>> from aer.spectrum import Spectrum

    # Generate a simple sum of a 10 Hz & a 50 Hz sines
    >>> frate = 200     # Frequency rate
    >>> tps = np.linspace(0., 1., frate)
    >>> sig = np.sin(6.28*10.*tps) + np.sin(6.28*50.*tps)

    # Compute the signal spectrum
    >>> spect = Spectrum()
    >>> spect.spectrum(sig, True, frate=frate)
    >>> pxx = spect.spectral_power()

    # Plot the signal spectrum
    >>> fig = spectralpower(spect.freq, pxx,
    ...     fig_params={'figsize': (19.20, 10.80)})
    >>> fig.show()
    """
    # Generate the figure
    fig = _build_figure(imgparams)
    # Plot the spectral power on the figure
    return plot(fig.gca(), freq, pxx,
        labels=labels, titles=title, legends=legends, rtexts=rtext, **imgparams)
#----------------------------------------------------------------------------#

#----------------------------   Cepstrum Power   ----------------------------#
def cepstralpower(quef, cxx,
    labels=(r"Quefrency [$\emptyset$]", "Power Density [a.u.]"),
    title="Cepstral Power Density",
    legends=None, rtext=None, **imgparams):
    r""" Plot the cepstral power density of a signal

    Generate a 1-plot figure and plot the `cxx` data (y-values) against
    `quef` (x-values). To this end, pass the so-generated figure and all
    the function's arguments to the `plot` module's `plot` function.

    If `fname` keyword argument is provided, save the figure in a file
    with that name, and with parameters the `save_params` keyword args.

    N.B.: refer to the `plot` module's `plot` function for details.

    Parameters
    ----------
    quef : array_like
        The cepstrum quefrencies (x-values).
    cxx : array_like
        The real-valued power cepstrum (y-values).
    [OPT] labels : 2-tuple of strings
        The x-axis & y-axis labels as (x-label, y-label). Use None to skip.
            :Default: (r"Quefrency [$\emptyset$]", "Power Density [a.u.]")
    [OPT] title : string
        The plot title. Set to None to ignore.
            :Default: "Cepstral Power Density"
    [OPT] legends : set of strings
        The plot legends, if any.
            :Default: None
    [OPT] rtext : string
        The right y-axis side text, if any.
            :Default: None

    Other Parameters
    ----------------
    **imgparams : inline keyword arguments, optional
        The sets of option keyword arguments for the `plot` module's
        `plot` function; refer to this function for details.

    Returns
    -------
    fig : plt's Figure
        The drawn figure. Display it with `fig.show()`.

    Examples
    --------
    >>> import numpy as np
    >>> from aer.spectrum import Cepstrum

    # Generate a simple sum of a 10 Hz & a 50 Hz sines
    >>> frate = 200     # Frequency rate
    >>> tps = np.linspace(0., 1., frate)
    >>> sig = np.sin(6.28*10.*tps) + np.sin(6.28*50.*tps)

    # Compute the signal spectrum, cepstrum and power cepstrum
    >>> cepst = Cepstrum()
    >>> cepst.spectrum(sig, True, frate=frate)  # Spectrum
    >>> cepst.cepstrum()                        # Cepstrum
    >>> cxx = cepst.cepstral_power()            # Cepstral power

    # Plot the signal power cepstrum
    >>> fig = cepstralpower(cepst.quef, cxx,
    ...     fig_params={'figsize': (19.20, 10.80)})
    >>> fig.show()
    """
    # Generate the figure
    fig = _build_figure(imgparams)
    # Plot the cepstral power on the figure
    return plot(fig.gca(), quef, cxx,
        labels=labels, titles=title, legends=legends, rtexts=rtext, **imgparams)
#----------------------------------------------------------------------------#

#------------------------------   Plot STFT   -------------------------------#
def spectrogram(spect, freqs=None,
    labels=("Time [s]", "Frequency [Hz]"),
    title="Signal spectrogram (STFT)",
    **imgparams):
    """ Plot the spectrogram of a signal as a 2D image with time as
    abscissa and frequency as ordinates

    Generate a 1-plot figure and plot the 2D `spect` data as an image.
    See the `plot` module's `set_labels` function for details about the
    decorators (labels & title).

    If `fname` keyword argument is provided, save the figure in a file
    with that name, and with parameters the `save_params` keyword args.

    Parameters
    ----------
    spect : 2D np.ndarray
        The signal spectrogram (time as x-values and freq as y-values).
    [OPT] freqs : array of scalars
        If provided, extract, format and write the frequencies of the
        signal at the bottom of the figure; do nothing if None.
            :Default: None
    [OPT] labels : 2-tuple of strings
        The x-axis & y-axis labels as (x-label, y-label). Use None to skip.
            :Default: (f"Time [s] ({spect.shape[1]} slices)",
                       f"Frequency [Hz] ({spect.shape[0]} bins)")
    [OPT] title : string
        The plot title. Set to None to ignore.
            :Default: "Signal spectrogram (STFT)"

    Other Parameters
    ----------------
    **imgparams : inline keyword arguments, optional
        The sets of option keyword arguments for the `plot` module's
        `plot` function; refer to this function for details.

    Other Parameters
    ----------------
    [OPT] fname : str
        The image-to-save pathname (path+name+ext).
            :Default: None (do not save the figure)
    [OPT] fig_params : dict
        The figure parameters passed to the Matplotlib's `subplots` function:
          - `figsize` (2-tuple of floats): the figure dimensions
          - `layout` (str): layout mechanism for plot positioning
            :Default: {'figsize': (19.20, 10.80), layout='constrained'}
    [OPT] save_params : dict
        The parameters to save the figure, passed to the Matplotlib's
        `savefig`; used only if `fname` is not empty nor None:
          - `bbox_inches` (str): the margins of the saved graphic image
          - `dpi` (int): image DPI (used if saved as a raster graphic)
            :Default: {'bbox_inches': 'tight', 'dpi': 100}
    [OPT] imshow_params : dict
        The image parameters passed to Matplotlib's `Axis`' `imshow`:
          - `aspect` (str): the aspect ratio of the Axes; prefer 'auto'
          - `origin` (str): draw upwards ('lower') or downwards ('upper')
          - `extent` (4-tuple of floats): x and y limits organized as:
                (xmin, xmax, ymin, ymax)
          - `cmap` (str): color map, e.g. 'viridis', 'plasma', 'magma'
            :Default: {'cmap': 'viridis', 'origin': 'lower',
                       'aspect': 'auto', 'extent': (0., 1., 0., 1.)}
    [OPT] colorbar_params : dict
        The color bar parameters passed to Matplotlib's `Figure`'s `colorbar`:
          - `label` (string): the color bar side label
            :Default: {'label': 'Magnitude [a.u.]'}
    [OPT] label_params : dict
        The labels parameters passed to the `set_labels` function from
        the `plot` module:
          - `size` (float): x and y labels font size
          - `labelpad` (float): x and y labels bounding box space pad
            :Default: {'size': 18}
    [OPT] title_params : dict
        The titles parameters passed to the `set_titles` function from
        the `plot` module:
          - `size` (float): title font size
          - `pad` (float): title bounding box space pad
            :Default: {'size': 21}
    [OPT] figtext_params : dict
        The bottom text parameters passed to Matplotlib's `figtext`:
          - `fontsize` (float): font size
          - `ha`, `va` (str): horizontal/vertical alignment
            :Default: {'ha': 'center', 'va': 'center', 'fontsize': 14,
                       'bbox': {'facecolor': 'whitesmoke',
                                'edgecolor': 'gray', 'boxstyle': 'round'}}

    Returns
    -------
    fig : plt's Figure
        The drawn figure. Display it with `fig.show()`.

    Examples
    --------
    >>> import numpy as np
    >>> from scipy.signal.windows import gaussian
    >>> from aer.spectrum import ShortTimeFT

    # Generate a simple signal
    >>> fs = 1000.                               # Sampling frequency
    >>> tstp = 2*np.pi*np.arange(0., 1., 1./fs)  # Time scale
    >>> sin1 = np.sin(50*tstp)                   # 50 Hz sine
    >>> sin2 = 3.*np.sin(250*tstp)               # 250 Hz sine
    >>> noise = 2.*np.random.random(len(tstp))   # Random noise

    # Build the spectrogram of the signal
    >>> win = gaussian(50, std=8, sym=True)      # Symmetric Gaussian window
    >>> stf = ShortTimeFT()                      # Instantiate the STFT
    >>> stf.stft_time(sin1+sin2+noise, win, 10)  # Build the STFT

    # Plot the spectrogram
    >>> fig = spectrogram(
    ...     stf.spectrogram(), (50, 250),
    ...     fname="Spectrogram.pdf", save_params={'dpi': 100},
    ...     fig_params={'figsize': (19.20, 10.80), 'layout': 'constrained'},
    ...     imshow_params={'cmap': 'magma', 'origin': 'lower', 'aspect': 'auto',
    ...                    'extent': (0.0, len(tstp)/fs, 0.0, fs/2)},
    ...     colorbar_params={'label': "Magnitude [a.u.]"},
    ...     label_params={'size': 18}, title_params={'size': 21},
    ...     figtext_params={'fontsize': 18})
    >>> fig.show()
    """

    # Default parameters
    params = {'fname': None, 'save_params': {'bbox_inches': 'tight'},
              'fig_params': {'figsize': (19.20, 10.80), 'layout': 'constrained'},
              'imshow_params': {'cmap': 'viridis', 'origin': 'lower',
                                'aspect': 'auto', 'extent': (0., 1., 0., 1.)},
              'colorbar_params': {'label': 'Magnitude [a.u.]'},
              'label_params': {'size': 18}, 'title_params': {'size': 21},
              'figtext_params': {'ha': 'center', 'va': 'center', 'fontsize': 14,
                  'bbox': {'facecolor': 'whitesmoke', 'edgecolor': 'gray',
                           'boxstyle': 'round'}}}

    # Append the default parameters to those possibly provided as arguments
    params = check_keys(imgparams, params)

    # Create the figure
    fig, axis = plt.subplots(**params['fig_params'])

    # Plot the image
    img = axis.imshow(spect, **params['imshow_params'])
    fig.colorbar(img, **params['colorbar_params'])

    # Add the title and the axes labels
    set_decorations(axis, labels, title,
        label_params=params['label_params'],
        title_params=params['title_params'])

    # If provided, add the frequencies at the bottom of the figure
    if freqs is not None:
        plt.figtext(0.46, -0.035, freqs2str(freqs, 1), **params['figtext_params'])

    # Save the figure into the specified file
    # If the filename contains directories, create them if they do not exist
    if params['fname'] not in (None, ''):
        plt.savefig(check_path(params['fname']), **params['save_params'])

    return fig
#----------------------------------------------------------------------------#

##############################################################################



##############################################################################
##                               Filterbanks                                ##
##############################################################################

#----------------------   Filter Frequency Response   -----------------------#
def filterbank_spectrum(freq, hfft,
    labels=("Frequency [Hz]", "Amplitude [a.u.]"),
    title="Frequency response of the filters of the bank",
    legends=None, rtexts=None, **imgparams):
    """ Plot the frequency response of the bank's filters

    Generate an N-plot figure and plot the `hfft` data (y-values) against
    `freq` (x-values). To this end, pass the so-generated figure and all
    the function's arguments to the `plot` module's `plot` function. The
    parameters for the image can be passed through the `fig_params` arg.

    If `fname` keyword argument is provided, save the figure in a file
    with that name, and with parameters the `save_params` keyword args.

    N.B.: refer to the `plot` module's `plot` function for details.

    Parameters
    ----------
    freq : (array of) array_like
        The spectrum frequencies (x-values).
    hfft : (array of) array_like
        The real-valued filter spectrums (y-values).
    [OPT] labels : 2-tuple of strings
        The x-axis & y-axis labels as (x-label, y-label). Use None to skip.
            :Default: ("Frequency [Hz]", "Amplitude [a.u.]")
    [OPT] title : string
        The plot title. Set to None to ignore.
            :Default: "Frequency response of the filters of the bank"
    [OPT] legends : set of strings
        The plot legends, if any.
            :Default: None
    [OPT] rtexts : string
        The right y-axis side texts, if any.
            :Default: None

    Other Parameters
    ----------------
    **imgparams : inline keyword arguments, optional
        The sets of option keyword arguments for the `plot` module's
        `plot` function; refer to this function for details.

    Returns
    -------
    fig : plt's Figure
        The drawn figure. Display it with `fig.show()`.

    Examples
    --------
    >>> import numpy as np
    >>> from aer.filters import FilterBank

    # Generate dummy data
    >>> frate = 1000.
    >>> tstp = np.linspace(0., 1., int(frate))
    >>> freqs = (10, 50, 100)
    >>> data = np.zeros_like(tstp)
    >>> for freq in freqs:
    ...     data += np.sin(6.28*freq*tstp)

    # Build the frequency-domain sub-bands on a given scale
    >>> nb_filters = 11
    >>> bandwidth = (0., 0.5*frate)

    # Compute the filters' temporal impulse response
    >>> fbank = FilterBank(nb_filters, bandwidth, 'rect')
    >>> fbank.build_bands(overlap=True, scale='hz')
    >>> fbank.build_filters(tstp, rescale=True, sym=True)

    # Compute the spectral representation of the filters (FFT)
    >>> filters_fft = fbank.spectrum(onesided=True, cmplx=False)

    # Plot the filters' frequency response
    >>> freq = np.fft.fftfreq(len(filters_fft[0])) * frate
    >>> fig = filterbank_spectrum(
    ...     freq, filters_fft,
    ...     legends=[[f"{band[0]} Hz -- {band[1]} Hz"]
    ...              for band in fbank.bands.round(1)],
    ...     rtexts=[f"F{i}" for i in range(1, nb_filters+1)],
    ...     fig_params={'figsize': (19.20, 10.80), 'nrows': nb_filters,
    ...                 'sharey': True, 'sharex': True},
    ...     legend_params={'loc': 'lower left'},
    ...     margin_params={'x': 0.01, 'y': 0.01},
    ...     space_params={'no_yspace': True},
    ...     fname="Fbank_Freq_Resp.png")
    >>> fig.show()
    """
    # Get the nb of rows and columns or set default values
    fig_params = imgparams['fig_params'].copy() if 'fig_params' in imgparams else {}
    if 'nrows' not in fig_params:
        fig_params['nrows'] = len(hfft) if isinstance(hfft, (tuple, list)) else 1
    if 'ncols' not in fig_params:
        fig_params['ncols'] = 1
    # Create the figure
    _, axs = plt.subplots(**fig_params)
    # Extract only the positive frequencies if mirrored spectrums
    if freq[-1] < 0.:
        lgt = len(freq) // 2
        freq, hfft = freq[:lgt], [h[:lgt].real for h in hfft]
    # Plot the filters' spectrums on the figure
    return plot(axs, freq, hfft,
        labels=labels, titles=title, legends=legends, rtexts=rtexts, **imgparams)
#----------------------------------------------------------------------------#

#-----------------------   Filter Temporal Response   -----------------------#
def filterbank(tstp, signals, bands, data=None,
    labels=("Time [s]", "Amplitude [a.u.]"),
    title="Original & Filtered signals",
    rtexts=("Sig", "F1"),
    **imgparams):
    """ Plot the temporal response of the bank's filters

    Generate a 1-plot figure and plot the filtered versions of a signal
    for several bank filters; if `data` is provided, plot the reference
    signal data on the top graph. N.B.: if `data` is provided, it is as-
    sumed that the reference signal and its filtered versions share the
    same x-values `tstp`. See the `plot` module's `set_labels` function
    for details about the decorators (labels, title and side texts).

    If `fname` keyword argument is provided, save the figure in a file
    with that name, and with parameters the `save_params` keyword args.

    Parameters
    ----------
    tstp : array_like
        The data timestamps; if None, use a list of indexes as abscissa.
    signals : array of arrays
        The time response of the filters or the filtered signal (the
        same function can technically be used for both).
    bands : array of 2-tuple
        The list of the sub-band bounding frequencies.
    [OPT] data : array_like
        The signal data samples.
            :Default: None (no reference signal plotted)
    [OPT] labels : 2-tuple of strings
        The x-axis & y-axis labels as (x-label, y-label). Use None to skip.
            :Default: ("Time [s]", "Amplitude [a.u.]")
    [OPT] title : string
        The plot title. Set to None to ignore.
            :Default: "Original & Filtered signals"
    [OPT] rtexts : tuple of strings
        The right y-axis side texts.
            :Default: ["F" + str(i) for i in range(1, len(signals)+1)]
                + ["Sig"] at the first place if `data` is not None

    Other Parameters
    ----------------
    [OPT] fname : str
        The image-to-save pathname (path+name+ext).
            :Default: None (do not save the figure)
    [OPT] fig_params : dict
        The figure parameters passed to the Matplotlib's `subplots` function:
          - `figsize` (2-tuple of floats): the figure dimensions
          - `layout` (str): layout mechanism for plot positioning
            :Default: {'figsize': (19.20, 10.80)}
    [OPT] save_params : dict
        The parameters to save the figure, passed to the Matplotlib's
        `savefig`; used only if `fname` is not empty nor None:
          - `bbox_inches` (str): the margins of the saved graphic image
          - `dpi` (int): image DPI (used if saved as a raster graphic)
            :Default: {'bbox_inches': 'tight'}
    [OPT] subfig_params : dict
        The subfigure parameters (top & bottom figures) passed to the
        Matplotlib's `subfigures` function:
          - `hspace` (float): relative yspace between figures
          - `height_ratios` (2-tuple of ints): relative area between figures
            :Default: {'height_ratios': (1, len(signals))}
    [OPT] plot_params : dict
        The plot parameters passed to the `plot_core` function from the
        `plot` module:
          - `fmt` (str): plot format (markers, lines, etc.)
          - `linewidth` (float): the width of the plots
          - `rasterized` (bool): plot rasterization (as raster graphic)
            :Default: {'linewidth': 2}
    [OPT] label_params : dict
        The labels parameters passed to the `set_labels` function from
        the `plot` module:
          - `size` (float): x and y labels font size
          - `labelpad` (float): x and y labels bounding box space pad
            :Default: {'size': 18}
    [OPT] title_params : dict
        The titles parameters passed to the `set_titles` function from
        the `plot` module:
          - `size` (float): title font size
          - `pad` (float): title bounding box space pad
          - `ha`, `va` (str): horizontal/vertical alignment
            :Default: {'size': 21}
    [OPT] rtext_params : dict
        The right y-axis side text parameters passed to the
        `set_right_yaxis_texts` function from the `plot` module:
          - `size` (float): right y-axis sidetext font size
          - `labelpad` (float): right y-axis side text bounding box space pad
          - `stacked` (bool): vertically stack characters or rotate label
            :Default: {'size': 14, 'stacked': True}
    [OPT] textbox_params : dict
        The textbox format containing the sub-bands passed to Matplotlib's
        `Text` function:
          - `ha`, `va` (str): horizontal/vertical alignment
          - `fontsize` (float): box text size
          - `bbox` (dict): box format/style
              - `facecolor` (str): color of the box, eg 'red', 'blue'
              - `edgecolor` (str): color of the edges, eg 'gray', 'green'
              - `boxstyle` (str): box format, eg 'round'
            :Default: {'ha': 'right', 'va': 'top', 'fontsize': 10,
                       'bbox': {'facecolor': 'whitesmoke', 'edgecolor':
                                'gray', 'boxstyle': 'round'}},
    [OPT] margin_params : dict
        The margins parameters passed to the Matplotlib's `Axis`' `margins`
        method through the `set_margins` function from the `plot` module:
          - `x`, `y` (float): the x/y-axis margin value
            :Default: {'x': 0.01, 'y': 0.1}
    [OPT] space_params : dict
        The space between plots passed to the `remove_spaces` function
        from the `plot` module:
          - `no_xspace` (bool): remove horizontal space between plots
          - `no_yspace` (bool): remove vertical space between plots
            :Default: {'no_xspace': False, 'no_yspace': True}

    Returns
    -------
    fig : plt's Figure
        The drawn figure. Display it with `fig.show()`.

    Examples
    --------
    >>> import numpy as np
    >>> import aer.filters as flt
    >>> import aer.data_tk as dtk

    # Generate a simple signal
    >>> fs = 1000.                               # Sampling frequency
    >>> tstp = 2*np.pi*np.arange(0., 1., 1./fs)  # Time scale
    >>> sin1 = np.sin(50*tstp)                   # 50 Hz sine
    >>> sin2 = 3.*np.sin(250*tstp)               # 250 Hz sine
    >>> noise = 2.*np.random.random(len(tstp))   # Random noise
    >>> data = sin1 + sin2 + noise               # Full dataset

    # Build the bank of filters
    >>> nb_filters = 11
    >>> bands = dtk.sub_bands([0, fs/2], nb_filters, overlap=True)
    >>> filters = [flt.rectangular_time(tstp, band) for band in bands]
    >>> signals = [flt.convolve(data, filt) for filt in filters]

    # Plot the bank of filters
    >>> fig = filterbank(tstp, signals, bands, data,
    ...     labels=("Time [s]", "Amplitude [a.u.]"),
    ...     title="Original & Filtered signals",
    ...     rtexts=["Sig"]+["F" + str(i) for i in range(1, len(signals)+1)])
    >>> fig.show()
    """

    # Default parameters
    params = {'fname': None, 'save_params': {'bbox_inches': 'tight'},
              'fig_params': {'figsize': (19.20, 10.80)},
              'subfig_params': {'height_ratios': (1, len(signals))},
              'plot_params': {'fmt': ''},
              'label_params': {'size': 18},
              'title_params': {'size': 21},
              'rtext_params': {'size': 14, 'stacked': True},
              'textbox_params': {
                  'ha': 'left', 'va': 'top', 'fontsize': 10,
                  'bbox': {'facecolor': 'whitesmoke', 'edgecolor':
                           'gray', 'boxstyle': 'round'}},
              'margin_params': {'x': 0.01, 'y': 0.1},
              'space_params': {'no_xspace': False, 'no_yspace': True}}

    # Append the default parameters to those possibly provided as arguments
    params = check_keys(imgparams, params)

    # Default labels, title and right y-axis side texts
    if rtexts == ("Sig", "F1"):
        rtexts = ["Sig"] + ["F" + str(i) for i in range(1, len(signals)+1)]

    # Build the figures
    fig = plt.figure(**params['fig_params'])

    # If no timestamps are provided, generate a vector of indexes
    if tstp is None:
        tstp = np.arange(len(data))

    # Top figure -- Original signal (if provided)
    if data is not None:
        # Split the figures in two
        subfigs = fig.subfigures(2, **params['subfig_params'])
        axs = subfigs[0].subplots(1)        # Top figure (data)
        fig = subfigs[1]                    # Bottom figure (filters)

        # Plot the reference data
        plot_core(axs, tstp, data, **params['plot_params'])

        # Top graph labels
        set_decorations(axs,
            None, title, rtexts=rtexts[0],
            label_params=params['label_params'],
            title_params=params['title_params'],
            rtext_params=params['rtext_params'])

        # Reduce the space margins on the right & left of the plots
        if params['margin_params'] != {}:
            set_margins(axs, **params['margin_params'])

        # Set no title and remove the 1st rtext for the following subplots
        title = None
        rtexts = rtexts[1:]

    # Bottom figure -- Filtered signal
    axs = fig.subplots(len(signals), sharex=True)

    # Plot the filtered signals & set the margins
    plot_core(axs, tstp, signals, **params['plot_params'])

    # Add the bandwidths as text boxes
    for axis, band in zip(axs, np.round(bands, 2)):
        axis.text(0.01, 0.90, "BW = "+str(band[0])+"-"+str(band[1])+" Hz",
                  transform=axis.transAxes, **params['textbox_params'])

    # Filtered signals labels
    set_decorations(axs,
        labels, title, rtexts=rtexts,
        label_params=params['label_params'],
        title_params=params['title_params'],
        rtext_params=params['rtext_params'])

    # Reduce the space margins on the right & left of the plots
    if params['margin_params'] != {}:
        set_margins(axs, **params['margin_params'])

    # Remove useless white spaces
    remove_spaces(fig, **params['space_params'])

    # Save the figure in the specified file
    # If the filename contains directories, create them if they do not exist
    if params['fname'] not in (None, ''):
        fig.get_figure().savefig(check_path(params['fname']), **params['save_params'])

    # Return the figure from the 'subfigure'
    return fig.get_figure()
#----------------------------------------------------------------------------#

##############################################################################z
