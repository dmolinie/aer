import numpy as np
from aer.filters._filters import *


def test_flip():
    """ Mirror a vector and stack the mirrored and original data """
    print(flip([1, 2, 3]))
    print(flip([1, 2, 3], 0))
    print(flip([2, 3], [-1, 0, +1], False))

def test_convolve():
    """ Operate the convolution two vectors """
    conv = convolve(np.linspace(-5, +5, 10), np.linspace(-1, +10, 10))

def test_rectangular_time():
    """ Rectangular-shaped filter time impulse response """
    tstp = np.linspace(100., 500., 200)
    gate = rectangular_time(tstp, (150., 300.))

def test_triangular_time():
    """ Triangular-shaped filter time impulse response """
    tstp = np.linspace(100., 500., 200)
    trig = triangular_time(tstp, (150., 300.))

def test_gammatone_time():
    """ Gammatone filter time impulse response """
    tstp = np.linspace(100., 500., 200)
    gamma = gammatone_time(tstp, (150., 300.), order=2)

def test_rectangular_freq():
    """ Frequency-domain rectangular-shaped filter """
    freq = np.linspace(100., 500., 200)
    gate = rectangular_freq(freq, 200., 300.)

def test_triangular_freq():
    """ Frequency-domain triangular-shaped filter """
    freq = np.linspace(100., 500., 200)
    trig = triangular_freq(freq, 200., 300.)

def test_overall_freq():
    """ Didactic example on spectral filtering """
    import os
    import matplotlib.pyplot as plt
    import aer.plot as plot
    import aer.display as disp
    plt.rcParams.update({'font.size': 12})

    # Load the data if they exist, or generate them else
    DATA_FOLDER = "dummy_data/"
    if os.path.exists(DATA_FOLDER):
        tstp = np.load(DATA_FOLDER+"tstp.npy")
        data = np.load(DATA_FOLDER+"data.npy")
        frate = np.load(DATA_FOLDER+"frate.npy")
        freqs = np.load(DATA_FOLDER+"freqs.npy")
    else:
        from base_signal import tstp, data, frate, freqs

    bandwidth = (50., 75.)          # Total bandwidth of the filterbank
    mid = len(tstp) // 2            # Half length of `tstp`

    # Time response of the filter
    filt = rectangular_time(tstp, bandwidth, False)
    ##filt = triangular_time(tstp, bandwidth, False)
    ##filt = gammatone_time(tstp, bandwidth, order=3)

    # Frequency responses (spectral representation)
    hfft = np.fft.fft(filt)         # Filter spectrum
    sfft = np.fft.fft(data)         # Signal spectrum
    # Rebuild the filtered signal
    sigfft = np.fft.ifft(hfft * sfft)
    # Frequency scale
    freq = np.fft.fftfreq(len(hfft))*frate

    # Build the figures
    fig = plt.figure(figsize=(19.18, 8.67))
    # Top figure -- Time data
    subfigs = fig.subfigures(2, hspace=0., height_ratios=[1., 1.33])
    # Plot the reference data
    topaxs = subfigs[0].subplots(3, sharex=True)
    subfigs[0].delaxes(topaxs[2])
    # Plot & decoration arguments
    plot_params = {'fmt': '-'}
    deco_args = {'label_params': {'size': 18},
                 'title_params': {'size': 21},
                 'legend_params': {'fontsize': 13, 'loc': 'upper right'},
                 'margin_params': {'x': 0.01, 'y': 0.1}}
    # Plot the data on the top graphs (temporal representations)
    plot.plot_core(topaxs, tstp, (data, sigfft.real), **plot_params)
    plot.set_decorations(topaxs,
        ("Time [s]", "Amplitude [a.u.]"),
        "Temporal representation",
        ("Original signal", "Rebuilt signal"), **deco_args)
    # Add the frequencies at the bottom of the figure
    subfigs[0].text(0.51, 0.24, disp.freqs2str(freqs),
        ha='center', va='center', fontsize=13,
        bbox={'facecolor': 'whitesmoke', 'edgecolor': 'gray', 'boxstyle': 'round'})
    # Bottom figure -- Spectral data
    blwaxs = subfigs[1].subplots(3, sharex=True)
    # Plot the data on the bottom graphs (spectral representations)
    plot.plot_core(blwaxs,
        freq[:mid],
        (sfft[:mid].real, hfft[:mid].real, (hfft[:mid].real)*(sfft[:mid].real)),
        **plot_params)
    plot.set_decorations(blwaxs,
        ("Frequency [Hz]", "Amplitude [a.u.]"),
        "Spectral representation",
        ("Signal spectrum", "Filter spectrum", "Filtered spectrum"), **deco_args)
    # Remove vertical spaces
    plot.remove_spaces(fig, no_yspace=True)
    # Save the figure into the specified file
    plt.savefig("Spectral_Filter.png", bbox_inches='tight')
    plt.show()
    #------------------------------------------------------------------------#

def test_overall_time():
    """ Didactic example on temporal filtering """
    import os
    import matplotlib.pyplot as plt
    import aer.plot as plot
    plt.rcParams.update({'font.size': 12})

    # Load the data if they exist, or generate them else
    DATA_FOLDER = "dummy_data/"
    if os.path.exists(DATA_FOLDER):
        tstp = np.load(DATA_FOLDER+"tstp.npy")
        data = np.load(DATA_FOLDER+"data.npy")
    else:
        from base_signal import tstp, data

    bandwidth = (95., 110.)         # Total bandwidth of the filterbank
    MIRROR = True
    tstp2 = flip(tstp, even=False) if MIRROR else tstp

    # Time response of the filter
    #filt = rectangular_time(tstp, bandwidth, MIRROR)
    filt = triangular_time(tstp, bandwidth, MIRROR)
    #filt = gammatone_time(tstp, bandwidth, order=3)

    # Filter the signal in the time domain (convolution)
    conv = convolve(data, filt)

    # Plot the original signal, filter response and rebuilt signal
    fig, axs = plt.subplots(3, 1, figsize=(19.20, 10.80))
    fig = plot.plot(axs, (tstp, tstp2, tstp), (data, filt, conv),
        ("Time [s]", "Amplitude [a.u.]"),
        "Temporal representation",
        ("Original signal", "Filter response", "Rebuilt signal"),
        fname="Temporal_Filter.png",
        label_params={'size': 15, 'repeat_labels': (False, True)},
        title_params={'size': 18},
        legend_params={'fontsize': 13})
    fig.show()



# Launch test/example functions
test_flip()

test_convolve()

test_rectangular_time()

test_triangular_time()

test_gammatone_time()

test_rectangular_freq()

test_triangular_freq()

test_overall_freq()

test_overall_time()

