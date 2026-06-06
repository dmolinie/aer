import numpy as np
from aer.filters._filterbank import *


def test_FilterBank():
    """ Filterbank class """
    import os
    import aer.data_tk as dtk
    import matplotlib.pyplot as plt
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

    #------------------   Filter Bank Frequency Response   ------------------#
    # Build the frequency-domain sub-bands on a given scale
    nb_filters = 11
    bandwidth = (0., 0.5*frate)

    # Compute the filters' temporal impulse response
    fbank = FilterBank(nb_filters, bandwidth, 'rect')
    fbank.build_bands(overlap=True, scale='hz')
    fbank.build_filters(tstp, rescale=True, sym=True)

    # Compute the spectral representation of the filters (FFT)
    filters_fft = fbank.spectrum(onesided=True, cmplx=False)

    # Plot the filters' frequency response
    freq = np.fft.fftfreq(len(filters_fft[0]))*frate
    fig = disp.filterbank_spectrum(
        freq, filters_fft, fname="Fbank_Freq_Resp.png")
    fig.show()
    #------------------------------------------------------------------------#

    #-----------------   FB Time Response -- Full Signal   ------------------#
    # Build the frequency-domain sub-bands on a given scale
    nb_filters = 8
    bandwidth = (0., 225.)
    #bandwidth = (0., 0.5*frate)

    # Build the filters time response
    fbank = FilterBank(nb_filters, bandwidth, 'rect')
    fbank.build_bands(overlap=True, scale='hz')
    fbank.build_filters(tstp, sym=True, rescale=True)

    # Filter the signal by convolving it with the filters' time response
    signals = 200.*fbank.filter_signal(data)

    # Plot the fitered signals
    plt.rcParams.update({'font.size': 14})
    fig = disp.filterbank(
        tstp, signals, fbank.bands, data,
        fname="Filterbank.pdf",
        fig_params={'figsize': (14.00, 10.80)},
        subfig_params={'hspace': 0.},
#        title_params={'size': 30},
#        label_params={'size': 25},
#        rtext_params={'size': 18},
#        textbox_params={'fontsize': 14}
        )
    fig.show()
    #------------------------------------------------------------------------#

    #----------------   FB Time Response -- Signal Chunks   -----------------#
    # Parameters
    nb_filters = 8              # Number of filters of the bank
    win_size = 50               # Vertical resolution (frequency)
    hop_size = 10               # Horizontal resolution (time)
    bandwidth = (0., 0.5*frate) # Filterbank bandwidth

    # Time and data chunks (consecutive overlapping segments)
    tstp_chunks = dtk.build_chunks(tstp, win_size, hop_size)
    data_chunks = dtk.build_chunks(data, win_size, hop_size)

    # Filter the signal chunks with the bank filters
    fbank = FilterBank(nb_filters, bandwidth, 'rect')
    fbank.build_bands(overlap=True, scale='hz')
    data_resp = fbank.filter_signal_chunks(
        tstp_chunks, data_chunks, norm=True, sym=True, rescale=True)

    # Plot the cumulative filtered signals
    offset = hop_size + dtk.pad_offset(len(data), win_size, hop_size)
    off = offset//2
    fig = disp.filterbank(
        tstp, data_resp[:, off:off-offset], fbank.bands, data,
        fname="FB_chunks.png",
        fig_params={'figsize': (19.31, 8.08)}, subfig_params={'hspace': 0.05})
    fig.show()
    #------------------------------------------------------------------------#

def test_FilterBank_init():
    """ Instantiate a FilterBank object (constructor) """
    # 10 rectangle filters with a total bandwidth between 50 and 500 Hz
    fbank = FilterBank(10, (50, 500), 'rectangle')
    # 100 gammatone filters with a total bandwidth between 0 and 1000 Hz
    fbank = FilterBank(100, (0, 1000), 'gammatone')

def test_FilterBank_build_bands():
    """ Split the bandwidth attribute into sub-bands on a given scale """
    fbank = FilterBank(4, (10, 30), 'rect')
    fbank.build_bands(overlap=False, scale='hz')
    print(fbank.bands.T)

def test_FilterBank_build_filters():
    """ Compute the time impulse response of the filters """
    fbank = FilterBank(4, (10, 30), 'rect')
    fbank.build_bands(overlap=False, scale='hz')
    fbank.build_filters(np.linspace(0, 2, 600))

def test_FilterBank_filter_signal():
    """ Filter the signal data with the filters of a bank """
    fbank = FilterBank(4, (10, 30), 'rect')
    fbank.build_bands(overlap=False, scale='hz')
    fbank.build_filters(np.linspace(0, 2, 600))
    tstp = np.linspace(0., 10., 1000)
    freq = np.array([100., 200.]) * 2*np.pi
    data = np.sin(tstp*freq[0]) + np.sin(tstp*freq[1])
    signals = fbank.filter_signal(data)

def test_FilterBank_filter_signal_chunks():
    """ Filter a set of signal chunks """
    import aer.data_tk as dtk
    # Parameters
    nb_filters = 15         # Number of filters of the bank
    win_size = 50           # Vertical resolution (frequency)
    hop_size = 10           # Horizontal resolution (time)
    freq = 10
    frate = 600
    # Data
    tstp = np.arange(0., 2., 1/frate)
    data = np.sin(2*np.pi*freq*tstp) + np.sin(2*np.pi*5*freq*tstp)
    bandwidth = (0, frate/2)
    # Time and data chunks (consecutive overlapping segments)
    tstp_chunks = dtk.build_chunks(tstp, win_size, hop_size)
    data_chunks = dtk.build_chunks(data, win_size, hop_size)
    # Filter the signal chunks with the bank filters
    fbank = FilterBank(nb_filters, bandwidth, 'rect')
    fbank.build_bands(overlap=True, scale='hz')
    resps = fbank.filter_signal_chunks(tstp_chunks, data_chunks)

def test_FilterBank_spectrum():
    """ Build the spectrum of the filters with the NumPy FFT """
    fbank = FilterBank(4, (10, 30), 'rect')
    fbank.build_bands(overlap=False, scale='hz')
    fbank.build_filters(np.linspace(0, 2, 600))
    filters_fft = fbank.spectrum(False, True)



# Launch test/example functions
test_FilterBank

test_FilterBank_init()

test_FilterBank_build()

test_FilterBank_build_bands()

test_FilterBank_build_filters()

test_FilterBank_filter_signal()

test_FilterBank_filter_signal_chunks()

test_FilterBank_spectrum()

