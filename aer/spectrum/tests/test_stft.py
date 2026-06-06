import numpy as np
import matplotlib.pyplot as plt
from scipy.signal.windows import gaussian
from aer.spectrum._stft import *


def test_STFT():
    """ Short-Time Fourier Transform class """
    import os
    import aer.display as disp
    # Load the data if they exist, or generate them else
    DATA_FOLDER = "dummy_data/"
    if os.path.exists(DATA_FOLDER):
        tstp = np.load(DATA_FOLDER+"tstp.npy")
        data = np.load(DATA_FOLDER+"data.npy")
        frate = np.load(DATA_FOLDER+"frate.npy")
        freqs = np.load(DATA_FOLDER+"freqs.npy")
    else:
        from base_signal import data, tstp, frate, freqs
    # STFT parameters
    win_size = 50                               # Vertical resolution (frequency)
    hop_size = 10                               # Horizontal resolution (time)
    # Build the STFT
    stf = ShortTimeFT()                         # Instantiate the STFT
    win = gaussian(win_size, std=8, sym=True)   # Symmetric Gaussian window
    stf.stft_time(data, win, 10)                # Build the STFT
    pxx = stf.spectrogram()                     # Build the spectrogram
    # plotlay the Spectrogram
    plt.rcParams.update({'font.size': 15})
    params = {'fig_params': {'figsize': (12.50, 10.80), 'layout': 'constrained'},
              'label_params': {'size': 25},
              'title_params': {'size': 30},
              'save_params': {'bbox_inches': 'tight', 'dpi': 100},
              'imshow_params': {'cmap': 'magma', 'origin': 'lower', 'aspect': 'auto',
                                'extent': (0., tstp[-1].round(0), 0., 0.5*frate)}}
    fig = disp.spectrogram(pxx, freqs=freqs, fname="Spectrogram.pdf", **params)
    fig.show()

def test_STFT_init():
    """ Instantiate an STFT object (constructor) """
    # Default parameters
    stf = ShortTimeFT()
    # With parameters for the NumPy's FFT
    stf = ShortTimeFT(n=100, norm='backward')

def test_STFT_stft_chunks():
    """ Apply the STFT to a set of signal chunks """
    import aer.data_tk as dtk
    data = np.sin(2*np.pi*5.*np.linspace(0., 1.0, 1000))
    chunks = dtk.build_chunks(data, 50, 50)
    stf = ShortTimeFT()
    stf.stft_chunks(chunks)

def test_STFT_stft_time():
    """ Compute the Short-Time Fourier Transform """
    data = np.sin(2*np.pi*5.*np.linspace(0., 1.0, 100))
    stf = ShortTimeFT(n=50, onesided=False, shift=False)
    stf.stft_time(data, gaussian(100, std=8, sym=True), 10)

def test_STFT_spectrogram():
    """ Build the spectrogram amplitude of an STFT """
    data = np.sin(2*np.pi*5.*np.linspace(0., 1.0, 100))
    stf = ShortTimeFT(n=50, onesided=False, shift=False)
    stf.stft_time(data, gaussian(100, std=8, sym=True), 10)
    spec = stf.spectrogram()

def test_stft_chunks():
    """ Apply the STFT to a set of signal chunks """
    import aer.data_tk as dtk
    data = np.sin(2*np.pi*5.*np.linspace(0., 1.0, 1000))
    chunks = dtk.build_chunks(data, 50, 50)
    freq, stft = stft_chunks(chunks, onesided=True, axis=-1)
    freq, stft = stft_chunks(chunks, onesided=False, shift=True, n=100)

def test_stft_time():
    """ Compute the Short-Time Fourier Transform """
    data = np.sin(2*np.pi*5.*np.linspace(0., 1.0, 100))
    win = gaussian(100, std=8, sym=True)
    freq, stft = stft_time(data, win, 10, onesided=False, shift=False)

def test_spectrogram():
    """ Build the spectrogram amplitude of an STFT """
    data = np.sin(2*np.pi*5.*np.linspace(0., 1.0, 100))
    freqs, stft = stft_time(data, gaussian(100, std=8, sym=True), 10, n=200)
    spec = spectrogram(stft)



# Launch test/example functions
test_STFT()

test_STFT_init()

test_STFT_stft_chunks()

test_STFT_stft_time()

test_STFT_spectrogram()

test_stft_chunks()

test_stft_time()

test_spectrogram()
