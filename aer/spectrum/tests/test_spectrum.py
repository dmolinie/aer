import numpy as np
from aer.spectrum._spectrum import *


def test_Spectrum():
    """ Spectrum class """
    import os
    import matplotlib.pyplot as plt
    import aer.display as disp
    # Load the data if they exist, or generate them else
    DATA_FOLDER = "dummy_data/"
    if os.path.exists(DATA_FOLDER):
        data = np.load(DATA_FOLDER+"data.npy")
        frate = np.load(DATA_FOLDER+"frate.npy")
    else:
        from base_signal import data, frate
    # Fourier Transform
    freq, sfft = spectrum(data, onesided=False, shift=False)
    freq *= frate
    # Signal spectrum
    freqs, spec = onesidedfft(freq, sfft)
    fig1 = disp.spectrum(freqs, spec, fname="Spectrum.png")
    # Signal spectral power density (periodogram)
    pxx = spectral_power(spec)
    fig2 = disp.spectralpower(freqs, pxx, fname="SpectralPower.png")
    # Class-based variant
    spect = Spectrum(axis=-1)
    spect.spectrum(data, onesided=True)
    spect.freq *= frate
    fig3 = disp.spectrum(spect.freq, spect.sfft, fname="Spectrum2.png")
    pxx = spect.spectral_power()
    fig4 = disp.spectralpower(spect.freq, pxx, fname="SpectralPower2.png")
    plt.show()

def test_Spectrum_spectrum():
    """ Compute the FFT of an input signal """
    data = np.sin(2*np.pi*5.*np.linspace(0., 1.0, 100))
    spect = Spectrum(axis=-1, norm=None, n=100)
    spect.spectrum(data, True, frate=100)

def test_Spectrum_spectral_power():
    """ Spectral power density estimation (periodogram) """
    data = np.sin(2*np.pi*5.*np.linspace(0., 1.0, 100))
    spect = Spectrum(axis=-1, norm=None)
    spect.spectrum(data)
    pxx = spect.spectral_power()

def test_onesidedfft():
    """ Given a two-sided FFT, return the corresponding one-sided FFT """
    data = np.sin(2*np.pi*5.*np.linspace(0., 1.0, 100))
    freq, sfft = spectrum(data, onesided=False, shift=False)
    freq_os, sfft_os = onesidedfft(freq, sfft)

def test_twosidedfft():
    """ Rebuild the whole FFT (neg & pos freqs) from a one-sided FFT """
    data = np.sin(2*np.pi*5.*np.linspace(0., 1.0, 100))
    freq, sfft = spectrum(data, onesided=True)
    freq_ts, sfft_ts = twosidedfft(freq, sfft, True)
    freq_ts, sfft_ts = twosidedfft(freq, sfft, False, False)

def test_spec_freq():
    """ Build the frequency scale for an FFT """
    data = np.sin(2*np.pi*5.*np.linspace(0., 1.0, 100))
    print(spec_freq(len(data), True))
    print(spec_freq(len(data), False, True))

def test_spec_sfft():
    """ Compute the FFT components of the data provided as input """
    data = np.sin(2*np.pi*5.*np.linspace(0., 1.0, 100))
    print(spec_sfft(data, True))
    print(spec_sfft(data, False, True, axis=-1, norm=None))

def test_spectrum():
    """ Compute the FFT of an input signal """
    data = np.sin(2*np.pi*5.*np.linspace(0., 1.0, 100))
    freq, sfft = spectrum(data, True)
    freq, sfft = spectrum(data, True, axis=-1, norm=None)
    freq, sfft = spectrum(data, True, n=500, norm='forward')

def test_spectral_power():
    """ Spectral Power Density estimation (periodogram) """
    data = np.sin(2*np.pi*5.*np.linspace(0., 1.0, 100))
    pxx = spectral_power(spec_sfft(data, True))



# Launch test/example functions
test_Spectrum()

test_Spectrum_spectrum()

test_Spectrum_spectral_power()

test_onesidedfft()

test_twosidedfft()

test_spec_freq()

test_spec_sfft()

test_spectrum()

test_spectral_power()

