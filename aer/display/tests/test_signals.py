import numpy as np
import matplotlib.pyplot as plt
from aer.display._signals import *


def test_freqs2str():
    """ Convert a list of scalar frequencies as a literal """
    print(freqs2str([1.123]))
    print(freqs2str([1.123, 2.456, 3.789]))
    print(freqs2str([1.123456789, 2.456789123, 3.789123456], 3))

def test_signal():
    """ Plot a time-domain signal against time """
    # Generate dummy data
    tstp = np.linspace(0., 1.)
    sine = np.sin(6.28*10.*tstp)
    # Plot the signal against time
    fig = signal(tstp, sine,
        fig_params={'figsize': (19.20, 10.80)},
        plot_params={'fmt': '--'})
    fig.show()

def test_spectrum():
    """ Plot the spectrum (Fourier Transform) of a signal """
    from aer.spectrum import Spectrum
    # Generate a simple sum of a 10 Hz & a 50 Hz sines
    frate = 200     # Frequency rate
    tps = np.linspace(0., 1., frate)
    sig = np.sin(6.28*10.*tps) + np.sin(6.28*50.*tps)
    # Compute the signal spectrum
    spect = Spectrum()
    spect.spectrum(sig, True, frate=frate)
    # Plot the signal spectrum
    fig = spectrum(spect.freq, spect.sfft,
        fig_params={'figsize': (19.20, 10.80)})
    fig.show()

def test_cepstrum():
    """ Plot the cepstrum of a signal """
    from aer.spectrum import Cepstrum
    # Generate a simple sum of a 10 Hz & a 50 Hz sines
    frate = 200     # Frequency rate
    tps = np.linspace(0., 1., frate)
    sig = np.sin(6.28*10.*tps) + np.sin(6.28*50.*tps)
    # Compute the signal spectrum & cepstrum
    cepst = Cepstrum()
    cepst.spectrum(sig, True, frate=frate)  # Spectrum
    cepst.cepstrum()                        # Cepstrum
    # Plot the signal cepstrum
    fig = cepstrum(cepst.quef, cepst.ceps,
        fig_params={'figsize': (19.20, 10.80)})
    fig.show()

def test_spectralpower():
    """ Plot the spectral power density (periodogram) of a signal """
    from aer.spectrum import Spectrum
    # Generate a simple sum of a 10 Hz & a 50 Hz sines
    frate = 200     # Frequency rate
    tps = np.linspace(0., 1., frate)
    sig = np.sin(6.28*10.*tps) + np.sin(6.28*50.*tps)
    # Compute the signal spectrum
    spect = Spectrum()
    spect.spectrum(sig, True, frate=frate)
    pxx = spect.spectral_power()
    # Plot the signal spectrum
    fig = spectralpower(spect.freq, pxx,
        fig_params={'figsize': (19.20, 10.80)})
    fig.show()

def test_cepstralpower():
    """ Plot the cepstral power density of a signal """
    from aer.spectrum import Cepstrum
    # Generate a simple sum of a 10 Hz & a 50 Hz sines
    frate = 200     # Frequency rate
    tps = np.linspace(0., 1., frate)
    sig = np.sin(6.28*10.*tps) + np.sin(6.28*50.*tps)
    # Compute the signal spectrum, cepstrum and power cepstrum
    cepst = Cepstrum()
    cepst.spectrum(sig, True, frate=frate)  # Spectrum
    cepst.cepstrum()                        # Cepstrum
    cxx = cepst.cepstral_power()            # Cepstral power
    # Plot the signal power cepstrum
    fig = cepstralpower(cepst.quef, cxx,
        fig_params={'figsize': (19.20, 10.80)})
    fig.show()

def test_spectrogram():
    """ Plot the spectrogram of a signal """
    from scipy.signal.windows import gaussian
    from aer.spectrum import ShortTimeFT
    # Generate a simple signal
    fs = 1000.                               # Sampling frequency
    tstp = 2*np.pi*np.arange(0., 1., 1./fs)  # Time scale
    sin1 = np.sin(50*tstp)                   # 50 Hz sine
    sin2 = 3.*np.sin(250*tstp)               # 250 Hz sine
    noise = 2.*np.random.random(len(tstp))   # Random noise
    # Build the spectrogram of the signal
    win = gaussian(50, std=8, sym=True)      # Symmetric Gaussian window
    stf = ShortTimeFT()                      # Instantiate the STFT
    stf.stft_time(sin1+sin2+noise, win, 10)  # Build the STFT
    # Plot the spectrogram
    fig = spectrogram(
        stf.spectrogram(), (50, 250),
        fname="Spectrogram.pdf", save_params={'dpi': 100},
        fig_params={'figsize': (19.20, 10.80), 'layout': 'constrained'},
        imshow_params={'cmap': 'magma', 'origin': 'lower', 'aspect': 'auto',
                       'extent': (0.0, len(tstp)/fs, 0.0, fs/2)},
        colorbar_params={'label': 'Magnitude [a.u.]'},
        label_params={'size': 18}, title_params={'size': 21},
        figtext_params={'fontsize': 18})
    fig.show()

def test_filterbank_spectrum():
    """ Plot the frequency response of the bank's filters """
    from aer.filters import FilterBank
    # Generate dummy data
    frate = 1000.
    tstp = np.linspace(0., 1., int(frate))
    freqs = (10, 50, 100)
    data = np.zeros_like(tstp)
    for freq in freqs:
        data += np.sin(6.28*freq*tstp)
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
    freq = np.fft.fftfreq(len(filters_fft[0])) * frate
    fig = filterbank_spectrum(
        freq, filters_fft,
        legends=[[f"{band[0]} Hz -- {band[1]} Hz"]
                 for band in fbank.bands.round(1)],
        rtexts=[f"F{i}" for i in range(1, nb_filters+1)],
        fig_params={'figsize': (19.20, 10.80), 'nrows': nb_filters,
                    'sharey': True, 'sharex': True},
        legend_params={'loc': 'lower left'},
        margin_params={'x': 0.01, 'y': 0.01},
        space_params={'no_yspace': True},
        fname="Fbank_Freq_Resp.png")
    fig.show()

def test_filterbank():
    """ Plot the temporal response of the bank's filters """
    import aer.filters as flt
    import aer.data_tk as dtk
    # Generate a simple signal
    fs = 1000.                               # Sampling frequency
    tstp = 2*np.pi*np.arange(0., 1., 1./fs)  # Time scale
    sin1 = np.sin(50*tstp)                   # 50 Hz sine
    sin2 = 3.*np.sin(250*tstp)               # 250 Hz sine
    noise = 2.*np.random.random(len(tstp))   # Random noise
    data = sin1 + sin2 + noise               # Full dataset
    # Build the bank of filters
    nb_filters = 11
    bands = dtk.sub_bands([0, fs/2], nb_filters, overlap=True)
    filters = [flt.rectangular_time(tstp, band) for band in bands]
    signals = [flt.convolve(data, filt) for filt in filters]
    # Plot the bank of filters
    fig = filterbank(tstp, signals, bands, data,
        labels=("Time [s]", "Amplitude [a.u.]"),
        title="Original & Filtered signals",
        rtexts=["Org"]+["F" + str(i) for i in range(1, len(signals)+1)])
    fig.show()



# Launch test/example functions
test_freqs2str()

test_signal()

test_spectrum()

test_cepstrum()

test_spectralpower()

test_cepstralpower()

test_spectrogram()

test_filterbank_spectrum()

test_filterbank()

