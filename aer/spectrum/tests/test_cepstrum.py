import numpy as np
from aer.spectrum._spectrum import *
from aer.spectrum._cepstrum import *


def test_Cepstrum():
    """ Cepstrum class """
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
    # Signal cepstrum (requires both neg & pos FFTs)
    quefs, ceps = cepstrum(sfft, True, True)
    fig1 = disp.cepstrum(quefs, ceps, fname="Cepstrum.png")
    # Signal cepstral power density
    cxx = cepstral_power(ceps)
    fig2 = disp.cepstralpower(quefs, cxx, fname="CepstralPower.png")
    # Class-based variant
    cepst = Cepstrum(axis=-1)
    cepst.spectrum(data)
    cepst.freq *= frate
    cepst.cepstrum()
    fig3 = disp.cepstrum(cepst.quef, cepst.ceps, fname="Cepstrum2.png")
    cxx = cepst.cepstral_power()
    fig4 = disp.cepstralpower(cepst.quef, cxx, fname="CepstralPower2.png")
    plt.show()

def test_Cepstrum_cepstrum():
    """ Cepstrum transformation of a given signal from its FT """
    data = np.sin(2*np.pi*5.*np.linspace(0., 1.0, 100))
    cepst = Cepstrum(axis=-1, norm=None, n=100)
    cepst.spectrum(data)
    cepst.spectrum(data, onesided=False, shift=False)
    cepst.cepstrum(n=100, norm='backward')

def test_Cepstrum_cepstral_power():
    """ Cepstral power density estimation """
    data = np.sin(2*np.pi*5.*np.linspace(0., 1.0, 100))
    cepst = Cepstrum(axis=-1, norm=None)
    cepst.spectrum(data)
    cepst.cepstrum(n=100, norm='backward')
    cxx = cepst.cepstral_power()

def test_ceps_quef():
    """ Build the frequency scale for an FFT """
    data = np.sin(2*np.pi*5.*np.linspace(0., 1.0, 100))
    sfft = spec_sfft(data, onesided=False, shift=False)
    print(ceps_quef(len(sfft), True))
    print(ceps_quef(len(sfft), False, True))

def test_ceps_sfft():
    """ Compute the FFT components of the data provided as input """
    data = np.sin(2*np.pi*5.*np.linspace(0., 1.0, 100))
    sfft = spec_sfft(data, onesided=False, shift=False)
    print(ceps_sfft(sfft, False, True))

def test_cepstrum():
    """ Cepstrum transformation of a given signal from its FT """
    data = np.sin(2*np.pi*5.*np.linspace(0., 1.0, 100))
    sfft = spec_sfft(data, onesided=False, shift=False)
    quef, ceps = cepstrum(sfft)

def test_cepstral_power():
    """ Cepstral power density """
    data = np.sin(2*np.pi*5.*np.linspace(0., 1.0, 100))
    sfft = spec_sfft(data, onesided=False, shift=False)
    ceps = ceps_sfft(sfft)
    cxx = cepstral_power(ceps)



# Launch test/example functions
test_Cepstrum()

test_Cepstrum_cepstrum()

test_Cepstrum_cepstral_power()

test_ceps_quef()

test_ceps_sfft()

test_cepstrum()

test_cepstral_power()

