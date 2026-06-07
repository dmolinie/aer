import numpy as np
from aer.data_tk._scaling import *


def test_extrema():
    """ Return the min and max values within the data """
    print(extrema(np.linspace(-5, +5, 5)))

def test_extrema_nd():
    """ Get the min & max values among several arrays """
    # Generate semi-random data
    rng = np.random.default_rng(seed=0)
    data = np.arange(1., 11.) + 1j*np.arange(1., 11.)
    rng.shuffle(data.real)
    rng.shuffle(data.imag)
    # Min & max of a unique data array
    print(extrema_nd(data))
    # Min & max between several data arrays
    print(extrema_nd([data, data+5j, data-3.]))

def test_rescale():
    """ Data rescaling between `bounds` """
    data = np.random.random((10, 2))
    data_norm, lims, bnds = rescale(data, (0., 1.), (-1., 1.))

def test_hzscale():
    """ Build a set of `samples` linearly spaced samples in the Hertz
        scale spanning from `min_freq` to `max_freq` Hz frequencies """
    print(hzscale(20., 20000., 10000))

def test_melscale():
    """ Build a set of `samples` linearly spaced samples in the Mel
        scale spanning from `min_freq` to `max_freq` Hz frequencies """
    print(melscale(20., 20000., 10000))

def test_barkscale():
    """ Build a set of `samples` linearly spaced samples in the Bark
        scale spanning from `min_freq` to `max_freq` Hz frequencies """
    print(barkscale(20., 20000., 10000))

def test_hz2mel():
    """ Convert Hertz frequency into Mel frequency """
    print(hz2mel(np.linspace(0., 100., 101)))

def test_mel2hz():
    """ Convert Mel frequency into Hertz frequency """
    print(mel2hz(np.linspace(0., 100., 101)))

def test_hz2bark():
    """ Convert Hertz frequency into Bark frequency """
    print(hz2bark(np.linspace(0., 100., 101)))

def test_bark2hz():
    """ Convert Bark frequency into Hertz frequency """
    print(bark2hz(np.linspace(0., 100., 101)))

def test_sub_bands():
    """ Split a bandwidth interval into sub-bands on a given scale """
    print(sub_bands((0, 12), 3, True, 'hz'))
    print(sub_bands((0, 12), 3, True, 'mel'))
    print(sub_bands((0, 12), 3, True, 'bark').round(3))



# Launch test/example functions
test_extrema()

test_extrema_nd()

test_rescale()

test_hzscale()

test_melscale()

test_barkscale()

test_hz2mel()

test_mel2hz()

test_hz2bark()

test_bark2hz()

test_sub_bands()

