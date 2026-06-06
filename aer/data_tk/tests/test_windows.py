import numpy as np
from aer.data_tk._windows import *


def test_cosine_sum():
    """ General Cosine-sum window """
    print(cosine_sum(7, [0.5, 0.5]))            # Hann
    print(cosine_sum(7, [0.54, 0.46]))          # Hamming
    print(cosine_sum(7, [0.42, 0.5, 0.08]))     # Blackman

def test_hann():
    """ Hann window function """
    print(hann(7))

def test_hamming():
    """ Hamming window function """
    print(hamming(7))

def test_blackman():
    """ Blackman window function """
    print(blackman(7))

def test_comp_windows():
    """ Plot the different windows on the same graph """
    import aer.plot as plot
    # Generate the windows
    nb_points = 100
    win_hann = hann(nb_points)
    win_hann_cs = cosine_sum(nb_points, [0.5, 0.5])
    win_hamming = hamming(nb_points)
    win_hamming_cs = cosine_sum(nb_points, [0.54, 0.46])
    win_blackman = blackman(nb_points, 0.16)
    win_blackman_cs = cosine_sum(nb_points, [0.42, 0.5, 0.08])
    fig = plot.plot(None, None,
        (win_hann, win_hann_cs, win_hamming,
         win_hamming_cs, win_blackman, win_blackman_cs),
        fig_params={'figsize': (19.20, 10.80)},
        legends=("Hann", "Hann CS", "Hamming",
                 "Hamming CS", "Blackman", "Blackman CS"),
        fname="Windows.png")
    fig.show()



# Launch test/example functions
test_cosine_sum()

test_hann()

test_hamming()

test_blackman()

test_comp_windows()

