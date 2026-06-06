import numpy as np
import aer.datasets
from aer.datasets.mivia_loader import *


def test_parse_xml():
    """ Parse an XML file from the MIVIA AE or RAE databases """
    # Data folders and data files
    XMLFOLDER = aer.datasets.__path__[0]+"/MIVIA_AE_example/training/00066.xml"
    classes, tstamps = parse_xml(XMLFOLDER)

def test_read_events():
    """ Read & parse a MIVIA XML events file """
    # Path to the XML file
    ROOT = f"{aer.datasets.__path__[0]}/MIVIA_AE_example/training/"
    XMLFOLDER = ROOT
    # Read and parse the test file
    ev_classes, ev_stamps = read_events(XMLFOLDER, 66)

def test_read_signal():
    """ Read & load a MIVIA database audio signal """
    # Path to the WAV file
    ROOT = f"{aer.datasets.__path__[0]}/MIVIA_AE_example/training/"
    SNDFOLDER = ROOT + "sounds/"
    # Dataset index and SNR (only available one in the example training set)
    IND, SNR = 66, 15
    # Load the "00066_3.wav" file with an SNR of 15 dB
    frate, signal = read_signal(SNDFOLDER, IND, SNR)
    # Load & rescale the data
    frate, signal_norm = read_signal(SNDFOLDER, IND, SNR,
        limits=(-2**15, 2**15-1), bounds=(-1., 1.), dtype=np.float32)

def test_build_dataset():
    """ Build the data & class chunks or sequences """
    # Path to the XML & WAV files
    ROOT = f"{aer.datasets.__path__[0]}/MIVIA_AE_example/training/"
    XMLFOLDER = ROOT                            # Events XML files folder
    SNDFOLDER = ROOT + "sounds/"                # Audio WAV files folder
    # Dataset index and SNR (only one available in the example training set)
    IND, SNR = 66, 15
    # Chunk & Sequences parameters
    frm_duration = 50e-3    # Frame duration in seconds
    chk_params = (frm_duration, frm_duration)
    seq_params = (10, 1)
    # Number of possible classes
    nb_classes = 4
    # Load the data and build the chunks
    specs, data_chks, class_chks = build_dataset(
        (XMLFOLDER, SNDFOLDER), (IND, SNR), nb_classes, chk_params)
    # Load the data and build the sequences
    specs, data_seqs, class_seqs = build_dataset(
        (XMLFOLDER, SNDFOLDER), (IND, SNR), nb_classes, chk_params, seq_params)
    # Load the data, normalize them and and build the chunks
    specs, data_chks, class_chks = build_dataset(
        (XMLFOLDER, SNDFOLDER), (IND, SNR), nb_classes, chk_params,
        limits=(-2**15, 2**15-1), bounds=(-1., 1.), dtype=np.float32)

def test_overall():
    """ Didactic example on how to use the MIVIA data loader """
    import matplotlib.pyplot as plt
    import aer.data_tk as dtk

    # Data folders and data files
    ROOT = aer.datasets.__path__[0]+"/MIVIA_AE_example/training/"
    XMLFOLDER = ROOT                            # Events XML files folder
    SNDFOLDER = ROOT + "sounds/"                # Audio WAV files folder
    NB_CLASSES = 4                              # Nb of possible classes
    # Set the dataset index and SNR (only available one in the example set)
    IND, SNR = 66, 15
    # Load the signal for testing
    frate, signal = read_signal(SNDFOLDER, IND, SNR)
    ev_classes, ev_tstamps = read_events(XMLFOLDER, IND)

    #-------------------------   Spectral Density   -------------------------#
    # Build the FFT and the spectral power distribution
    freq = np.fft.rfftfreq(len(signal))*frate
    sfft = np.fft.rfft(signal)
    pxx = (sfft.real)**2 / len(signal)
    # Plot the spectral density for every file
    plt.figure(figsize=(9.00, 10.00))
    plt.plot(freq[:len(sfft)//6], pxx[:len(sfft)//6])
    plt.ylim([0., 1e9])
    plt.tight_layout()
    plt.savefig(f"pxx_{IND}_{SNR}.png")
    plt.close()
    #------------------------------------------------------------------------#

    #-------------------   Load Dataset & Build Chunks   --------------------#
    # Choose the input data format
    SEQUENCES = False
    # Set the frame & hop durations
    if SEQUENCES:
        FRM_DURATION = 50e-3                        # Frame duration in seconds
        HOP_DURATION = FRM_DURATION                 # Hop duration in seconds
        chk_params = (FRM_DURATION, FRM_DURATION)   # Chunks settings
        seq_params = (10, 1)                        # Sequences settings
    else:
        FRM_DURATION = 100e-3                       # Frame duration in seconds
        HOP_DURATION = 50e-3                        # Hop duration in seconds
        chk_params = (FRM_DURATION, HOP_DURATION)   # Chunks settings
        seq_params = None
    #------------------------------------------------------------------------#

    #-------------------   Build (Sequences of) Chunks   --------------------#
    # Manually build the datasets
    chk_params = (int(chk_params[0]*frate), int(chk_params[1]*frate))
    classes = dtk.chunk_classes(
        ev_tstamps * frate, ev_classes, len(signal), *chk_params)
    if SEQUENCES:
        data_set = dtk.data_sequences(signal, chk_params, seq_params)
        class_set = dtk.class_sequences(classes, NB_CLASSES, seq_params)
    else:
        data_set = dtk.data_chunks(signal, chk_params)
        class_set = dtk.class_chunks(classes, NB_CLASSES)

    # Automatically load the data and build the (sequences of) chunks
    specs, data, classes = build_dataset(
        (XMLFOLDER, SNDFOLDER), (IND, SNR),
        NB_CLASSES, chk_params, seq_params)
    #------------------------------------------------------------------------#



# Launch test/example functions
test_parse_xml()

test_read_events()

test_read_signal()

test_build_dataset()

test_overall()

