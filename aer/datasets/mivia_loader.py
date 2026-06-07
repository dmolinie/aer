""" MIVIA data loader toolkit

Authors: Dylan MOLINIE
Company: CEA Paris-Saclay, Gif-sur-Yvette, France
Contact: dylan.molinie@cea.fr
Date: March 2025
Last revised: March 2026

License: GPLv3

MIVIA data limits: -2^{15} --> +2^{15}-1
Event categories: 0: background, 1: glass, 2: gunshot, 3: screams
"""

__all__ = ['SNR', 'LIMITS', 'CATEGORIES',
    'parse_xml', 'read_events', 'read_signal', 'build_dataset']

import os as _os
import xml.etree.ElementTree as _ET

import numpy as _np
from scipy.io import wavfile as _wavfile

from aer.data_tk import _scaling as _scl
from aer.data_tk import _data_parser as _dps

# Corresp. between the MIVIA soundfiles extension and their SNR
SNR = {5: '1', 10: '2', 15: '3', 20: '4', 25: '5', 30: '6'}

# MIVIA AE data lower & upper limits
LIMITS = (-2**15, 2**15-1)

# Corresp. between the class numbers and their meaning
CATEGORIES = {0: 'background', 1: 'glass', 2: 'gunshot', 3: 'screams'}


##############################################################################
##                             MIVIA XML Parser                             ##
##############################################################################

def parse_xml(xmlfile):
    """ Parse an XML file from the MIVIA AE or RAE databases

    Take the name of an XML file of the MIVIA Audio Event (AE) or Road
    Audio Event (RAE) databases, parse them and extract the categories
    (classes) and time stamps of start and end of the events.

    Since the XML files were generated in MATLAB, the categories start
    from 1; thus subtract each of them by 1 to make them start from 0.

    Parameters
    ----------
    xmlfile : str
        The name of the XML file to read, containing N events.

    Returns
    -------
    classes : np.ndarray
        The N-long vector representing the classes of the events.
    tstamps : np.ndarray
        The Nx2-long array containing the timestamps of the start and
        end of the events.

    Examples
    --------
    >>> classes, tstamps = parse_xml(<mivia_file.xml>)
    """

    # Read & parse the XML file
    tree = _ET.parse(xmlfile)
    root = tree.getroot()

#    # Get the type of the background
#    backgnd = root[1][0][2].text

    # Get the events type & times
    classes = _np.empty(len(root[0]), int)
    tstamps = _np.empty((len(root[0]), 2), float)

    # Autocasts are faster, but less comprehensive
    for i, event in enumerate(root[0]):
        classes[i] = int(event[1].text) - 1     # Event start from 1
        tstamps[i] = float(event[3].text), float(event[4].text)

    return classes, tstamps
#    return backgnd, classes, tstamps

##############################################################################



##############################################################################
##                       Read & Load XML & WAV Files                        ##
##############################################################################

#------------------------   Read & Parse XML File   -------------------------#
def read_events(folder, index):
    """ Read & parse a MIVIA XML events file

    Take the folder containing the XML files and the index of the audio
    signal, and read & parse the corresponding XML file to retrieve the
    events start and end times and their respective classes.

    This function is essentially a wrapper for the `parse_xml` function.

    Parameters
    ----------
    folder : string
        The relative or absolute path to the folder containing the XML
        files; should end with an OS-specific delimiter.
    index : int
        The index (number) of the XML file to read.

    Returns
    -------
    ev_classes : np.ndarray
        The N-long vector representing the classes of the events.
    ev_tstamps : np.ndarray
        The Nx2-long array containing the timestamps of the start and
        end of the events.

    Examples
    --------
    # The aer project `datasets` package contains an example of the MIVIA
    # dataset, composed of two subdatasets of 1 file each: files 66 for
    # "training" and 29 for "testing", both with an SNR of 15 ('3').

    >>> import aer.datasets

    # Path to the XML file
    >>> ROOT = f"{aer.datasets.__path__[0]}/MIVIA_AE_example/training/"
    >>> XMLFOLDER = ROOT

    # Dataset index (only available one in the example training set)
    >>> IND = 66

    # Read and parse the test file
    >>> ev_classes, ev_stamps = read_events(XMLFOLDER, IND)
    """

    # Check that the data folder exists
    if not _os.path.exists(folder):
        print("XML files folder not found, returning nothing")
        return None

    # Get the events classes and start and end times
    return parse_xml(_os.path.join(folder, f"{index:05d}.xml"))
#----------------------------------------------------------------------------#

#-------------------------   Load WAV Sound File   --------------------------#
def read_signal(folder, index, snr=30, **rescale):
    """ Read & load a MIVIA database audio signal

    Take the folder containing the audio files, the index of the sound
    signal and the desired SNR, and read & load the corresponding audio
    file to retrieve both the sampling rate and the signal components.

    In the MIVIA dataset, any signal is available with six SNRs: 5, 10,
    15, 20, 25 and 30 dB; each SNR corresponds to a file extension, e.g.
    `X_1` or 5 dB or `X_6` for 30 dB; the correspondence between these
    numbers and the file extensions are listed in the `SNR` global var.

    Parameters
    ----------
    folder : string
        The relative or absolute path to the folder containing the sound
        files; should end with an OS-specific delimiter.
    index : int
        The index (number) of the sound file to read.
    [OPT] snr : int
        The Signal-to-Noise Ratio of the file to read.
            :Default: 30 (extension `X_6`)

    Other Parameters
    ----------------
    [GLB] SNR : dict
        The `SNR` global variable, which defines the MIVIA equivalence
        btw any file extension index and the real SNR value; the `snr`
        arg. is used as a key for `SNR` to access the equivalent index.
    **rescale : inline keyword arguments
        The rescaling parameters, passed to the `rescale` function from
        the `data_tk` module; if left empty, do not rescale the data.
        Possible arguments are (see the `rescale` function for details):
          - limits (2-tuple): data lower & upper bounds (e.g. -2^n & 2^n-1)
          - bounds (2-tuple): new lower & upper bounds (e.g. 0 & 1)
          - axis (int): axis along which to get the min & max, if needed
          - dtype (data type): data type to use to format the arrays.

    Returns
    -------
    frate : float
        The signal sampling frequency.
    signal : 1D np.ndarray of floats
        The (possible rescaled) signal data sensu stricto.

    Examples
    --------
    # The aer project `datasets` package contains an example of the MIVIA
    # dataset, composed of two subdatasets of 1 file each: files 66 for
    # "training" and 29 for "testing", both with an SNR of 15 ('3').

    >>> import aer.datasets

    # Path to the WAV file
    >>> ROOT = f"{aer.datasets.__path__[0]}/MIVIA_AE_example/training/"
    >>> SNDFOLDER = ROOT + "sounds/"

    # Dataset index and SNR (only available one in the example training set)
    >>> IND, SNR = 66, 15

    # Load the "00066_3.wav" file with an SNR of 15 dB
    >>> frate, signal = read_signal(SNDFOLDER, IND, SNR)

    # Load & rescale the data
    >>> frate, signal_norm = read_signal(SNDFOLDER, IND, SNR,
    ...     limits=(-2**15, 2**15-1), bounds=(-1., 1.), dtype=np.float32)
    """

    # Check that the data folder exists
    if not _os.path.exists(folder):
        print("Sound files folder not found, returning nothing")
        return None

    # Read and load the frequency rate and the soundfile (wav)
    frate, signal = _wavfile.read(_os.path.join(folder, f"{index:05d}_{SNR[snr]}.wav"))

    # If any rescaling parameters is provided, rescale the data
    if len(rescale) != 0:
        signal = _scl.rescale(signal, **rescale)[0]

    # Return the sampling frequency & the signal data
    return frate, signal
#----------------------------------------------------------------------------#

#----------------------------   Build Dataset   -----------------------------#
def build_dataset(
    folders, file_params, nb_classes, chk_params, seq_params=None, **rescale):
    """ Build the data & class chunks or sequences

    Take the folders of the XML & WAV files and the chunks and sequences
    parameters (length & hop) and build the sequences of chunks from the
    audio data, and the sequences of classes from the XML file events.

    It is a wrapper for the `read_signal` and `read_events` functions;
    see these functions for details.

    If the sequences parameters are missed out (None), build only the
    chunks, do not gather them into sequences, return them directly.

    N.B.: the two parameters of both `chk_params` & `seq_params` should
        be numbers of samples (e.g. the amount of samples in a frame).
        If floating values are provided, assume they are time lapses,
        and multiply them by the sampling frequency (retrieved when
        reading the dataset) and convert them into integers.

    Parameters
    ----------
    folder : 2-tuple of strings
        The paths to the folders containing the XML file listing the ev-
        ents (first component) and to the WAV file containing the audio
        signal data (second component).
    file_params : 2-tuple of ints
        The index (number) of the sound file to read and the SNR; for
        instance, (1, 30) to load the "00001_6.wav" file.
    nb_classes : int
        The number of possible classes, for the SoftMax function.
    chk_params : 2-tuple of ints
        The chunks format, organized as (chk_win_size, chk_hop_size).
        Both values should be amounts of samples (integers).
    [OPT] seq_params : 2-tuple of ints
        The sequences format, organized as (seq_win_size, seq_hop_size).
        If None, do not build sequences, build only data chunks and the
        vectors of probabilities of their classes. Both values should be
        amounts of samples (integers).
            :Default: None

    Other Parameters
    ----------------
    [GLB] SNR : dict
        The `SNR` global variable, which defines the MIVIA equivalence
        btw any file extension index and the real SNR value; the `snr`
        arg. is used as a key for `SNR` to access the equivalent index.
    **rescale : inline keyword arguments
        The rescaling parameters, passed to the `rescale` function from
        the `data_tk` module; if left empty, do not rescale the data.
        Possible arguments are (see the `rescale` function for details):
          - limits (2-tuple): data lower & upper bounds (e.g. -2^n & 2^n-1)
          - bounds (2-tuple): new interval lower & upper (e.g. 0 & 1)
          - axis (int): axis along which to get the min & max, if needed
          - dtype (data type): data type to use to format the arrays.

    Returns
    -------
    specs : 2-tuple of ints
        The sampling frequency (1st component) and number of samples in
        the original (not padded) signal used to build the chunks (2nd).
        Can optionally be used to build the chunks of timestamps.
    data_seqs : 2D or 3D np.ndarray of floats, depending on `seq_params`
        The sequences of chunks, organized as K sequences of M chunks of
        N samples each, where N is the first component of `chk_params`,
        and K and M are determined dynamically. If `seq_params` is None,
        no sequence is built, and the chunks are directly returned.
    class_seqs : 2D or 3D np.ndarray of floats, depending on `seq_params`
        The sequences of probabilities for a chunk to belong to any pos-
        sible class, with 1.0 at the chunk class used as component index,
        and 0.0 anywhere else. If `seq_params` is None, the probabilities
        are regrouped into a 2D array, one row per chunk, where a row is
        the vector of probabilities for the chunk to belong to any class.

    Examples
    --------
    # The aer project `datasets` package contains an example of the MIVIA
    # dataset, composed of two subdatasets of 1 file each: files 66 for
    # "training" and 29 for "testing", both with an SNR of 15 ('3').

    >>> import aer.datasets

    # Path to the XML & WAV files
    >>> ROOT = f"{aer.datasets.__path__[0]}/MIVIA_AE_example/training/"
    >>> XMLFOLDER = ROOT                    # Events XML files folder
    >>> SNDFOLDER = ROOT + "sounds/"        # Audio WAV files folder

    # Dataset index and SNR (only available one in the example training set)
    >>> IND, SNR = 66, 15

    # Chunk & Sequences parameters
    >>> frm_duration = 50e-3    # Frame duration in seconds
    >>> chk_params = (frm_duration, frm_duration)
    >>> seq_params = (10, 1)

    # Number of possible classes
    >>> nb_classes = 4

    # Load the data and build the chunks
    >>> specs, data_chks, class_chks = build_dataset(
    ...     (XMLFOLDER, SNDFOLDER), (IND, SNR), nb_classes, chk_params)

    # Load the data and build the sequences
    >>> specs, data_seqs, class_seqs = build_dataset(
    ...     (XMLFOLDER, SNDFOLDER), (IND, SNR), nb_classes, chk_params, seq_params)

    # Load the data, normalize them and and build the chunks
    >>> specs, data_chks, class_chks = build_dataset(
    ...     (XMLFOLDER, SNDFOLDER), (IND, SNR), nb_classes, chk_params,
    ...     limits=(-2**15, 2**15-1), bounds=(-1., 1.), dtype=np.float32)
    """

    # Retrieve the events contained inside the XML file
    ev_classes, ev_tstamps = read_events(folders[0], file_params[0])

    # Load the audio signal from the WAV file, and rescale its data if needed
    frate, signal = read_signal(
        folders[1], file_params[0], file_params[1], **rescale)

    # Check if the params are nb of samples (ints) or timestamps (floats)
    # If they are timestamps, convert them into number of samples
    if isinstance(chk_params[0], float) and isinstance(chk_params[1], float):
        chk_params = (int(chk_params[0]*frate), int(chk_params[1]*frate))

    # If no sequence demanded, build the signal chunks only
    if seq_params is None:
        data_set = _dps.data_chunks(signal, chk_params)
    # Else, build the signal chunks and gather them into sequences
    else:
        data_set = _dps.data_sequences(signal, chk_params, seq_params)

    # Determine the classes of the chunks and that of the events
    classes = _dps.chunk_classes(
        ev_tstamps * frate, ev_classes, len(signal), *chk_params)

    # If no sequence demanded, get the class of every chunk separately
    if seq_params is None:
        class_set = _dps.class_chunks(classes, nb_classes)
    # Else, build the sequences of chunk classes
    else:
        class_set = _dps.class_sequences(classes, nb_classes, seq_params)

    return (frate, len(signal)), data_set, class_set
#----------------------------------------------------------------------------#

##############################################################################
