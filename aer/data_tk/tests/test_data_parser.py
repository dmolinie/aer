import numpy as np
from aer.data_tk._data_parser import *


def test_pad_offset():
    """ Number of samples to be added to the vector """
    print(pad_offset(100, 50, 10))
    print(pad_offset(100, 25, 10))
    print(pad_offset(100, 26, 13))
    print(pad_offset(100, 33, 13))

def test_pad():
    """ Distribute the offset samples to both sides of the data vector """
    print(pad([1, 2, 3], 2))
    print(pad([1, 2, 3], 3, 5))
    print(pad([1, 2, 3], 3, 9, 2))
    print(pad([1, 2, 3], 3, value=123))

def test_get_hop_size():
    """ Retrieve the hop size from a set of chunks """
    chunks = build_chunks(np.arange(0., 100.), 50, 10)
    print(get_hop_size(chunks))

def test_get_hop_size_data():
    """ Retrieve the hop size from the data vector & chunks shapes """
    data = np.arange(512)
    win_size = 100
    hop_size = 50
    data = pad(data, pad_offset(len(data), win_size, hop_size))
    chunks = build_chunks(data, win_size, hop_size, False)
    hop_size_bak = get_hop_size_data(*chunks.shape, len(data))

def test_build_chunks():
    """ Build the chunks using a unit sliding window """
    data = np.arange(100) + 1                 # Generate the data vector
    chks = build_chunks(data, 50, None)       # 10%+1 (6 samples) overlap
    chks = build_chunks(data, 50, 10)         # 10 samples overlap
    chks = build_chunks(data, 50, 50)         # No overlap
    chks = build_chunks(data, 50, 50, True)   # 50 zeros appended (3 chunks)
    chks = build_chunks(data, 50, 50, False)  # 0 zero appended (2 chunks)

def test_chunk_classes():
    """ Build the vector of the classes of the signal chunks """
    # General parameters
    frate = 800                             # Sampling rate
    frm_duration = 50e-3                    # Frame duration in secs
    win_size = int(frm_duration*frate)      # Nb of samples per frame
    hop_size = win_size // 2
    # Generate the signal
    signal = np.arange(0., 10.09, 1./frate) + 0.1
    # Simulate some events
    # (whose stamps are [[752, 1256], [2768, 3325], [5560, 6509]])
    ev_stamps = np.array([[0.94, 1.57], [3.46, 4.15625], [6.95, 8.13625]])
    # Set the event classes (classes start from 1, `0` is the background)
    ev_classes = np.arange(len(ev_stamps)) + 1
    #---- No overlap between the chunks
    chunks = build_chunks(signal, win_size, win_size, False)
    classes = chunk_classes(
        ev_stamps * frate, ev_classes, len(signal), win_size)
    #---- Overlapping chunks
    chunks = build_chunks(signal, win_size, hop_size, False)
    classes = chunk_classes(
        ev_stamps * frate, ev_classes, len(signal), win_size, hop_size)

def test_data_chunks():
    """ Build 2D chunks of 1D vectors """
    chk_params = (20, 10)   # Chunk parameters
    # Generate the 1D data vector
    data = np.arange(100) + 1
    # Build the sequences of chunks
    data_chks = data_chunks(data, chk_params)

def test_class_chunks():
    """ Build the chunk classes in a one-hot encoding fashion """
    # General parameters
    frate = 40                              # Sampling rate
    frm_duration = 200e-3                   # Frame duration in secs
    nb_classes = 4                          # Number of possible classes
    win_size = int(frm_duration*frate)      # Nb of samples per frame
    hop_size = win_size // 2                # Hop size
    chk_params = (win_size, hop_size)
    # Generate the signal
    signal = np.arange(0., 10.09, 1./frate) + 0.1
    # Simulate some events
    # (whose stamps are [[752, 1256], [2768, 3325], [5560, 6509]])
    ev_stamps = np.array([[0.94, 1.57], [3.46, 4.15625], [6.95, 8.13625]])
    # Set the event classes (classes start from 1, 0 is the background)
    ev_classes = np.arange(len(ev_stamps)) + 1
    # Determine the classes of the chunks and that of the events
    classes = chunk_classes(
        ev_stamps * frate, ev_classes, len(signal), *chk_params)
    # Build the probability vectors of classes
    class_chks = class_chunks(classes, nb_classes)
    # Build the chunks of data (optional)
    data_chks = data_chunks(signal, chk_params)

def test_data_sequences():
    """ Build 3D sequences of 2D chunks from a 1D data vector """
    chk_params = (20, 10)   # Chunk parameters
    seq_params = (10, 2)    # Sequence parameters
    # Generate the 1D data vector
    data = np.arange(100) + 1
    # Build the sequences of chunks
    data_seqs = data_sequences(data, chk_params, seq_params)

def test_class_sequences():
    """ Build the sequences of classes in a one-hot encoding fashion """
    # General parameters
    frate = 40                              # Sampling rate
    frm_duration = 200e-3                   # Frame duration in secs
    nb_classes = 4                          # Number of possible classes
    win_size = int(frm_duration*frate)      # Nb of samples per frame
    hop_size = win_size // 2                # Hop size
    chk_params = (win_size, hop_size)
    seq_params = (10, 3)
    # Generate the signal
    signal = np.arange(0., 10.09, 1./frate) + 0.1
    # Simulate some events
    # (whose stamps are [[752, 1256], [2768, 3325], [5560, 6509]])
    ev_stamps = np.array([[0.94, 1.57], [3.46, 4.15625], [6.95, 8.13625]])
    # Set the event classes (classes start from 1, 0 is the background)
    ev_classes = np.arange(len(ev_stamps)) + 1
    # Determine the classes of the chunks and that of the events
    classes = chunk_classes(
        ev_stamps * frate, ev_classes, len(signal), *chk_params)
    # Build the sequences of classes
    class_seqs = class_sequences(classes, nb_classes, seq_params)
    # Build the sequences of data (optional)
    data_seqs = data_sequences(signal, chk_params, seq_params)



# Launch test/example functions
test_pad_offset()

test_pad()

test_get_hop_size()

test_get_hop_size_data()

test_build_chunks()

test_chunk_classes()

test_data_chunks()

test_class_chunks()

test_data_sequences()

test_class_sequences()

