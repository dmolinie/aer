""" Tools to parse a continuous 1D data array into (sequences) of chunks

Authors: Dylan MOLINIE
Company: CEA Paris-Saclay, Gif-sur-Yvette, France
Contact: dylan.molinie@cea.fr
Date: March 2025
Last revised: March 2026

License: GPLv3
"""

__all__ = [
    'pad_offset', 'pad', 'get_hop_size', 'get_hop_size_data',
    'get_nb_chunks', 'build_chunks', 'chunk_classes',
    'data_chunks', 'class_chunks', 'data_sequences', 'class_sequences']

import numpy as np


##############################################################################
##                              Signal Chunks                               ##
##############################################################################

#-----------------------------   Data Padding   -----------------------------#
def pad_offset(vec_size, win_size, hop_size):
    """ Number of samples to be added to the vector

    Provide the number of samples which should be added to the input
    data vector so that (vec_size - win_size) / hop_size is an integer.
    The returned quantity `off` should be added to vec_size, so that:
        vec_size_pad = vec_size + off
        (vec_size_pad - win_size) % hop_size = 0

    Parameters
    ----------
    vec_size : int
        The size in samples of the input (data) vector.
    win_size : int
        The size in samples of the sliding window.
    hop_size : int
        The hop size in samples, i.e. the window offset size.

    Returns
    -------
    off : int
        The number of samples to be added to the vectors.

    Examples
    --------
    >>> pad_offset(100, 50, 10)    # 0
    >>> pad_offset(100, 25, 10)    # 5
    >>> pad_offset(100, 26, 13)    # 4
    >>> pad_offset(100, 33, 13)    # 11
    """

    # Euclidean division remainders (edr)
    edr_nh = vec_size % hop_size
    edr_wh = win_size % hop_size

    # Divisibility
    if (vec_size - win_size) % hop_size == 0:       # H|N & H|W
        return 0
    if edr_nh != 0 and edr_wh != 0:                 # Not H|N & not H|W
        shift = edr_wh - edr_nh
        return shift if shift > 0 else hop_size + shift
    if edr_nh != 0:                                 # Not H|N & H|W
        return hop_size - edr_nh
    return edr_wh                                   # H|N & not H|W

def pad(data, offset, value=0., win_size=1):
    """ Distribute the offset samples to both sides of the data vector

    Take the input data vector and add N * win_size components to both
    sides of it (beginning and end). The value provided as `offset` is
    divided by two; if `offset` is even, set N to offset//2; else, add
    one more data at the end than at the beginning. The padding adds N
    groups of `win_size` constant components, whose values are defined
    by the `value` argument.

    Parameters
    ----------
    data : array_like
        The input data vector.
    offset : int
        The total number of samples to add.
    [OPT] value : scalar or array of scalars
        The padding values; if `0`, pad the `data` vector with zeros; if
        any scalar, pad it with this value instead.
            :Default: 0. (add 0s to both sides)
    [OPT] win_size : int
        The number of samples to add for each `offset`; if 1, simply add
        `offset//2` 0s to the beginning, and `offset - offset//2` to the
        end; else, multiply these quantities by `win_size`.
            :Default: 1

    Returns
    -------
    paddata : np.ndarray
        The padded array (completed with `0s`).

    Examples
    --------
    >>> pad([1, 2, 3], 2)             # [0, 1, 2, 3, 0]
    >>> pad([1, 2, 3], 3, 5)          # [5, 1, 2, 3, 5, 5]
    >>> pad([1, 2, 3], 3, 9, 2)       # [9, 9, 1, 2, 3, 9, 9, 9, 9]
    >>> pad([1, 2, 3], 3, value=123)  # [123, 1, 2, 3, 123, 123]
    """
    off = abs(int(offset)//2)
    return np.pad(data, (off*win_size, (offset-off)*win_size),
                  'constant', constant_values=value)
#----------------------------------------------------------------------------#

#-----------------------   Retrieve Hop From Chunks   -----------------------#
def _get_hop_size(chunk1, chunk2):
    """ Retrieve the hop size from two chunks """
    # Compare every component of chunk1 until it matches the first of chunk2
    # and return as hop size the number of times this comparison failed
    hop = 0
    while hop < len(chunk1) and chunk1[hop] != chunk2[0]:
        hop += 1
    return hop

def get_hop_size(chunks):
    """ Retrieve the hop size from a set of chunks

    Take a set of chunks and retrieve their hop size. To do so, take two
    consecutive chunks, count the nb of components from the first chunk
    which do no match with the first component of the second chunk, and
    use this value as hop size. To ensure it is the true hop size (e.g.
    the same value can be in two chunks but not be the same data), com-
    pute the hop size for two sets of two consecutive chunks and stop
    only if they are the same, or if all the sets have been tested.

    Parameters
    ----------
    chunks : 2D np.ndarray
        The data chunks (slices), organized as `nb_chunks x chunks`,
        with `chunks` the chunks of size `win_size`.

    Returns
    -------
    hop : int
        The (estimated) hop size

    Examples
    --------
    >>> chunks = build_chunks(np.arange(0., 100.), 50, 10)
    >>> get_hop_size(chunks)  # 10
    """
    col, hop_bfr, hop_aft = 2, 0, 1
    while hop_bfr != hop_aft and col < len(chunks):
        hop_bfr = _get_hop_size(chunks[col-2], chunks[col-1])
        hop_aft = _get_hop_size(chunks[col-1], chunks[col])
        col += 1
    return hop_aft

def get_hop_size_data(nb_chunks, win_size, vec_size):
    """ Retrieve the hop size from the data vector & chunks shapes

    Parameters
    ----------
    nb_chunks : int
        The number of chunks.
    win_size : int
        The number of samples in any chunk.
    vec_size : int
        The number of samples in the original (padded) vector.

    Returns
    -------
    hop_size : int
        The chunks hop size.

    Examples
    --------
    >>> import numpy as np

    >>> data = np.arange(512)
    >>> win_size = 100
    >>> hop_size = 50

    >>> data = pad(data, pad_offset(len(data), win_size, hop_size))
    >>> chunks = build_chunks(data, win_size, hop_size, False)
    >>> hop_size_bak = get_hop_size_data(*chunks.shape, len(data))
    """
    return (vec_size - win_size) // (nb_chunks - 1)
#----------------------------------------------------------------------------#

#-----------------------------   Build Chunks   -----------------------------#
def get_nb_chunks(vec_size, win_size, hop_size):
    """ Number of chunks in the input data vector (cf. `pad_offset`) """
    return (vec_size - win_size) // hop_size + 1

def build_chunks(data, win_size, hop_size=None, add_hop=True):
    """ Build the chunks using a unit sliding window

    Parameters
    ----------
    data : array_like
        The input data vector. Should be padded (cf. `pad` function);
        pad them automatically if they are not.
    win_size : int
        The size in samples of the sliding window.
    [OPT] hop_size : int
        The hop size in samples, i.e. the window offset size. If set
        to None, use 10% of `win_size` plus 1 (to ensure it is > 0).
        To have a null overlap, set `hop_size` to `win_size`.
            :Default: None
    [OPT] add_hop : bool
        If `hop_size` zeros should be appended to the `data` vector, in
        addition to those already needed to allow the split into chunks.
            :Default: True (always add `hop_size` zeros)

    Returns
    -------
    chunks : 2D np.ndarray
        The data chunks (slices), organized as `nb_chunks x chunks`,
        with `chunks` the chunks of size `win_size`.

    Examples
    --------
    >>> import numpy as np

    >>> data = np.arange(100) + 1                 # Generate the data vector

    >>> chks = build_chunks(data, 50, None)       # 10%+1 (6 samples) overlap
    >>> chks = build_chunks(data, 50, 10)         # 10 samples overlap
    >>> chks = build_chunks(data, 50, 50)         # No overlap

    >>> chks = build_chunks(data, 50, 50, True)   # 50 zeros appended (3 chunks)
    >>> chks = build_chunks(data, 50, 50, False)  # 0 zero appended (2 chunks)
    """

    # Check hop size (must be > 0)
    if hop_size in (None, 0):
        hop_size = win_size // 10 + 1           # +1 to ensure hop > 0

    # Additional zeros to append to the data to be padded
    shift = hop_size if add_hop else 0

    # Pad data if needed
    data = pad(data, shift + pad_offset(len(data), win_size, hop_size))

    # Number of chunks
    nb_chunks = get_nb_chunks(len(data), win_size, hop_size)

    # Build the chunks
    chunks = np.empty((nb_chunks, win_size), float)
    for i in range(nb_chunks):
        pos = i * hop_size                      # Jump to next chunk
        chunks[i] = data[pos:pos+win_size]      # Current slice

    return chunks
#----------------------------------------------------------------------------#

##############################################################################



##############################################################################
##                           Build Events Classes                           ##
##############################################################################

#-------------------------   Build Chunks Classes   -------------------------#
def __classes_round(ev_chunks):
    """ Round the 1st and 2nd cols to the nearest lower and upper ints """
    ev_chunks[:, 0] = np.round(ev_chunks[:, 0])     # Nearest ints
    ev_chunks[:, 1] = np.round(ev_chunks[:, 1])     # Nearest ints
    return np.array(ev_chunks, int)                 # Convert to ints

def _classes_nooverlap(ev_stamps, offset, win_size):
    """ Build the classes of the signal chunks when no overlap """
    # Get the relative indices of the chunks containing the stamps
    return __classes_round((ev_stamps + offset) / win_size)

def _classes_overlap(ev_stamps, offset, win_size, hop_size):
    """ Build the classes of the signal chunks when there is an overlap """

    # Determine the start and end chunks containing each of the events
    # and remove the negative chunks possibly created by windowing
    ind_chk = (ev_stamps - win_size + hop_size + offset) // hop_size
    ind_chk[ind_chk < 0] = 0

    # Compute the relative position of the stamps inside the chunks
    ev_chunks = ind_chk + (ev_stamps - ind_chk * hop_size + offset) / win_size

    # If null step (win_size == hop_size), shift nothing, round the chunk
    # relative positions and return them as integers
    # N.B.: if win_size == hop_size, prefer `_classes_nooverlap` function
    if hop_size == win_size:
        return __classes_round(ev_chunks)

    # Shift (in relative quantity) when a sample appears several times
    step = 1. - hop_size / win_size         # Step btw two consecutive occur.
    shift = (win_size // hop_size) * step   # Shift btw first and last occur.

    # Get the first valid occurrence for the start and end of the events
    for i in range(len(ev_chunks)):

        # Starting from the first occurrence of a stamp, add as many `step`
        # as necessary until finding the first occurrence contained in the
        # first half of the chunk (i.e. its fractional part is < 0.5)
        while np.round(ev_chunks[i, 0] % 1., 6) > 0.5:
            ev_chunks[i, 0] += step

        # Starting from the last occurrence of a stamp, subtract as many
        # `step` as needed until finding the first occurrence contained in
        # the second half of the chunk (i.e. its fractional part is > 0.5)
        ev_chunks[i, 1] += shift-step
        while np.round(ev_chunks[i, 1] % 1., 6) < 0.5:
            ev_chunks[i, 1] -= step

    # Round the fractional positions to the closest integers
    return __classes_round(ev_chunks)

def chunk_classes(ev_stamps, ev_classes, sig_length, win_size, hop_size=None):
    """ Build the vector of the classes of the signal chunks

    Considering a signal is split into M chunks, build a M-long vector
    in which the i-th component is the class of the i-th chunk.

    The original signal is assumed to contain N events; every event is
    assumed to be delimited by its start and end indices in the signal:
    its start and end are represented by ints which also designate the
    pos. of the event first and last data in the signal, respectively;
    these events are assumed listed in the Nx2-long `ev_stamps` array,
    in which each row contains the start and end indices of an event.
    The classes corresponding to the events are assumed listed in the
    N-long vector `ev_classes`; e.g., `ev_classes[0]` is the class of
    the event bounded by the indices contained in `ev_stamps[0]`.

    Instantiate a `classes` vector of the same length as the amount of
    chunks and initialize its components to `0` (assuming class `0` is
    `background`); then, for an event, identify the chunks across which
    it spans and set the corresponding components of `classes` to the
    class of that event. Also, if the start (end) of an event is at the
    end (start) of a chunk, ignore this chunk.

    Note that, since the original signal vector has likely been padded
    (i.e. zeros have been added to the vector so as to allow its split-
    ting into chunks), retrieve the amount of zeros added and shift the
    event delimiting indices by half the padding offset (since as many
    zeros are added to both the beginning and end of the signal).

    N.B.: the events must be delimited by indices; if their boundaries
        are times instead (the event start and end times), they should
        be multiplied by the sampling rate to obtain the corresponding
        indices. If `ev_stamps` are times, and by denoting `frate` the
        sampling frequency, provide `ev_stamps*frate` as 1st argument.

    Parameters
    ----------
    ev_stamps : 2D array (size of Nx2 values)
        The indices of the start and end of the events; if `frate` is
        provided, consider they are times and not indices and multiply
        the array's values by `frate` before processing.
    ev_classes : 1D array (N values)
        The class of each event (one class per event).
    sig_length : int
        The length of the original signal (i.e. its nb of components)
        to determine the padding offset.
    win_size : int
        The length of the chunk's window (nb of samples per chunk).
    [OPT] hop_size : int
        The hop size, i.e the nb of samples between two frames. If None,
        consider there is no overlap and use `win_size` as hop size.
            :Default: None

    Returns
    -------
    classes : 1D array of length `chunks_shape[0]`
        The classes of the chunks, one class per chunk; the classes are
        assumed to start from 0, with 0 reserved for `background`.

    Examples
    --------
    >>> import numpy as np

    # General parameters
    >>> frate = 800                             # Sampling rate
    >>> frm_duration = 50e-3                    # Frame duration in secs
    >>> win_size = int(frm_duration*frate)      # Nb of samples per frame
    >>> hop_size = win_size // 2

    # Generate the signal
    >>> signal = np.arange(0., 10.09, 1./frate) + 0.1

    # Simulate some events
    # (whose stamps are [[752, 1256], [2768, 3325], [5560, 6509]])
    >>> ev_stamps = np.array([[0.94, 1.57], [3.46, 4.15625], [6.95, 8.13625]])

    # Set the event classes (classes start from 1, `0` is the background)
    >>> ev_classes = np.arange(len(ev_stamps)) + 1

    #---- No overlap between the chunks
    >>> chunks = build_chunks(signal, win_size, win_size, False)
    >>> classes = chunk_classes(
    ...     ev_stamps * frate, ev_classes, len(signal), win_size)

    #---- Overlapping chunks
    >>> chunks = build_chunks(signal, win_size, hop_size, False)
    >>> classes = chunk_classes(
    ...     ev_stamps * frate, ev_classes, len(signal), win_size, hop_size)
    """

    # If no `hop_size`, consider there is no overlap nor gap btw the chunks
    if hop_size is None:
        hop_size = win_size

    # Compute the number of chunks considering the data are padded
    offset = pad_offset(sig_length, win_size, hop_size)
    nb_chunks = get_nb_chunks(sig_length+offset, win_size, hop_size)
    offset = offset // 2

    # Select the right function to build the classes, wrt to `hop_size` (if
    # win_size == hop_size, both `_classes_nooverlap` & `_classes_overlap`
    # provide the same results, but the former is much faster)
    if win_size == hop_size:
        ev_chunks = _classes_nooverlap(ev_stamps, offset, win_size)
    else:
        ev_chunks = _classes_overlap(ev_stamps, offset, win_size, hop_size)
        ev_chunks[ev_chunks >= nb_chunks] = nb_chunks   # Remove OOB indices

    # Set the class of the events contained in `ev_stamps` to their corres-
    # ponding classes contained in `ev_classes`, and to `0` for the others
    classes = np.zeros(nb_chunks, int)
    for event, cls in zip(ev_chunks, ev_classes):
        classes[event[0]:event[1]] = cls

    return classes
#----------------------------------------------------------------------------#

##############################################################################



##############################################################################
##                               Build Chunks                               ##
##############################################################################

#--------------------------   Build Data Chunks   ---------------------------#
def data_chunks(signal, chk_params):
    """ Build 2D chunks of 1D vectors

    Essentially an alias for the `build_chunks` function from `data_tk`
    module, which is directly called and returned, with the `add_hop`
    argument statically set to False. See this function of the details.

    Parameters
    ----------
    signal : 1D array
        The input data vector. Should be padded (cf. `pad` function);
        pad them automatically if they are not.
    chk_params : 2-tuple of ints
        The chunks format, organized as (win_size, hop_size). Both values
        should be amounts of samples (integers).

    Returns
    -------
    data_chks : 2D np.ndarray of floats
        The chunks built from the original `signal`, wrt `chk_params`.

    Examples
    --------
    >>> import numpy as np

    >>> chk_params = (20, 10)   # Chunk parameters

    # Generate the 1D data vector
    >>> data = np.arange(100) + 1

    # Build the sequences of chunks
    >>> data_chks = data_chunks(data, chk_params)
    """
    return build_chunks(signal, *chk_params, False)
#----------------------------------------------------------------------------#

#----------------------   Chunk Probability Vectors   -----------------------#
def class_chunks(classes, nb_classes):
    """ Build the chunk classes in a one-hot encoding fashion

    Take a 1D list of classes (typically outputted by `chunk_classes`)
    and the total nb of possible classes, and build the matrix of size
    `len(classes) x nb_classes`, in which each row represents the proba-
    bilities that the corresponding chunk belongs to any of the classes.
    In practice, fill the matrix with zeros, and, by referring to `cls`
    as the i-th component of `classes`, set the `cls` component of the
    i-th matrix row to 1.0, and leave the others at 0.0. Note that the
    classes are assumed to start from 0, with 0 for `background`.

    These values represent the probabilities for any chunk to belong to
    every of the `nb_classes`; for a given sequence chunk, they should
    sum to 1.0. In practice, they are compliant to the SoftMax output
    format, in particular when used as Neural Network output layer.

    Parameters
    ----------
    classes : 1D array of ints
        The classes of the signal chunks (frames); typically outputted
        by the `chunk_classes` function. Note that the classes should
        be contiguous and start from 0, with 0 denoting `background`.
        Any class category should be lower than `nb_classes`.
    nb_classes : int
        The number of possible classes; used as second dimension of the
        `sequences` output.

    Returns
    -------
    chk_classes : 2D np.ndarray of floats
        The vector of probabilities for a chunk to belong to any class,
        with 1.0 at the chunk class used as component index, and 0.0 else.

    Examples
    --------
    >>> import numpy as np

    # General parameters
    >>> frate = 40                              # Sampling rate
    >>> frm_duration = 200e-3                   # Frame duration in secs
    >>> nb_classes = 4                          # Number of possible classes

    >>> win_size = int(frm_duration*frate)      # Nb of samples per frame
    >>> hop_size = win_size // 2                # Hop size

    >>> chk_params = (win_size, hop_size)

    # Generate the signal
    >>> signal = np.arange(0., 10.09, 1./frate) + 0.1

    # Simulate some events
    # (whose stamps are [[752, 1256], [2768, 3325], [5560, 6509]])
    >>> ev_stamps = np.array([[0.94, 1.57], [3.46, 4.15625], [6.95, 8.13625]])

    # Set the event classes (classes start from 1, 0 is the background)
    >>> ev_classes = np.arange(len(ev_stamps)) + 1

    # Determine the classes of the chunks and that of the events
    >>> classes = chunk_classes(
    ...     ev_stamps * frate, ev_classes, len(signal), *chk_params)

    # Build the probability vectors of classes
    >>> class_chks = class_chunks(classes, nb_classes)

    # Build the chunks of data (optional)
    >>> #data_chks = data_chunks(signal, chk_params)
    """
    chk_classes = np.zeros((len(classes), nb_classes), float)
    for i, cls in enumerate(classes):
        chk_classes[i, cls] = 1.        # Set a 1.0 proba to the class
    return chk_classes
#----------------------------------------------------------------------------#

##############################################################################



##############################################################################
##                             Build Sequences                              ##
##############################################################################

#-------------------------   Build Data Sequences   -------------------------#
def data_sequences(data, chk_params, seq_params):
    """ Build 3D sequences of 2D chunks from a 1D data vector

    Take a 1D vector of data, split it into a set of chunks and gather
    them into sequences of chunks. The chunks and sequences structures
    should be specified in `chk_params` and `seq_params`, respectively;
    both should be 2-tuples, organized as (window_size, hop_size), i.e.
    the first value is the amount of samples in a chunk and the second
    the amount of samples separating two consecutive chunks (set it to
    the same value as the window size to have no overlap).

    If the data vector is st it cannot be split into an integer number
    of chunks, add as many zeros as necessary to both sides of it so
    that the padded vector can be split into an integer nb of chunks,
    which can themselves be regrouped into an integer nb of sequences.

    Parameters
    ----------
    data : 1D array
        The vector of data to split into sequences of chunks.
    chk_params : 2-tuple of ints
        The chunks format, organized as (chk_win_size, chk_hop_size).
        Both values should be amounts of samples (integers).
    seq_params : 2-tuple of ints
        The sequences format, organized as (seq_win_size, seq_hop_size).
        Both values should be amounts of samples (integers).

    Returns
    -------
    data_seqs : 3D np.ndarray
        The sequences of chunks, organized as K sequences of M chunks of
        N samples each, where N is the first component of `chk_params`,
        and K and M are determined dynamically.

    Examples
    --------
    >>> import numpy as np

    >>> chk_params = (20, 10)   # Chunk parameters
    >>> seq_params = (10, 2)    # Sequence parameters

    # Generate the 1D data vector
    >>> data = np.arange(100) + 1

    # Build the sequences of chunks
    >>> data_seqs = data_sequences(data, chk_params, seq_params)
    """

    # Number of 0s to be added to the data vector
    off_chk = pad_offset(len(data), *chk_params)

    # Number of chunks that will be built from the data vector
    nb_chks = get_nb_chunks(len(data) + off_chk, *chk_params)

    # Number of 0s to be added to the chunks
    off_seq = pad_offset(nb_chks, *seq_params)

    # Pad the data (add the 0. to both sides)
    data = pad(data, off_chk)                           # Fulfill chunks
    data = pad(data, off_seq, 0., chk_params[1])        # Fulfill sequences

    # Compute the final number of sequences after padding
    nb_seqs = get_nb_chunks(nb_chks + off_seq, *seq_params)

    # Build the sequences of chunks
    shift = seq_params[1] * chk_params[1]   # Nb of samples to skip btw 2 seqs
    data_seqs = np.zeros((nb_seqs, seq_params[0], chk_params[0]), float)
    for i in range(nb_seqs):
        pos = i * shift                                     # Next sequence
        for j in range(seq_params[0]):
            data_seqs[i, j] = data[pos:pos+chk_params[0]]   # Current chunk
            pos += chk_params[1]                            # Next chunk

    return data_seqs
#----------------------------------------------------------------------------#

#---------------------   Sequence Probability Vectors   ---------------------#
def class_sequences(classes, nb_classes, seq_params):
    """ Build the sequences of classes in a one-hot encoding fashion

    Take a 1D list of classes (typically outputted by `chunk_classes`),
    the number of possible classes and the parameters of the sequences
    (window and hop sizes), and build the sequences of classes. First,
    estimate the number of sequences wrt to the length of `classes` and
    the `seq_params` sequence parameters; then, instantiate a 3D array
    with shape `nb_sequences x length_sequence x nb_classes`; finally,
    for every chunk of a sequence, set the corresponding component of
    this vector to 1.0, and the others to 0.0. For example, if a chunk
    has class 2, set the 3rd (i.e. index 2) component of the sequence
    class vector to 1.0, and the others to 0.0; note that the classes
    are assumed to start from 0, with 0 being reserved for `background`.

    These values represent the probabilities for any chunk to belong to
    every of the `nb_classes`; for a given sequence chunk, they should
    sum to 1.0. In practice, they are compliant to the SoftMax output
    format, in particular when used as Neural Network output layer.

    If the purpose if to simply create a 2D sequence of classes, prefer
    using the `build_chunks` function from the `data_tk` module instead.

    Parameters
    ----------
    classes : 1D array of ints
        The classes of the signal chunks (frames); typically outputted
        by the `chunk_classes` function. Note that the classes should
        be contiguous and start from 0, with 0 denoting `background`.
        Any class category should be lower than `nb_classes`.
    nb_classes : int
        The number of possible classes; used as second dimension of the
        `sequences` output.
    seq_params : 2-tuple of ints
        The sequences format, organized as (seq_win_size, seq_hop_size).
        Both values should be amounts of samples (integers).

    Returns
    -------
    class_seqs : 3D np.ndarray of floats
        The sequences of probabilities for a chunk to belong to any pos-
        sible class, with 1.0 at the chunk class used as component index,
        and 0.0 anywhere else.

    Examples
    --------
    >>> import numpy as np

    # General parameters
    >>> frate = 40                              # Sampling rate
    >>> frm_duration = 200e-3                   # Frame duration in secs
    >>> nb_classes = 4                          # Number of possible classes

    >>> win_size = int(frm_duration*frate)      # Nb of samples per frame
    >>> hop_size = win_size // 2                # Hop size

    >>> chk_params = (win_size, hop_size)
    >>> seq_params = (10, 3)

    # Generate the signal
    >>> signal = np.arange(0., 10.09, 1./frate) + 0.1

    # Simulate some events
    # (whose stamps are [[752, 1256], [2768, 3325], [5560, 6509]])
    >>> ev_stamps = np.array([[0.94, 1.57], [3.46, 4.15625], [6.95, 8.13625]])

    # Set the event classes (classes start from 1, 0 is the background)
    >>> ev_classes = np.arange(len(ev_stamps)) + 1

    # Determine the classes of the chunks and that of the events
    >>> classes = chunk_classes(
    ...     ev_stamps * frate, ev_classes, len(signal), *chk_params)

    # Build the sequences of classes
    >>> class_seqs = class_sequences(classes, nb_classes, seq_params)

    # Build the sequences of data (optional)
    >>> #data_seqs = data_sequences(signal, chk_params, seq_params)
    """

    # Get the padding offsets for the chunks and for the sequences
    off_seq = pad_offset(len(classes), *seq_params)

    # Pad the classes (add as many `0` (background) as there are sequences)
    classes = pad(classes, off_seq, 0)

    # Get the number of sequences once the class vector is padded
    nb_seqs = get_nb_chunks(len(classes), *seq_params)

    # Build the sequences of classes in a one-hot encoding fashion:
    # in a `nb_classes`-long vector, give a proba of 1.0 to the com-
    # ponent corresponding to the chunk class, and 0.0 to the others
    class_seqs = np.zeros((nb_seqs, seq_params[0], nb_classes), float)
    for i in range(nb_seqs):
        pos = i * seq_params[1]             # Next class vector/chunk
        for j, cls in enumerate(classes[pos:pos+seq_params[0]]):
            class_seqs[i, j, cls] = 1.      # Set a 1.0 proba to the class

    return class_seqs
#----------------------------------------------------------------------------#

##############################################################################
