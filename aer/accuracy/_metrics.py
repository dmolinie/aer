""" Functions to measure a model accuracy on chunks or sequences

Authors: Dylan MOLINIE
Company: CEA Paris-Saclay, Gif-sur-Yvette, France
Contact: dylan.molinie@cea.fr
Date: April 2025
Last revised: April 2026

License: GPLv3

TP: an event was detected and there was actually one (right detection)
TN: no event was detected and there was actually not (right non-detection)
FP: an event was detected whereas there was actually not (false detection)
FN: no event was detected whereas there was actually one (false non-detection)
"""

__all__ = [
    'round_classes', 'accuracy_items', 'accuracy_chunks',
    'accuracy_sequences', 'confusion_matrix', 'conf_mat_to_acc_items',
    'compute_avgs', 'sensitivity', 'specificity']

import numpy as np


##############################################################################
##                         Chunks & Sequences Hits                          ##
##############################################################################

#-------------------------   Round Classes Array   --------------------------#
def round_classes(classes):
    """ For a 2D array, set the max of a row to True and the rest to False

    Considering an MxM array, with M the amount of data chunks and M the
    number of possible classes to which any chunk can belong to, for any
    of its rows, find its maximum value and set the corresponding cell
    to `True`, and set all the others to `False`. If the same max value
    appears several times in the same row, consider its is a noise and
    set to True the first cell, and False all the others.

    Parameters
    ----------
    classes : 2D np.ndarray
        The array of the classes to round, whose rows are typically the
        estimates in response to data chunks or sequences.

    Returns
    -------
    hits : 2D np.ndarray of Booleans
        The rounded classes, where the row-wise maximum value are set to
        True, and the others to False. There is only one `True` per row;
        if the same maximum value appears several times in a row, only
        the first component is set to True, and any others to False.

    Examples
    --------
    >>> import numpy as np

    >>> classes = np.random.random((100, 4))
    >>> hits = round_classes(classes)
    """
    hits = np.where(classes == np.max(classes, -1).reshape(-1, 1), True, False)
    # If a row has several hits (i.e. the same max value appears several times),
    # consider that this chunk is noisy, and thus set the first value to True,
    # and the others to False; this case should be rare with floating values
    noise_as_default = np.full_like(hits[0], False)     # [F, F, F, ..., F]
    noise_as_default[0] = True                          # [T, F, F, ..., F]
    hits[hits.sum(1) > 1] = noise_as_default            # Replace noisy rows
    return hits
#----------------------------------------------------------------------------#

#------------------------   Accuracy on 2D Arrays   -------------------------#
def accuracy_items(class_est, class_ref):
    """ Classifications accuracy items on 2D arrays (class by class)

    Take two sets of 2D arrays, representing the estimated and reference
    classes for instance, and count the number of:
      - True Positives (TP): an event is detected when there is an event
      - False Positives (FP): an event is detected when there is no event
      - False Negatives (FN): no event is detected when there is an event
      - True Negatives (TN): no event is detected when there is no event

    Wrap these scores inside a dictionary and add the number of `True`
    (event) and `False` (no event) in the estimate and reference arrays.

    Parameters
    ----------
    class_est : 2D np.ndarray
        The array of the estimated classes.
    class_ref : 2D np.ndarray
        The array of the reference classes.
    N.B.: the order matters; if reversed, FPs and FNs will be reversed.

    Returns
    -------
    items : dict
        The amounts of TPs, FPs, FNs & TNs, as arrays of length the num-
        ber of classes, and the respective numbers of `True` and `False`
        in the two vectors. The keys are:
          - 'TP', 'FP', 'FN' & 'TN' for the scores
          - '0s_est' & '1s_est' for the nb of 0s and 1s in `class_est`
          - '0s_ref' & '1s_ref' for the nb of 0s and 1s in `class_ref`
        The last items can be used to compute statistics on the scores.

    Examples
    --------
    >>> import numpy as np

    >>> class_ref = np.random.random((100, 4))
    >>> class_est = np.random.random((100, 4))

    >>> acc_items = accuracy_items(class_est, class_ref)    # Order matters
    >>> print(acc_items)
    {'TP': array([5, 8, 7, 8]),
     'FP': array([15, 17, 21, 19]),
     'FN': array([20, 12, 21, 19]),
     'TN': array([60, 63, 51, 54]),
     '1s_ref': array([25, 20, 28, 27]),
     '0s_ref': array([75, 80, 72, 73]),
     '1s_est': array([20, 25, 28, 27]),
     '0s_est': array([80, 75, 72, 73])}
    """

    # For each row (chunk), replace the max value by True & the others by False
    cls_est = round_classes(class_est)
    cls_ref = round_classes(class_ref)

    # Compute the amounts of TP, FP, FN and TN:
    #      TN  FN  FP  TP
    # Est  0   0   1   1
    # Ref  0   1   0   1
    items = {}
    items['TP'] = np.sum(np.logical_and(cls_est, cls_ref), 0)
    items['FP'] = np.sum(np.logical_and(cls_est, np.logical_not(cls_ref)), 0)
    items['FN'] = np.sum(np.logical_and(np.logical_not(cls_est), cls_ref), 0)
    items['TN'] = len(cls_ref) - items['TP'] - items['FP'] - items['FN']
#    items['TN'] = np.sum(np.logical_not(np.logical_or(cls_est, cls_ref)), 0)

    # Number of events in the reference classes
    items['1s_ref'] = np.sum(cls_ref, 0)
    items['0s_ref'] = len(cls_ref) - items['1s_ref']
#    items['0s_ref'] = np.sum(np.logical_not(cls_ref), 0)

    # Number of events in the estimated classes
    items['1s_est'] = np.sum(cls_est, 0)
    items['0s_est'] = len(cls_est) - items['1s_est']
#    items['0s_est'] = np.sum(np.logical_not(cls_est), 0)

    return items
#----------------------------------------------------------------------------#

#--------------------------   Confusion Matrix   ----------------------------#
def confusion_matrix(class_est, class_ref):
    """ Confusion matrix between two sets of classes

    Take two sets of 2D arrays, representing the estimated and reference
    classes for instance, and build their confusion matrix: for any class
    in the reference set, compute the number of times it has been labeled
    with every possible class in the estimate set.

    Parameters
    ----------
    class_est : 2D np.ndarray
        The array of the estimated classes.
    class_ref : 2D np.ndarray
        The array of the reference classes.
    N.B.: the order matters; the matrix will be transposed if reversed.

    Returns
    -------
    mat : dict
        The confusion matrix, with the reference classes as rows and the
        estimated ones as columns. The classes are referred to as `C{i}`,
        starting from 0, which also serve as keys to the dictionary.

    Examples
    --------
    >>> import numpy as np

    >>> class_ref = np.random.random((100, 4))
    >>> class_est = np.random.random((100, 4))

    >>> conf_mat = confusion_matrix(class_est, class_ref)   # Order matters
    >>> print(conf_mat)
    {'C0': array([ 2,  6,  8, 11]),
     'C1': array([1, 5, 6, 6]),
     'C2': array([7, 8, 5, 7]),
     'C3': array([4, 9, 9, 6])}
    """

    # For each row (chunk), replace the max value by True & the others by False
    cls_est = round_classes(class_est)
    cls_ref = round_classes(class_ref)

    # Compute confusion matrix (reference classes as rows, estimated as cols)
    mat = {}
    for i, cls in enumerate(cls_ref.T):
        mat[f'C{i}'] = np.sum(cls_est[np.argwhere(cls).ravel()], 0)

    return mat
#----------------------------------------------------------------------------#

#------------------   Confusion Matrix to Accuracy Items   ------------------#
def _conf_mat_to_acc_items(matrix, zeros_ones):
    """ Compute the amounts of TPs, FPs, FNs & TNs from a confusion matrix """
    # Wrap the values of the dict into a NumPy array
    if isinstance(matrix, dict):
        matrix = np.array(list(matrix.values()))
    # Compute the sums only once
    sum_rows = np.sum(matrix, 0)
    sum_cols = np.sum(matrix, 1)
    sum_full = np.sum(matrix)
    # Compute the number of TPs, FPs, FNs and TNs
    items = {}
    items['TP'] = np.diag(matrix)
    items['FP'] = sum_rows - items['TP']
    items['FN'] = sum_cols - items['TP']
    items['TN'] = sum_full - items['TP'] - items['FP'] - items['FN']
    # Add the number of 0s and 1s in the reference and estimated classes
    if zeros_ones:
        items['1s_ref'] = sum_cols
        items['0s_ref'] = sum_full - items['1s_ref']
        items['1s_est'] = sum_rows
        items['0s_est'] = sum_full - items['1s_est']
    return items

def conf_mat_to_acc_items(matrix, zeros_ones=False):
    """ Compute the amounts of TPs, FPs, FNs & TNs from a confusion matrix

    Take a confusion matrix for a set of classes and retrieve the amounts
    of True Positives (TPs), False Positives (FPs), False Negatives (FNs)
    and True Negatives (TNs), and wrap them into a dict. If `zeros_ones`
    is True, add also the amounts of `0s` and `1s` in both the reference
    (rows) and estimate (cols) sets that may be used for statistics.

    If `matrix` is a list of confusion matrices, do this for any of them
    and wrap the returned dictionaries of items into a list.

    Parameters
    ----------
    matrix : (list of) dict(s)
        The confusion matrix of the classes. Can be a list of arrays.
    [OPT] zeros_ones : bool
        If the number of 0s and 1s in both the reference and estimated
        classes should be listed (cf. `accuracy_items` function).
            :Default: False

    Returns
    -------
    items : (list of) dict(s) of np.ndarrays
        The amounts of TPs, FPs, FNs and TNs; same shape type and shape
        as `matrix` (direct dict or list of dicts).

    Examples
    --------
    >>> import numpy as np

    >>> class_ref = np.random.random((100, 4))
    >>> class_est = np.random.random((100, 4))

    # Build the confusion matrix
    >>> conf_mat = confusion_matrix(class_est, class_ref)   # Order matters

    # Single confusion matrix
    >>> print(conf_mat_to_acc_items(conf_mat))
    {'TP': array([8, 3, 5, 9]),
     'FP': array([14, 16, 23, 22]),
     'FN': array([19, 18, 21, 17]),
     'TN': array([59, 63, 51, 52])}

    # List of confusion matrices
    >>> print(conf_mat_to_acc_items([conf_mat, conf_mat]))
    [{'TP': array([8, 3, 5, 9]),
      'FP': array([14, 16, 23, 22]),
      'FN': array([19, 18, 21, 17]),
      'TN': array([59, 63, 51, 52])},
     {'TP': array([8, 3, 5, 9]),
      'FP': array([14, 16, 23, 22]),
      'FN': array([19, 18, 21, 17]),
      'TN': array([59, 63, 51, 52])}]
    """
    # If confusion matrices obtained on chunks
    if isinstance(matrix, dict):
        return _conf_mat_to_acc_items(matrix, zeros_ones)
    # Or on sequences of chunks (i.e. 2 dicts)
    return [_conf_mat_to_acc_items(mat, zeros_ones) for mat in matrix]
#----------------------------------------------------------------------------#

##############################################################################



##############################################################################
##                       Chunks & Sequences Accuracy                        ##
##############################################################################

#------------------------   Get Accuracy on Chunks   ------------------------#
def accuracy_chunks(class_est, class_ref, conf_mat=True):
    """ Classification accuracy on data chunks

    Considering a set of `nb_chunks` chunks, take the estimated classes
    (e.g. provided by a Torch model) and the reference (objective) ones,
    compare both sets chunk by chunk and count their matches (`hits`).

    If `conf_mat` is True, build the confusion matrix between the sets;
    cf. `confusion_matrix` function. Else, compute the accuracy items:
    True Positives (TPs), False Positives (FPs), True Negatives (TNs)
    and False Negatives (FNs), + the number of `False` and `True` for
    each class for both input arrays; cf. `accuracy_items` function.

    The `class_est` & `class_ref` inputs can either be direct arrays or
    lists of arrays; in both cases, the dictionaries of hits are wrapped
    into a list for consistency with the other accuracy-related functions.

    Parameters
    ----------
    class_est : 1D or 2D array_like
        The estimated classes, typically outputted by a Torch model.
    class_ref : 1D or 2D array_like
        The reference classes, which serve as objective during training.
    [OPT] conf_mat : bool
        If True, build the confusion matrix, or the accuracy items else.
            :Default: True (build the confusion matrices)
    N.B.: the order between `class_est` and `class_ref` matters.

    Returns
    -------
    hits : list of dict(s)
        The list of confusion matrices if `conf_mat` is True, or of the
        amounts of TPs, FPs, FNs and TNs and of `True` and False` in the
        two input arrays else. Always a list, even when there is only one
        dictionary issued. Each dictionary item is an array with length
        the number of classes.

    Examples
    --------
    >>> import numpy as np

    # Class arrays shape
    >>> NB_CHUNKS = 10
    >>> NB_CLASSES = 4

    # Generate dummy estimated and reference classes
    >>> class_est = np.random.random((NB_CHUNKS, NB_CLASSES))
    >>> class_ref = np.random.random((NB_CHUNKS, NB_CLASSES))

    # Normalize the classes (sum is 1)
    >>> class_est /= class_est.sum(1, keepdims=True)
    >>> class_ref /= class_ref.sum(1, keepdims=True)

    # Compute the number of correctly classified chunks (hits)
    >>> hits = accuracy_chunks(class_est, class_ref)

    # Compute the hits on sets of classes
    >>> hits = accuracy_chunks(
    ...     [class_est, class_est], [class_ref, class_ref])
    """

    # Choose between confusion matrix & accuracy items
    func = confusion_matrix if conf_mat else accuracy_items

    # If single arrays, return a list of 1 dict
    if isinstance(class_est, np.ndarray) and isinstance(class_ref, np.ndarray):
        return [func(class_est, class_ref)]

    # If lists of arrays, return a list of N dicts
    return [func(cls_est, cls_ref)
            for cls_est, cls_ref in zip(class_est, class_ref)]
#----------------------------------------------------------------------------#

#----------------------   Get accuracy on Sequences   -----------------------#
def _class_seqs(args, nb_classes):
    """ Number of times that any integers between 0 and `nb_classes-1`
    appear in any rows of `args` """
    classes = np.empty((len(args), nb_classes), int)
    # Get the number of occurrences of each class in a sequence
    for i in range(nb_classes):
        classes[:, i] = np.sum(args == i, 1)
    return classes

def _acc_seqs(class_est, class_ref, func):
    """ Classification accuracy on sequences """
    # Get the class of any chunks in any sequences (2D array)
    args_est = np.argmax(class_est, -1)
    args_ref = np.argmax(class_ref, -1)
    # For any sequence, associate a vector of length the amount of classes,
    # and filled with the number of chunks of these classes in the sequence
    class_seqs_est = _class_seqs(args_est, class_est.shape[-1])
    class_seqs_ref = _class_seqs(args_ref, class_ref.shape[-1])
    # Flatten the second axis (the sequences) to stack the chunks
    class_est = class_est.reshape(-1, class_ref.shape[-1])
    class_ref = class_ref.reshape(-1, class_ref.shape[-1])
    # Statistics on the correctly classified chunks & sequences
    hits_chks = func(class_est, class_ref)
    hits_seqs = func(class_seqs_est, class_seqs_ref)
    return hits_chks, hits_seqs

def accuracy_sequences(class_est, class_ref, conf_mat=True):
    """ Classification accuracy on sequences of chunks

    Considering a set of `nb_sequences` sequences of `nb_chunks` chunks,
    take the estimated classes of the sequence chunks and the reference
    ones, and compute the `hits` at the chunk and sequence levels. If
    `conf_mat` is True, build the confusion matrix between the two sets;
    cf. `confusion_matrix` function. Else, compute the accuracy items:
    TPs, FPs, TNs and FNs, + the number of `False` and `True` for each
    class for both input arrays; cf. `accuracy_items` function.

    At the chunk level, all the chunks of any sequences are flattened,
    removing the last dimension (NxMxK 3D array --> (NxM)xK 2D array);
    then, these chunks are passed to the `accuracy_chunks` function.

    At the sequence level, the K classes of any chunks are replaced by
    single ints representing their classes (argmax), issuing an array
    of one less dimension. For any row of this array, the number of 0s,
    1s, 2s, etc. (any value between 0 and K) is enumerated, resulting
    in a new array which is passed to the `accuracy_chunks` function
    (NxMxK array --> NxM array --> NxK array).

    The `class_est` & `class_ref` inputs can either be direct arrays or
    lists of arrays; in both cases, the dictionaries of hits are wrapped
    into a list for consistency with the other accuracy-related functions.

    Parameters
    ----------
    class_est : 2D or 3D array_like
        The estimated classes, typically outputted by a Torch model.
    class_ref : 2D or 3D array_like
        The reference classes, which serve as objective during training.
    [OPT] conf_mat : bool
        If True, build the confusion matrix, or the accuracy items else.
            :Default: True (build the confusion matrices)
    N.B.: the order between `class_est` and `class_ref` matters.

    Returns
    -------
    hits_chks : list of dict(s)
        At the chunk level, the list of confusion matrices if `conf_mat`
        is True, or of the amounts of TPs, FPs, FNs and TNs and of `True`
        and False` in the two input arrays else. Always a list, even when
        there is only one dictionary issued. Each dictionary component is
        an array with length the number of classes.
    hits_seqs : list of dict(s)
        Same as `hits_chks` but at the sequence level.

    Examples
    --------
    >>> import numpy as np

    # Class arrays shape
    >>> NB_CHUNKS = 10
    >>> NB_SEQUENCES = 5
    >>> NB_CLASSES = 4

    # Generate dummy estimated and reference classes
    >>> class_est = np.random.random((NB_SEQUENCES, NB_CHUNKS, NB_CLASSES))
    >>> class_ref = np.random.random((NB_SEQUENCES, NB_CHUNKS, NB_CLASSES))

    # Normalize the classes (sum is 1)
    >>> class_est /= class_est.sum(1, keepdims=True)
    >>> class_ref /= class_ref.sum(1, keepdims=True)

    # Compute the number of correctly classified chunks (hits)
    >>> hits = accuracy_sequences(class_est, class_ref)

    # Compute the hits on sets of classes
    >>> hits = accuracy_sequences(
    ...     [class_est, class_est], [class_ref, class_ref])
    """

    # Choose between confusion matrix & accuracy items
    func = confusion_matrix if conf_mat else accuracy_items

    # If single arrays, return a list of 1 dict
    if isinstance(class_est, np.ndarray) and isinstance(class_ref, np.ndarray):
        return [_acc_seqs(class_est, class_ref, func)]

    # If lists of arrays, return a list of dicts
    return [_acc_seqs(cls_est, cls_ref, func)
            for cls_est, cls_ref in zip(class_est, class_ref)]
#----------------------------------------------------------------------------#

#-----------------------   Compute & Clean Averages   -----------------------#
def compute_avgs(hits, key_stat='TP', key_lgts='1s_ref', mean=True):
    """ Rates of an indicator per category

    Take a list of dicts of `hits` (cf. the `load_hits` function) and,
    for every dict `hit` in `hits`, compute the percentage of correct
    classifications defined as the ratio between `hit[key_stat]` and
    `hit[key_lgts]` multiplied by 100; wrap all these values (one per
    dict) into an np.ndarray. If `mean`, add an additional column and
    row (both at the vector end) that contain the mean percentages of
    correct classifications along the rows and columns, respectively.

    In the case of an event was originally missing in the dataset, its
    number of occurrences should be 0, issuing a total of 0 reference
    hits for this class, leading to a division by 0, represented by an
    NaN value. To avoid this, replace all the `NaNs` obtained during
    computation by the sentinel value `-1`, indicating that the event
    was originally missing in the reference data. Note that a warning
    may be raised by NumPy but it can safely be ignored here.

    N.B.: operate on accuracy items, not on confusion matrices.

    Parameters
    ----------
    hits : list of dict(s)
        The amounts of TPs, FPs, FNs & TNs, as arrays of length the num-
        ber of classes, and the respective numbers of `True` and `False`
        in the two vectors. See `accuracy_items` for details. Should be
        a list of 1 dict if the hit scores were obtained on chunks, or
        a set of 2 dicts if obtained on sequences.
    [OPT] key_stat : string
        The dict key to access the values used as numerators.
            :Default: 'TP' (number of True Positives)
    [OPT] key_lgts : string
        The dict key to access the values used as denominators.
            :Default: '1s_ref' (class occurrences in the ref data)
    [OPT] mean : bool
        If the mean of the rows should be added at the array's bottom
        (last row). If so, the length the returns is increased by 1.
            :Default: True

    Returns
    -------
    avgs : array of floats
        The array of percentages. With N the number of dicts (length of
        `hits`) and K the number of classes (length of any item of any
        dict in `hits`), `avgs` has shape NxK if the hits were obtained
        on chunks, or Nx2xK on sequences. The percentages are computed
        wrt the `hits` structure: the i-th row of `avgs` corresponds to
        the i-th dict (or couple of dicts) of `hits`. If `mean` is True,
        the mean of the arrays are appended as a new column, increasing
        the value of K by 1, and the mean of all rows is appended as a
        new row at the end of the array, increasing the value of N by 1.

    Examples
    --------
    >>> import numpy as np

    # Class arrays shape
    >>> NB_CHUNKS = 10
    >>> NB_SEQUENCES = 5
    >>> NB_CLASSES = 4

    #--- Hits on chunks
    # Generate dummy estimated and reference classes
    >>> class_est = np.random.random((NB_CHUNKS, NB_CLASSES))
    >>> class_ref = np.random.random((NB_CHUNKS, NB_CLASSES))

    # Normalize the classes (sum is 1)
    >>> class_est /= class_est.sum(-1, keepdims=True)
    >>> class_ref /= class_ref.sum(-1, keepdims=True)

    # Compute the number of correctly classified chunks (hits)
    >>> hits = accuracy_chunks(class_est, class_ref, conf_mat=False)
    >>> avgs = compute_avgs(hits, 'TP', '1s_ref', mean=False)
    >>> print(avgs.shape)
    (1, 4)

    # Compute them on sets of classes
    >>> hits = accuracy_chunks(
    ...     [class_est, class_est], [class_ref, class_ref],
    ...     conf_mat=False)
    >>> avgs = compute_avgs(hits, 'TP', '1s_ref', mean=True)
    >>> print(avgs.shape)
    (3, 5)

    #--- Hits on sequences of chunks
    # Generate dummy estimated and reference classes
    >>> class_est = np.random.random((NB_SEQUENCES, NB_CHUNKS, NB_CLASSES))
    >>> class_ref = np.random.random((NB_SEQUENCES, NB_CHUNKS, NB_CLASSES))

    # Normalize the classes (sum is 1)
    >>> class_est /= class_est.sum(-1, keepdims=True)
    >>> class_ref /= class_ref.sum(-1, keepdims=True)

    # Compute the number of correctly classified sequences (hits)
    >>> hits = accuracy_sequences(class_est, class_ref, conf_mat=False)
    >>> avgs = compute_avgs(hits, 'TP', '1s_ref', mean=False)
    >>> print(avgs.shape)
    (1, 2, 4)

    # Compute them on sets of classes
    >>> hits = accuracy_sequences(
    ...     [class_est, class_est], [class_ref, class_ref],
    ...     conf_mat=False)
    >>> avgs = compute_avgs(hits, 'TP', '1s_ref', mean=True)
    >>> print(avgs.shape)
    (3, 2, 5)
    """

    # If 'chunks', retrieve the 'key_stat' and number of events for each dataset
    if isinstance(hits[0], dict):
        tpos = np.array([hit[key_stat] for hit in hits])
        lgts = np.array([hit[key_lgts] for hit in hits])
    # Else, flatten those of the chunks & of the sequences
    else:
        tpos = np.array([[hit[0][key_stat], hit[1][key_stat]] for hit in hits])
        lgts = np.array([[hit[0][key_lgts], hit[1][key_lgts]] for hit in hits])

    if mean:
        # Append the sums of the columns on the last columns (classes mean)
        tpos = np.concatenate((tpos, tpos.sum(-1, keepdims=True)), axis=-1)
        lgts = np.concatenate((lgts, lgts.sum(-1, keepdims=True)), axis=-1)
        # Append the sums of the rows on the last rows (datasets mean)
        tpos = np.concatenate((tpos, tpos.sum(0, keepdims=True)), axis=0)
        lgts = np.concatenate((lgts, lgts.sum(0, keepdims=True)), axis=0)

    # Compute the percentages
    avgs = (100. * tpos / lgts).round(3)
    # When no event type is not present in the dataset, replace the NaN by -1
    avgs = np.nan_to_num(avgs, nan=-1)

    return avgs
#----------------------------------------------------------------------------#

#----------------------   Classification Sensitivity   ----------------------#
def sensitivity(hits, mean=False):
    """ Compute the "sensitivity" of a set of classifications

    Consider a classification task evaluated on a set of datasets using
    the amount of True/False Positives/Negatives (TPs, FPs, FNs & TNs)
    as accuracy indicators, wrapped into dictionaries, one per dataset
    (cf. `accuracy_chunks` & `accuracy_sequences` functions). For each
    dictionary/dataset, compute the classification "sensitivity" (also
    called True Positive rate), which is defined as:
        sens = TPs / (TPs + FNs)

    This measure gives the probability of the (correct) detection of an
    event given that such an event occurs. The closer to 1, the better.

    The above-mentioned indicators are supposed to be arrays of K col-
    umns, with K the number of classes; for every dictionary in `hits`,
    the sensitivity is computed class by class. If a class was missing
    in the original dataset, this may lead to a division by 0., issuing
    an NaN value. Replace such values by the `-1` sentinel value if so.
    Also, if `mean` is True, compute the mean of the sensitivities of
    all classes (but ignore the NaN values).

    N.B.: this function accepts both classification results obtained on
        chunks and on sequences; in the first case, any items of `hits`
        should be single dicts; else, they should be tuples of 2 dicts.

    Parameters
    ----------
    hits : list of dict(s)
        The amounts of TPs, FPs, FNs & TNs, as arrays of length the num-
        ber of classes, and the respective numbers of `True` and `False`
        in the two vectors. See `accuracy_items` for details. Should be
        a list of 1 dict if the hit scores were obtained on chunks, or
        a set of 2 dicts if obtained on sequences.
    [OPT] mean : bool
        If the class-by-class sensitivities should be averaged or not.
            :Default: False

    Returns
    -------
    sens : array of floats
        The sensitivities, one row per item in `hits`. With N the number
        of dicts in `hits` & K the number of classes (length of any item
        of any dict in `hits`), `sens` has shape NxK if the `hits` were
        obtained on chunks, or Nx2xK if on sequences. If `mean` is True,
        the last dimension of the array is averaged, resulting in the
        flattening of `sens` (thus it has either shape N or Nx2).

    Examples
    --------
    # See the `Examples` of `specificity` for an example with sequences

    >>> import numpy as np

    # Class arrays shape
    >>> NB_CHUNKS = 10
    >>> NB_CLASSES = 4

    # Generate dummy estimated and reference classes
    >>> class_est = np.random.random((NB_CHUNKS, NB_CLASSES))
    >>> class_ref = np.random.random((NB_CHUNKS, NB_CLASSES))

    # Normalize the classes (sum is 1)
    >>> class_est /= class_est.sum(-1, keepdims=True)
    >>> class_ref /= class_ref.sum(-1, keepdims=True)

    # Compute the number of correctly classified sequences (hits)
    >>> hits = accuracy_chunks(class_est, class_ref, conf_mat=False)
    >>> sens = sensitivity(hits, False)
    >>> print(sens.shape)
    (1, 4)

    # Compute them on sets of classes
    >>> hits = accuracy_chunks(
    ...     [class_est, class_est], [class_ref, class_ref], conf_mat=False)
    >>> sens = sensitivity(hits, True)
    >>> print(sens.shape)
    (2,)
    """

    # If 'chunks', compute the sensitivity on its dictionaries
    if isinstance(hits[0], dict):
        sens = np.array([hit['TP'] / (hit['TP'] + hit['FN']) for hit in hits])
    # Else, compute it for each of the two dicts (chunks & sequences)
    else:
        sens = np.array(
            [(hit[0]['TP'] / (hit[0]['TP'] + hit[0]['FN']),
              hit[1]['TP'] / (hit[1]['TP'] + hit[1]['FN'])) for hit in hits])

    if mean:
        # Compute the mean (ignore NaN values)
        sens = np.nanmean(sens, -1)
    else:
        # Indicate absence of occurrences in a class (NaN) by -1
        sens = np.nan_to_num(sens, nan=-1.)

    return sens
#----------------------------------------------------------------------------#

#----------------------   Classification Specificity   ----------------------#
def specificity(hits, mean=False):
    """ Compute the "specificity" of a set of classifications

    Consider a classification task evaluated on a set of datasets using
    the amount of True/False Positives/Negatives (TPs, FPs, FNs & TNs)
    as accuracy indicators, wrapped into dictionaries, one per dataset
    (cf. `accuracy_chunks` & `accuracy_sequences` functions). For each
    dictionary/dataset, compute the classification "specificity" (also
    called True negative rate), which is defined as:
        spec = TNs / (TNs + FPs)

    This measure gives the probability of the (correct) non-detection
    of an event given that such an event does not occur. The closer
    to 1, the better.

    The above-mentioned indicators are supposed to be arrays of K col-
    umns, with K the number of classes; for every dictionary in `hits`,
    the specificity is computed class by class. If a class was missing
    in the original dataset, this may lead to a division by 0., issuing
    an NaN value. Replace such values by the `-1` sentinel value if so.
    Also, if `mean` is True, compute the mean of the specificities of
    all classes (but ignore the NaN values).

    N.B.: this function accepts both classification results obtained on
        chunks and on sequences; in the first case, any items of `hits`
        should be single dicts; else, they should be tuples of 2 dicts.

    Parameters
    ----------
    hits : list of dict(s)
        The amounts of TPs, FPs, FNs & TNs, as arrays of length the num-
        ber of classes, and the respective numbers of `True` and `False`
        in the two vectors. See `accuracy_items` for details. Should be
        a list of 1 dict if the hit scores were obtained on chunks, or
        a set of 2 dicts if obtained on sequences.
    [OPT] mean : bool
        If the class-by-class specificities should be averaged or not.
            :Default: False

    Returns
    -------
    spec : array of floats
        The specificities, one row per item in `hits`. With N the number
        of dicts in `hits` & K the number of classes (length of any item
        of any dict in `hits`), `spec` has shape NxK if the `hits` were
        obtained on chunks, or Nx2xK if on sequences. If `mean` is True,
        the last dimension of the array is averaged, resulting in the
        flattening of `spec` (thus it has either shape N or Nx2).

    Examples
    --------
    # See the `Examples` of `sensitivity` for an example with chunks

    >>> import numpy as np

    # Class arrays shape
    >>> NB_CHUNKS = 10
    >>> NB_SEQUENCES = 5
    >>> NB_CLASSES = 4

    # Generate dummy estimated and reference classes
    >>> class_est = np.random.random((NB_SEQUENCES, NB_CHUNKS, NB_CLASSES))
    >>> class_ref = np.random.random((NB_SEQUENCES, NB_CHUNKS, NB_CLASSES))

    # Normalize the classes (sum is 1)
    >>> class_est /= class_est.sum(-1, keepdims=True)
    >>> class_ref /= class_ref.sum(-1, keepdims=True)

    # Compute the number of correctly classified sequences (hits)
    >>> hits = accuracy_sequences(class_est, class_ref, conf_mat=False)
    >>> spec = specificity(hits, False)
    >>> print(spec.shape)
    (1, 2, 4)

    # Compute them on sets of classes
    >>> hits = accuracy_sequences(
    ...     [class_est, class_est], [class_ref, class_ref], conf_mat=False)
    >>> spec = specificity(hits, True)
    >>> print(spec.shape)
    (2, 2)
    """

    # If 'chunks', compute the specificity on its dictionaries
    if isinstance(hits[0], dict):
        spec = np.array([hit['TN'] / (hit['TN'] + hit['FP']) for hit in hits])
    # Else, compute it for each of the two dicts (chunks & sequences)
    else:
        spec = np.array(
            [(hit[0]['TN'] / (hit[0]['TN'] + hit[0]['FP']),
              hit[1]['TN'] / (hit[1]['TN'] + hit[1]['FP'])) for hit in hits])

    if mean:
        # Compute the mean (ignore NaN values)
        spec = np.nanmean(spec, -1)
    else:
        # Indicate absence of occurrences in a class (NaN) by -1
        spec = np.nan_to_num(spec, nan=-1.)

    return spec
#----------------------------------------------------------------------------#

##############################################################################
