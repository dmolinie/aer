""" Functions to compare two sets of accuracy items

Authors: Dylan MOLINIE
Company: CEA Paris-Saclay, Gif-sur-Yvette, France
Contact: dylan.molinie@cea.fr
Date: April 2025
Last revised: April 2026

License: GPLv3
"""

__all__ = [
    'euclidean', 'compute_dists_chks', 'compute_dists_seqs', 'find_best']

import numpy as np


##############################################################################
##                            Compare Hit Scores                            ##
##############################################################################

def euclidean(vec1, vec2):
    """ Compute the 1D Euclidean distance btw 2 arrays """
    return np.sqrt(np.sum((np.mean(vec1, 1) - vec2)**2))

def _compute_ratios(hits):
    """ Compute the stats on hit scores (ratios) """
    return np.array([
        hits['TP'] / hits['1s_ref'], hits['FP'] / hits['1s_ref'],
        hits['FN'] / hits['0s_ref'], hits['TN'] / hits['0s_ref']])

#-------------------   Chunks Ratio-Objective Distance   --------------------#
def compute_dists_chks(hits):
    """ Compute the Euclidean distance between the accuracy stats vector
    obtained on a dataset (chunks) and the objective accuracy vector

    Take a list of hit dicts got on data chunks (cf. `accuracy_chunks`
    function), sum the dicts key by key and compute the ratios of hits
    by computing the number of TPs and FPs over the amount of `1s` and
    the number of FNs and TNs over the amount of `0s`. Finally, return
    the Euclidean distance between this array and the objective one,
    defined as the ratios of `1` for TPs & TNs and of `0` for FPs & FNs.

    N.B.: operate on accuracy items, not on confusion matrices.

    Parameters
    ----------
    hits : list of 1 dict
        The amounts of TPs, FPs, FNs & TNs, as arrays of length the num-
        ber of classes, and the respective numbers of `True` and `False`
        in the two vectors. See `accuracy_items` for details.

    Returns
    -------
    dists : np.ndarray
        The Euclidean distance btw the hit ratios & the reference ones.

    Examples
    --------
    >>> import numpy as np

    # Class arrays shape
    >>> BATCH, NB_CHKS, NB_CLASSES = 5, 10, 4

    # Generate a batch of hits on dummy estimated and reference classes
    >>> hits = []
    >>> for _ in range(BATCH):
    ...     class_est = np.random.random((NB_CHKS, NB_CLASSES))
    ...     class_ref = np.random.random((NB_CHKS, NB_CLASSES))
    ...     hits += accuracy_chunks(class_est, class_ref, False)

    # Compute the distances between the hits arrays
    >>> dists = compute_dists_chks(hits)
    """
    # Sum the dictionaries key by key
    hits_avg = hits[0]
    for hit in hits[1:]:
        for key in hit.keys():
            hits_avg[key] += hit[key]
    # Compute the Euclidean dist btw the ratios & the optimal results
    # (TP=1, FP=0, FN=0, TN=1 --> (1, 0, 0, 1))
    return euclidean(_compute_ratios(hits_avg), np.array([1., 0., 0., 1.]))
#----------------------------------------------------------------------------#

#------------------   Sequences Ratio-Objective Distance   ------------------#
def compute_dists_seqs(hits):
    """ Compute the Euclidean distance between the accuracy stats vector
    obtained on a dataset (sequences) and the objective accuracy vector

    Take a list of hit dicts got on sequences (cf. `accuracy_sequences`
    function), sum the dicts key by key and compute the ratios of hits
    by computing the number of TPs and FPs over the amount of `1s` and
    the number of FNs and TNs over the amount of `0s`. Finally, return
    the Euclidean distance between this array and the objective one,
    defined as the ratios of `1` for TPs & TNs and of `0` for FPs & FNs.
    Note that this is done on both the hits got on chunks and sequences.

    N.B.: operate on accuracy items, not on confusion matrices.

    Parameters
    ----------
    hits : list of 2 dicts
        The amounts of TPs, FPs, FNs & TNs, as arrays of length the num-
        ber of classes, and the respective numbers of `True` and `False`
        in the two vectors. See `accuracy_items` for details.

    Returns
    -------
    dists : np.ndarray
        The Euclidean distance btw the hit ratios & the reference ones.

    Examples
    --------
    >>> import numpy as np

    # Class arrays shape
    >>> BATCH, NB_SEQS, NB_CHKS, NB_CLASSES = 5, 5, 10, 4

    # Generate a batch of hits on dummy estimated and reference classes
    >>> hits = []
    >>> for _ in range(BATCH):
    ...     class_est = np.random.random((NB_SEQS, NB_CHKS, NB_CLASSES))
    ...     class_ref = np.random.random((NB_SEQS, NB_CHKS, NB_CLASSES))
    ...     hits += accuracy_sequences(class_est, class_ref, False)

    # Compute the distances between the hits arrays
    >>> dists = compute_dists_seqs(hits)
    """
    # Sum the dictionaries key by key
    hits_avg_chks = hits[0][0]
    hits_avg_seqs = hits[1][0]
    for hit_chks, hit_seqs in zip(hits[0][1:], hits[1][1:]):
        for key_chks, key_seqs in zip(hit_chks.keys(), hit_seqs.keys()):
            hits_avg_chks[key_chks] += hit_chks[key_chks]
            hits_avg_seqs[key_seqs] += hit_seqs[key_seqs]
    hits_avg = np.vstack(
        (_compute_ratios(hits_avg_chks), _compute_ratios(hits_avg_seqs)))
    # Compute the Euclidean dist btw the ratios & the optimal results
    # (TP=1, FP=0, FN=0, TN=1 --> (1, 0, 0, 1) for chunks & sequences)
    dists_ref = np.array([1., 0., 0., 1., 1., 0., 0., 1.])
    return euclidean(hits_avg, dists_ref)
#----------------------------------------------------------------------------#

#----------------------   Relative Ranks in 2 Arrays   ----------------------#
def find_best(idx_trn, idx_gen):
    """ Find the common value with the lowest rank in two arrays

    Take two arrays of ints, assumed to represent position indices, and
    find the index with the "lowest rank" in both arrays. To this end,
    associate every index to a position "rank", defined as its relative
    position in both arrays. Return these summed relative positions in a
    sorted order (the first item is the index in the corresponding array
    with the lowest "rank") and the sorted list of indices.

    Parameters
    ----------
    idx_trn : np.ndarray
        The first array of integer indices.
    idx_gen : np.ndarray
        The second array of integer indices.

    Returns
    -------
    pos_trn : np.ndarray
        The relative positions of the indices in the `idx_trn` array,
        sorted by ascending "rank" order.
    pos_gen : np.ndarray
        Same as `pos_trn` but for the `idx_gen` input array.
    idx_best : np.ndarray
        The values of the `idx_trn` sorted by rank order.

    Examples
    --------
    # The "best" index is `5` since it the one that appears the
    # earliest in both index arrays

    >>> import numpy as np

    >>> idx_trn = np.array([6, 5, 9, 0, 4, 2, 1, 3, 7, 8])
    >>> idx_gen = np.array([5, 9, 6, 0, 4, 2, 1, 3, 7, 8])

    >>> pos_trn, pos_gen, idx_best = find_best(idx_trn, idx_gen)
    >>> print(pos_trn, pos_gen, idx_best)
    [1 0 2 3 4 5 6 7 8 9] [0 2 1 3 4 5 6 7 8 9] [5 6 9 0 4 2 1 3 7 8]
    """
    pos_trn = np.array(
        [i+np.argwhere(idx == idx_gen)[0, 0] for i, idx in enumerate(idx_trn)])
    pos_gen = np.array(
        [i+np.argwhere(idx == idx_trn)[0, 0] for i, idx in enumerate(idx_gen)])
    pos_trn = pos_trn.argsort()
    pos_gen = pos_gen.argsort()
    return pos_trn, pos_gen, idx_trn[pos_trn]   # Equiv: idx_gen[pos_gen]
#----------------------------------------------------------------------------#

##############################################################################
