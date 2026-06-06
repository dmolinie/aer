import numpy as np
from aer.accuracy._comparison import *
from aer.accuracy._metrics import accuracy_chunks, accuracy_sequences


def test_euclidean():
    """ Compute the 1D Euclidean distance btw 2 arrays """
    print(euclidean([[1., 2., 3.]], [[4., 5., 6.]]))

def test_compute_dists_chks():
    """ Compute the Euclidean distance between the accuracy stats vector
    obtained on a dataset (chunks) and the objective accuracy vector """
    # Class arrays shape
    BATCH, NB_CHKS, NB_CLASSES = 5, 10, 4
    # Generate a batch of hits on dummy estimated and reference classes
    hits = []
    for _ in range(BATCH):
        class_est = np.random.random((NB_CHKS, NB_CLASSES))
        class_ref = np.random.random((NB_CHKS, NB_CLASSES))
        hits += accuracy_chunks(class_est, class_ref, False)
    # Compute the distances between the hits arrays
    dists = compute_dists_chks(hits)

def test_compute_dists_seqs():
    """ Compute the Euclidean distance between the accuracy stats vector
    obtained on a dataset (sequences) and the objective accuracy vector """
    BATCH, NB_SEQS, NB_CHKS, NB_CLASSES = 5, 5, 10, 4
    # Generate a batch of hits on dummy estimated and reference classes
    hits = []
    for _ in range(BATCH):
        class_est = np.random.random((NB_SEQS, NB_CHKS, NB_CLASSES))
        class_ref = np.random.random((NB_SEQS, NB_CHKS, NB_CLASSES))
        hits += accuracy_sequences(class_est, class_ref, False)
    # Compute the distances between the hits arrays
    dists = compute_dists_seqs(hits)

def test_find_best():
    """ Find the common value with the lowest rank in two arrays """
    idx_trn = np.array([6, 5, 9, 0, 4, 2, 1, 3, 7, 8])
    idx_gen = np.array([5, 9, 6, 0, 4, 2, 1, 3, 7, 8])
    pos_trn, pos_gen, idx_best = find_best(idx_trn, idx_gen)
    print(pos_trn, pos_gen, idx_best)



# Launch test/example functions
test_euclidean()

test_compute_dists_chks()

test_compute_dists_seqs()

test_find_best()

