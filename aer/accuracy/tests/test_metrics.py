import numpy as np
from aer.accuracy._metrics import *


def test_round_classes():
    """ For a 2D array, set the max of a row to True and the rest to False """
    classes = np.random.random((100, 4))
    hits = round_classes(classes)
    print(hits)

def test_accuracy_items():
    """ Classifications accuracy items on 2D arrays (class by class) """
    class_ref = np.random.random((100, 4))
    class_est = np.random.random((100, 4))
    acc_items = accuracy_items(class_est, class_ref)    # Order matters
    print(acc_items)

def test_confusion_matrix():
    """ Confusion matrix between two sets of classes """
    class_ref = np.random.random((100, 4))
    class_est = np.random.random((100, 4))
    conf_mat = confusion_matrix(class_est, class_ref)   # Order matters
    print(conf_mat)

def test_conf_mat_to_acc_items():
    """ Compute the amounts of TPs, FPs, FNs & TNs from a confusion matrix """
    class_ref = np.random.random((100, 4))
    class_est = np.random.random((100, 4))
    conf_mat = confusion_matrix(class_est, class_ref)   # Order matters
    acc_items = conf_mat_to_acc_items(conf_mat)
    print(acc_items)

def test_accuracy_chunks():
    """ Classification accuracy on data chunks """
    import numpy as np
    # Class arrays shape
    NB_CHUNKS = 10
    NB_CLASSES = 4
    # Generate dummy estimated and reference classes
    class_est = np.random.random((NB_CHUNKS, NB_CLASSES))
    class_ref = np.random.random((NB_CHUNKS, NB_CLASSES))
    # Normalize the classes (sum is 1)
    class_est /= class_est.sum(1, keepdims=True)
    class_ref /= class_ref.sum(1, keepdims=True)
    # Compute the number of correctly classified chunks (hits)
    hits = accuracy_chunks(class_est, class_ref)
    # Compute the hits on sets of classes
    hits = accuracy_chunks([class_est, class_est], [class_ref, class_ref])

def test_accuracy_sequences():
    """ Classification accuracy on sequences of chunks """
    # Class arrays shape
    NB_CHUNKS = 10
    NB_SEQUENCES = 5
    NB_CLASSES = 4
    # Generate dummy estimated and reference classes
    class_est = np.random.random((NB_SEQUENCES, NB_CHUNKS, NB_CLASSES))
    class_ref = np.random.random((NB_SEQUENCES, NB_CHUNKS, NB_CLASSES))
    # Normalize the classes (sum is 1)
    class_est /= class_est.sum(1, keepdims=True)
    class_ref /= class_ref.sum(1, keepdims=True)
    # Compute the number of correctly classified chunks (hits)
    hits = accuracy_sequences(class_est, class_ref)
    # Compute the hits on sets of classes
    hits = accuracy_sequences([class_est, class_est], [class_ref, class_ref])

def test_compute_avgs():
    """ Rates of TPs per category with the mean at the bottom """
    # Class arrays shape
    NB_CHUNKS = 10
    NB_SEQUENCES = 5
    NB_CLASSES = 4
    #--- Hits on chunks
    # Generate dummy estimated and reference classes
    class_est = np.random.random((NB_CHUNKS, NB_CLASSES))
    class_ref = np.random.random((NB_CHUNKS, NB_CLASSES))
    # Normalize the classes (sum is 1)
    class_est /= class_est.sum(-1, keepdims=True)
    class_ref /= class_ref.sum(-1, keepdims=True)
    # Compute the number of correctly classified chunks (hits)
    hits = accuracy_chunks(class_est, class_ref, conf_mat=False)
    avgs = compute_avgs(hits, 'TP', '1s_ref', mean=False)
    print(avgs.shape)
    # Compute them on sets of classes
    hits = accuracy_chunks(
        [class_est, class_est], [class_ref, class_ref],
        conf_mat=False)
    avgs = compute_avgs(hits, 'TP', '1s_ref', mean=True)
    print(avgs.shape)
    #--- Hits on sequences of chunks
    # Generate dummy estimated and reference classes
    class_est = np.random.random((NB_SEQUENCES, NB_CHUNKS, NB_CLASSES))
    class_ref = np.random.random((NB_SEQUENCES, NB_CHUNKS, NB_CLASSES))
    # Normalize the classes (sum is 1)
    class_est /= class_est.sum(-1, keepdims=True)
    class_ref /= class_ref.sum(-1, keepdims=True)
    # Compute the number of correctly classified sequences (hits)
    hits = accuracy_sequences(class_est, class_ref, conf_mat=False)
    avgs = compute_avgs(hits, 'TP', '1s_ref', mean=False)
    print(avgs.shape)
    # Compute them on sets of classes
    hits = accuracy_sequences(
        [class_est, class_est], [class_ref, class_ref],
        conf_mat=False)
    avgs = compute_avgs(hits, 'TP', '1s_ref', mean=True)
    print(avgs.shape)

def test_sensitivity():
    """ Compute the "sensitivity" of a set of classifications """
    # Class arrays shape
    NB_CHUNKS = 10
    NB_CLASSES = 4
    # Generate dummy estimated and reference classes
    class_est = np.random.random((NB_CHUNKS, NB_CLASSES))
    class_ref = np.random.random((NB_CHUNKS, NB_CLASSES))
    # Normalize the classes (sum is 1)
    class_est /= class_est.sum(-1, keepdims=True)
    class_ref /= class_ref.sum(-1, keepdims=True)
    # Compute the number of correctly classified sequences (hits)
    hits = accuracy_chunks(class_est, class_ref, conf_mat=False)
    sens = sensitivity(hits, False)
    print(sens.shape)
    # Compute them on sets of classes
    hits = accuracy_chunks(
        [class_est, class_est], [class_ref, class_ref], conf_mat=False)
    sens = sensitivity(hits, True)
    print(sens.shape)

def test_specificity():
    """ Compute the "specificity" of a set of classifications """
    # Class arrays shape
    NB_CHUNKS = 10
    NB_SEQUENCES = 5
    NB_CLASSES = 4
    # Generate dummy estimated and reference classes
    class_est = np.random.random((NB_SEQUENCES, NB_CHUNKS, NB_CLASSES))
    class_ref = np.random.random((NB_SEQUENCES, NB_CHUNKS, NB_CLASSES))
    # Normalize the classes (sum is 1)
    class_est /= class_est.sum(-1, keepdims=True)
    class_ref /= class_ref.sum(-1, keepdims=True)
    # Compute the number of correctly classified sequences (hits)
    hits = accuracy_sequences(class_est, class_ref, conf_mat=False)
    spec = specificity(hits, False)
    print(spec.shape)
    # Compute them on sets of classes
    hits = accuracy_sequences(
        [class_est, class_est], [class_ref, class_ref], conf_mat=False)
    spec = specificity(hits, True)
    print(spec.shape)



# Launch test/example functions
test_round_classes()

test_accuracy_items()

test_confusion_matrix()

test_conf_mat_to_acc_items()

test_accuracy_chunks()

test_accuracy_sequences()

test_compute_avgs()

test_sensitivity()

test_specificity()

