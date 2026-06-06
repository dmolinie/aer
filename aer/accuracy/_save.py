""" Functions save & load the model accuracy on chunks or sequences

Authors: Dylan MOLINIE
Company: CEA Paris-Saclay, Gif-sur-Yvette, France
Contact: dylan.molinie@cea.fr
Date: April 2025
Last revised: April 2026

License: GPLv3

TP: an event was detected and there was actually one (right detection)
TN: no event was detected and there was actually not (right no-detection)
FP: an event was detected whereas there was actually not (false detection)
FN: no event was detected whereas there was actually one (false no-detection)
"""

__all__ = ['save_hits', 'load_hits']

import csv

import numpy as np

from aer.tools._tools import check_path


##############################################################################
##                             Saving Functions                             ##
##############################################################################

def __flatten(dictionary):
    """ Flatten the values of a dictionary as a single list of values """
    return np.ravel(list(dictionary.values())).tolist()

#-------------------------   Save Chunk Accuracy   --------------------------#
def _save_acc_chks(fname, hits):
    """ Save hits on chunks (confusion matrix or accuracy items) """

    keys = hits.keys() if isinstance(hits, dict) else hits[0].keys()

    # Write the results in the CSV file
    with open(fname, 'w', newline='', encoding='utf-8') as file:
        writer = csv.writer(file)

        # Build the headers
        headers, labels = [], []
        for head in keys:
            headers += [head, "", "", ""]
            labels += ["B", "GB", "GS", "S"]

        # Write the headers
        writer.writerow(headers)
        writer.writerow(labels)

        # Write the statistics saved in `hits`
        if isinstance(hits, dict):
            writer.writerow(__flatten(hits))
        else:
            for hit in hits:
                writer.writerow(__flatten(hit))
#----------------------------------------------------------------------------#

#-----------------------   Save Sequences Accuracy   ------------------------#
def _save_acc_seqs(fname, hits):
    """ Save accuracy on sequences (confusion matrix or accuracy items) """

    if isinstance(hits[0], dict) and isinstance(hits[1], dict):
        keys = hits[0].keys()
    else:
        keys = hits[0][0].keys()

    # Write the results in the CSV file
    with open(fname, 'w', newline='', encoding='utf-8') as file:
        writer = csv.writer(file)

        # Build the headers
        headers, labels = [], []
        for fmt in ('chks', 'seqs'):
            for head in keys:
                headers += [head, fmt, "", ""]
                labels += ["B", "GB", "GS", "S"]

        # Write the headers
        writer.writerow(headers)
        writer.writerow(labels)

        # Write the statistics saved in 'hits'
        if isinstance(hits[0], dict) and isinstance(hits[1], dict):
            writer.writerow(__flatten(hits[0]) + __flatten(hits[1]))
        else:
            for hit in hits:
                writer.writerow(__flatten(hit[0]) + __flatten(hit[1]))
#----------------------------------------------------------------------------#

#------------------------------   Save Hits   -------------------------------#
def save_hits(fname, hits):
    """ Save classification hit scores on chunks or sequences

    Consider the set of hit scores obtained on a dataset, each component
    of `hits` being the scores obtained on a single dataset, with every
    `hit` of `hits` being either a dict returned by the `accuracy_chunks`
    function, or two dicts returned by the `accuracy_sequences` function;
    the hits can either be classification confusion matrices or accuracy
    items (FPs, FPs, etc.). Then, "flatten" all the values of any `hit`,
    and write them in the CSV file entitled `fname`.

    If `fname` contains directories, create them if they do not exist;
    also, append the '.csv' extension to it if it does not end with it.

    Parameters
    ----------
    fname : string
        The (path)name of the CSV file where to save results; should end
        with '.csv'; if not, this extension is automatically appended.
    hits : list of dicts
        Set of confusion matrices (cf. `confusion_matrix` function) or
        of accuracy items (cf. `accuracy_items` function). Should be a
        list of dicts if the hit scores were obtained on chunks, or of
        set of 2 dicts if obtained on sequences. In any case, every dict
        component is an array with length the number of classes.

    Returns
    -------
    None : write the files directly.

    Examples
    --------
    # The aer project `datasets` package contains an example of the MIVIA
    # dataset, composed of two subdatasets of 1 file each: files 66 for
    # "training" and 29 for "testing", both with an SNR of 15 ('3').

    >>> import aer.datasets
    >>> import aer.datasets.mivia_loader as loader

    # Dataset parameters
    >>> ROOT = f"{aer.datasets.__path__[0]}/MIVIA_AE_example/training/"
    >>> XMLFOLDER = ROOT
    >>> SNDFOLDER = ROOT + "sounds/"

    # Dataset index and SNR (only available one in the example training set)
    >>> IND, SNR = 66, 15

    # Chunks parameters
    >>> FRM_DURATION = 50e-3    # Frame duration in seconds
    >>> CHK_PARAMS = (FRM_DURATION, FRM_DURATION)

    # Number of possible classes
    >>> NB_CLASSES = 4

    # Load the data and build the chunks
    >>> specs, data_chks, class_chks = loader.build_dataset(
    ...     (XMLFOLDER, SNDFOLDER), (IND, SNR), NB_CLASSES, CHK_PARAMS)

    # Generate a dummy torch model
    >>> import torch
    >>> model = torch.nn.Sequential(
    ...     torch.nn.Linear(data_chks.shape[-1], NB_CLASSES),
    ...     torch.nn.Softmax(1))

    # Test the model on some files
    >>> conf_mat = True         # Confusion matrices, not accuracy items
    >>> class_est = model.forward(torch.Tensor(data_chks)).detach().numpy()
    >>> hits = accuracy_chunks(class_est, class_chks, conf_mat)

    # Save the results (list of 1 dict)
    >>> save_hits(f"Hits/Testing_{SNR}_chks.csv", hits)

    # Save the results (list of 3 dicts)
    >>> save_hits(f"Hits/Testing_chks_3.csv", hits+hits+hits)
    """
    # If every component of 'hits' is a single dict, assume they are chunks
    if isinstance(hits[0], dict):
        _save_acc_chks(check_path(fname, '.csv'), hits)
    # Else, assume they are sequences
    else:
        _save_acc_seqs(check_path(fname, '.csv'), hits)
#----------------------------------------------------------------------------#

#------------------------------   Load Hits   -------------------------------#
def load_hits(fname, sequences=False, offset=4):
    """ Read and load classification hit scores from a CSV file

    Read the hit scores contained in the filename specified as argument
    and format them into a list of dictionaries; this list contains as
    many dicts as there are rows in the file, minus two rows of headers.
    If `sequences` is True, build two dicts, the first for the chunks &
    the second one for the sequences. The hits can either be confusion
    matrices or accuracy items (TPs, FPs, etc.).

    If `fname` contains directories, create them if they do not exist;
    also, append the '.csv' extension to it if it does not end with it.

    Parameters
    ----------
    fname : string
        The name of the CSV file where to save results; should end with
        '.csv'; if not, this extension is automatically appended.
    [OPT] sequences : bool
        If the file lists the hits obtained on chunks or sequences (as
        two dicts are built when dealing with sequences).
            :Default: False
    [OPT] offset : int
        The offset when reading the hits; typically the number of classes.
            :Default: 4

    Returns
    -------
    classes : list of strings
        The classes listed in the file (second row of it).
    hits : list of dicts
        If `sequences` is False, the hit scores obtained on chunks, one
        dict per dataset (row); if it is True, the scores obtained on
        both chunks & sequences, as two dicts wrapped into a list.
        See the `save_hits` function for the format of the dicts.

    Examples
    --------
    # The aer project `datasets` package contains an example of the MIVIA
    # dataset, composed of two subdatasets of 1 file each: files 66 for
    # "training" and 29 for "testing", both with an SNR of 15 ('3').

    >>> import aer.datasets
    >>> import aer.datasets.mivia_loader as loader

    # Dataset parameters
    >>> ROOT = f"{aer.datasets.__path__[0]}/MIVIA_AE_example/training/"
    >>> XMLFOLDER = ROOT
    >>> SNDFOLDER = ROOT + "sounds/"

    # Dataset index and SNR (only available one in the example training set)
    >>> IND, SNR = 66, 15

    # Chunks parameters
    >>> FRM_DURATION = 50e-3    # Frame duration in seconds
    >>> CHK_PARAMS = (FRM_DURATION, FRM_DURATION)
    >>> SEQ_PARAMS = (10, 1)

    # Number of possible classes
    >>> NB_CLASSES = 4

    # Load the data and build the chunks
    >>> specs, data_chks, class_chks = loader.build_dataset(
    ...     (XMLFOLDER, SNDFOLDER), (IND, SNR), NB_CLASSES, CHK_PARAMS, SEQ_PARAMS)

    # Generate a dummy torch model
    >>> import torch
    >>> model = torch.nn.Sequential(
    ...     torch.nn.Linear(data_chks.shape[-1], NB_CLASSES),
    ...     torch.nn.Softmax(1))

    # Test the model on some files
    >>> conf_mat = True         # Confusion matrices, not accuracy items
    >>> class_est = model.forward(torch.Tensor(data_chks)).detach().numpy()
    >>> hits = accuracy_sequences(class_est, class_chks, conf_mat)

    # Save the results (list of 1 tuple of 2 dicts)
    >>> save_hits(f"Hits/Testing_seqs.csv", hits)
    # Load the classes & hits
    >>> classes, hits = load_hits(f"Hits/Testing_seqs.csv", True, NB_CLASSES)

    # Save the results (list of 3 tuples of 2 dicts)
    >>> save_hits(f"Hits/Testing_seqs_3.csv", hits+hits+hits)
    # Load the classes & hits
    >>> classes, hits = load_hits(f"Hits/Testing_seqs_3.csv", True, NB_CLASSES)
    """

    # Read & load the data from the CSV file
    with open(check_path(fname, '.csv'), 'r', encoding='utf-8') as file:
        reader = csv.reader(file, delimiter=',')

        # Read the headers
        headers = next(reader)              # Type of hits
        nb_cols = len(headers)              # Number of columns
        headers = list(headers[::offset])   # Extract the headers
        classes = next(reader)[:offset]     # Event classes

        # Read & load the data
        hits = []
        if not sequences:
            # Read & load every row and build the dict of hit scores
            for row in reader:
                hits.append(
                    {head: np.array(row[pos:pos+offset], int)
                     for pos, head in zip(range(0, nb_cols, offset), headers)})
        else:
            # Get the number of columns (and ignore the headers as well)
            cols = nb_cols // 2             # Border btw chunks & sequences
            head = [headers[0:cols//offset], headers[cols//offset:nb_cols//offset]]
            # Read every row and build the dicts of hits on chunks & sequences
            for row in reader:
                hits.append(
                    [{head: np.array(row[pos:pos+offset], int)
                      for pos, head in zip(range(0, cols, offset), head[0])},
                     {head: np.array(row[pos:pos+offset], int)
                      for pos, head in zip(range(cols, nb_cols, offset), head[1])}])

    # Return the event classes and the dictionaries of hit scores
    return classes, hits
#----------------------------------------------------------------------------#

##############################################################################
