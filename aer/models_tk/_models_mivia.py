""" Toolkit to train & test Torch models on the MIVIA Audio Events Dataset

Authors: Dylan MOLINIE
Company: CEA Paris-Saclay, Gif-sur-Yvette, France
Contact: dylan.molinie@cea.fr
Date: June 2025
Last revised: April 2026

License: GPLv3
"""
# pylint: disable=too-many-arguments, too-many-positional-arguments

__all__ = ['DEVICE',
    'train_model_mivia', 'test_model_loss_mivia', 'test_model_accuracy_mivia']

import numpy as np
import torch

from aer import accuracy as acc
from aer.datasets.mivia_loader import build_dataset, LIMITS
from ._models_tk import torch_dataset, train_batch

DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


##############################################################################
##                       Training & Testing on MIVIA                        ##
##############################################################################

def _build_params_dict(dataset_params, parsing_params):
    """ Fuse two input dicts to be 'train_model_mivia' format compliant """
    params = {'folders': dataset_params['folders'],
              'file_params': [None, dataset_params['SNR']],
              **parsing_params}
    return params

#------------------------   Model Training (MIVIA)   ------------------------#
def _train_model_mivia(model, loss_fct, optimizer, params, index, fit_params, rescale):
    """ Train the model on the MIVIA data files specified by `index` """
    off = min(index)
    losses = np.empty(len(index))
#    for i, ind in enumerate(index):
    for ind in index:
        print("Training on dataset", ind)
        # Select the current file
        params['file_params'][0] = ind
        # Load the data and build the chunks / sequences of chunks
        data, classes = build_dataset(**params, **rescale)[1:]
        # Convert the NumPy arrays into Torch tensors
        data = torch.Tensor(data).unsqueeze(1).to(DEVICE)  # NCin(H)W, with Cin=1
        classes = torch.Tensor(classes).to(DEVICE)         # N(H)Cout, with Cout=4
        # Wrap the data & classes into a Torch DataLoader
        dataset = torch_dataset(data, classes, **fit_params)
        # Train the model with the current dataset
        # (mean training loss returned, that is not used here)
        _ = train_batch(dataset, model, loss_fct, optimizer)
        # Evaluate the model's loss on the training data
        with torch.no_grad():
            losses[ind-off] = loss_fct(model(data), classes).item()
#            losses[i] = loss_fct(model(data), classes).item()
    return losses

def train_model_mivia(
    model, loss_fct, optimizer,
    dataset_params, parsing_params, index,
    epochs=1, bounds=(-1., +1.), seed=0, **fit_params):
    """ Train a model on a set of files from the MIVIA AE dataset

    Take a trainable Torch model & the specs of the MIVIA AE data files
    to use for training, and train the model on them. Perform training
    successively: for any file, load its data and parse them into chunks
    or sequences of chunks and train the model on them; then, do this
    with the next file and continue training on it; do so for any file.

    To load data, take the dataset (XML and WAV files locations and SNR)
    and parsing parameters (window and hop sizes for the chunks, and se-
    quences if so) and the indexes of the files to use for training, and
    pass them to the `build_dataset` function from `mivia_loader` module.
    Rescale the data between `bounds` before processing (the original
    data limits are retrieved from the `mivia_loader` module).

    See `train_batch` function for details about the training procedure.

    N.B.: the order of the files, specified by the `index` argument, is
        automatically shuffled at the beginning of every epoch; the data
        of a file/dataset can be shuffled in addition if the `shuffle`
        keyword argument is set to `True`.

    N.B.: for a given epoch, the data of a file are loaded and used to
        perform model training, and are then dropped to save memory; as
        such, they are reloaded for any epoch, costing some computation.

    Parameters
    ----------
    model : Torch Model
        The Torch model to train.
    loss_fct : Torch loss function
        The model evaluation loss function; see `torch.nn.modules.loss`.
    optimizer : Torch optimizer method
        The model weights update optimizer; see `torch.optim`. Must be
        linked to the model, e.g.: `torch.optim.SGD(model.parameters())`.
    dataset_params : 2-key dict
        The parameters of the dataset:
          - folders (2-tuple of strs): the paths to the XML & WAV files;
          - SNR (int): the (fixed) SNR of the dataset to use.
    parsing_params : 3-dict
        The parsing parameters to build the chunks, and sequences is so:
          - nb_classes (int): the number of possible classes (SoftMax);
                if omitted, retrieve it from the model output shape;
          - chk_params (2-tuple of ints): the chunks specs;
          - seq_params (2-tuple of ints): the sequences specs (do not
                provide this key set it to None for no sequence).
    index : iterable of ints
        The set of indexes of the files to use for training; can either
        be list-alike (list or tuple) or a range.
    [OPT] epochs : int
        The number of epochs for training, i.e. the number of times the
        files listed in `index` are loaded and used for training.
    [OPT] bounds : 2-tuple of floats
        The lower & upper bounds of the interval into which to rescale
        data, i.e. the new min & max.
            :Default: (-1., +1.)
    [OPT] seed : int
        The seed for the NumPy's RNG used to shuffle the list of indexes
        (i.e. the order of the sound files to load and use for training).
        Can be used to reproduce results while keeping semi-randomness.
            :Default: 0

    Other Parameters
    ----------------
    **fit_params : inline keyword arguments, optional
        The parameters for the Torch `DataLoader` class data wrapper. See
        the class `torch.utils.data.DataLoader` for details.
          - `batch_size` (int): number of samples per gradient update;
                must be > 1 if model contains `BatchNorm` layer.
          - `drop_last` (bool): ignore the last batch if incomplete
          - `shuffle` (bool): whether to shuffle the sound files data.
                Does not apply to the order of the sound files, that are
                automatically shuffled before every epoch.

    Returns
    -------
    losses : 1D or 2D np.ndarrays
        The loss scores achieved on the training data once the model is
        trained (`fit`), organized as a KxN array, with `K` the number
        of epochs and `N` is the number of datasets (length of `index`).
        If there is only 1 epoch, the 1st dimension is discarded, and
        `losses` becomes a 1D array only. Note that the losses are that
        returned by the Torch `loss_fct` function passed as argument.

    Examples
    --------
    # The aer project `datasets` package contains an example of the MIVIA
    # dataset, composed of two subdatasets of 1 file each: files 66 for
    # "training" and 29 for "testing", both with an SNR of 15 ('3').

    >>> import torch
    >>> import aer.datasets
    >>> import aer.datasets.mivia_loader as loader
    >>> from aer.models import SincNet

    # Folder containing the XML & WAV files
    >>> ROOT = f"{aer.datasets.__path__[0]}/MIVIA_AE_example/training/"
    >>> XMLFOLDER = ROOT
    >>> SNDFOLDER = ROOT + "sounds/"

    # Dataset index and SNR (only available one in the example training set)
    >>> IND, SNR = 66, 15

    # Dataset parameters
    >>> dataset_params = {'folders': (XMLFOLDER, SNDFOLDER), 'SNR': SNR}
    >>> FRATE = 32000
    >>> NB_CLASSES = 4

    # Parsing parameters
    #--- Chunks only variant
    >>> frm_dur = 100e-3                  # Frame duration in seconds
    >>> hop_dur = 50e-3                   # Hop duration in seconds
    >>> chk_params = (frm_dur, hop_dur)   # Chunks settings
    >>> parsing_params = {'nb_classes': NB_CLASSES,
    ...     'chk_params': chk_params, 'seq_params': None}
    >>> input_shape = (1, int(FRATE*frm_dur))

    #--- Chunks & Sequences variant
    >>> frm_dur = 50e-3                   # Frame duration in seconds
    >>> chk_params = (frm_dur, frm_dur)   # Chunks settings
    >>> seq_params = (10, 1)              # Sequences settings
    >>> parsing_params = {'nb_classes': NB_CLASSES,
    ...     'chk_params': chk_params, 'seq_params': seq_params}
    >>> input_shape = (1, seq_params[0], int(FRATE*frm_dur))

    # Build the model
    >>> scl_params = {'nb_filters': 80, 'filter_length': 251, 'frate': FRATE,
    ...               'bandwidth': (50., FRATE/2), 'padding': 'same'}
    >>> model = SincNet(input_shape, NB_CLASSES, scl_params,
    ...                 reg_linear=0.3, reg_conv=(3, 0.1))

    # Define the loss function
    >>> loss_fct = torch.nn.CrossEntropyLoss()

    # Define the optimizer to use for model training
    >>> optimizer = torch.optim.SGD(model.parameters(), lr=0.001, momentum=0.9)

    # Train the model on some files only (3 times the same here)
    >>> loss = train_model_mivia(model, loss_fct, optimizer,
    ...    dataset_params, parsing_params, (IND, IND, IND),
    ...    batch_size=4, drop_last=True, shuffle=True)

    # Train the model on a range of files (range of 1 here)
    >>> loss = train_model_mivia(model, loss_fct, optimizer,
    ...     dataset_params, parsing_params, range(IND, IND+1),
    ...     batch_size=4, drop_last=True, shuffle=True)

    # Train the model on a range of files with specific training parameters
    >>> loss = train_model_mivia(model, loss_fct, optimizer,
    ...     dataset_params, parsing_params, range(IND, IND+1),
    ...     epochs=10, batch_size=4, drop_last=True, shuffle=True)
    """
    # N.B.: `model.train(True)` is called in `train_batch`

    # Build the dictionary of the data files parameters
    if 'nb_classes' not in parsing_params.keys():
        parsing_params['nb_classes'] = model.output_shape[-1]
    params = _build_params_dict(dataset_params, parsing_params)

    # Check that 'index' complies with the 'rng.shuffle' function format
    if isinstance(index, range):
        index = np.arange(index.start, index.stop)  # If 'index' is a range
    elif isinstance(index, tuple):
        index = np.array(index)                     # Must be mutable

    # Train the model on the datasets
    rng = np.random.default_rng(seed)               # RNG to shuffle the indexes
    rescale = {'limits': LIMITS, 'bounds': bounds}  # Bounds to rescale the data
    if epochs == 1:
        rng.shuffle(index)
        losses = _train_model_mivia(
            model, loss_fct, optimizer, params, index, fit_params, rescale)
    else:
        losses = np.empty((epochs, len(index)))
        for i in range(epochs):
            print("Training epoch", i)
            rng.shuffle(index)
            losses[i] = _train_model_mivia(
                model, loss_fct, optimizer, params, index, fit_params, rescale)

    # Return the losses achieved on training
    return losses
#----------------------------------------------------------------------------#

#------------------------   Model Testing (MIVIA)   -------------------------#
def test_model_loss_mivia(
    model, loss_fct, dataset_params, parsing_params, index, bounds=(-1., +1.)):
    """ Evaluate model loss on a set of files from the MIVIA AE dataset

    Take a fully trained Torch model and the specs of the MIVIA AE data
    files to check accuracy on; test the data file one after the other.
    The data can either be organized into chunks or sequences of chunks.
    Rescale the data between `bounds` before processing (the original
    data limits are retrieved from the `mivia_loader` module).

    For both cases, the model accuracy is evaluated using the loss func-
    tion passed as the `loss_fct` argument.

    These evaluations are made for every dataset listed in `index` and
    gathered into a single list; if the data are chunks, every item of
    the list is a single dict, one per dataset; if they are sequences,
    every item is a set of two dicts, the first for the hits on chunks
    and the second one for those on sequences.

    To load data, take the dataset (XML and WAV files locations and SNR)
    and parsing parameters (window and hop sizes for the chunks, and se-
    quences if so) and the indexes of the files to use for testing, and
    pass them to the `build_dataset` function from `mivia_loader` module.

    N.B.: to save computation, the Torch model is automatically passed
        to `eval` mode (`model.eval()`) and the evaluation is performed
        with `torch.no_grad()`.

    Parameters
    ----------
    model : Torch Model
        The Torch model to evaluate; should be trained prior.
    loss_fct : Torch loss function
        The model evaluation loss function; see `torch.nn.modules.loss`.
    dataset_params : 2-key dict
        The parameters of the dataset:
          - folders (2-tuple of strs): the paths to the XML & WAV files;
          - SNR (int): the (fixed) SNR of the dataset to use.
    parsing_params : 3-dict
        The parsing parameters to build the chunks, and sequences is so:
          - nb_classes (int): the number of possible classes (SoftMax);
                if omitted, retrieve it from the model output shape;
          - chk_params (2-tuple of ints): the chunks specs;
          - seq_params (2-tuple of ints): the sequences specs (do not
                provide this key set it to None for no sequence).
    index : iterable of ints
        The set of indexes of the files to use for testing; can either
        be list-alike (list or tuple) or a range.
    [OPT] bounds : 2-tuple of floats
        The lower & upper bounds of the interval into which to rescale
        data, i.e. the new min & max.
            :Default: (-1., +1.)

    Returns
    -------
    loss : 1D np.ndarray
        The model evaluations made on the N datasets, with N the length
        of `index`, using the loss function specified as argument.

    Examples
    --------
    # The aer project `datasets` package contains an example of the MIVIA
    # dataset, composed of two subdatasets of 1 file each: files 66 for
    # "training" and 29 for "testing", both with an SNR of 15 ('3').

    >>> import torch
    >>> import aer.datasets
    >>> import aer.datasets.mivia_loader as loader
    >>> from aer.models import SincNet

    # Folder containing the XML & WAV files
    >>> ROOT = f"{aer.datasets.__path__[0]}/MIVIA_AE_example/training/"
    >>> XMLFOLDER = ROOT
    >>> SNDFOLDER = ROOT + "sounds/"

    # Dataset index and SNR (only available one in the example training set)
    >>> IND, SNR = 66, 15

    # Dataset parameters
    >>> dataset_params = {'folders': (XMLFOLDER, SNDFOLDER), 'SNR': SNR}
    >>> FRATE = 32000
    >>> NB_CLASSES = 4

    # Parsing parameters
    # Chunks variant, see the 'test_model_accuracy_mivia' example for sequences
    >>> frm_dur = 100e-3                  # Frame duration in seconds
    >>> hop_dur = 50e-3                   # Hop duration in seconds
    >>> chk_params = (frm_dur, hop_dur)   # Chunks settings
    >>> parsing_params = {'nb_classes': NB_CLASSES,
    ...     'chk_params': chk_params, 'seq_params': None}
    >>> input_shape = (1, int(FRATE*frm_dur))

    # Build the model
    >>> scl_params = {'nb_filters': 80, 'filter_length': 251, 'frate': FRATE,
    ...               'bandwidth': (50., FRATE/2), 'padding': 'same'}
    >>> model = SincNet(input_shape, NB_CLASSES, scl_params,
    ...                 reg_linear=0.3, reg_conv=(3, 0.1))

    # Define the loss function
    >>> loss_fct = torch.nn.CrossEntropyLoss()

    # Test the model on some files
    >>> loss = test_model_loss_mivia(
    ...     model, loss_fct, dataset_params, parsing_params, (IND,))

    # Test the model on a range of files (range of 1 here)
    >>> loss = test_model_loss_mivia(
    ...     model, loss_fct, dataset_params, parsing_params, range(IND, IND+1))
    """

    # Pass the model to `eval` mode to save computation
    model.eval()

    # Build the dictionary of the data files parameters
    if 'nb_classes' not in parsing_params.keys():
        parsing_params['nb_classes'] = model.output_shape[-1]
    params = _build_params_dict(dataset_params, parsing_params)

    # Load and test accuracy on every file
    rescale = {'limits': LIMITS, 'bounds': bounds}  # Bounds to rescale the data
    losses = np.empty(len(index))
    for i, ind in enumerate(index):
        print("Computing loss on dataset", ind)
        # Select the current file
        params['file_params'][0] = ind
        # Load the data and build the chunks / sequences of chunks
        data, classes = build_dataset(**params, **rescale)[1:]  # No `frate`
        # Convert the NumPy arrays into Torch tensors
        data = torch.Tensor(data).unsqueeze(1).to(DEVICE)  # Format NCW, with C=1
        classes = torch.Tensor(classes).to(DEVICE)
        # Evaluate the model's loss on the testing data
        with torch.no_grad():
            losses[i] = loss_fct(model(data), classes).item()

    # Return the losses achieved on testing
    return losses
#----------------------------------------------------------------------------#

##############################################################################



##############################################################################
##                      (Set of) Files Accuracy Check                       ##
##############################################################################

#------------------------   Check Model on Chunks   -------------------------#
def _test_chunks_mivia(model, params, index, rescale, conf_mat=True):
    """ Test the accuracy of a model on a set of MIVIA AE files whose
    data are organized into chunks only (no sequence) """
    # pylint: disable=E1101
    # Load and test accuracy on every file
    items = []
    for ind in index:
#        print("Computing classification accuracy on dataset", ind)
        # Select the current file
        params['file_params'][0] = ind
        # Load the data and build the chunks of data
        data, cls_ref = build_dataset(**params, **rescale)[1:]  # No `frate`
        tensor = torch.Tensor(data).unsqueeze(1).to(DEVICE)
        # Use the model to predict the classes in response to the chunks
        with torch.no_grad():
            cls_est = model.forward(tensor)
        # Compute the hit scores on the estimates (numpy arrays expected)
        items += acc.accuracy_chunks(
            cls_est.cpu().numpy(), cls_ref.cpu().numpy(), conf_mat)
    return items
#----------------------------------------------------------------------------#

#-----------------------   Check Model on Sequences   -----------------------#
def _test_sequences_mivia(model, params, index, rescale, conf_mat=True):
    """ Test the accuracy of a model on a set of MIVIA AE files whose
    data are organized into sequences of chunks """
    # pylint: disable=E1101
    # Load and test accuracy on every file
    items = []
    for ind in index:
#        print("Computing classification accuracy on dataset", ind)
        # Select the current file
        params['file_params'][0] = ind
        # Load the data and build the chunks / sequences of chunks
        data, cls_ref = build_dataset(**params, **rescale)[1:]  # No `frate`
        tensor = torch.Tensor(data).unsqueeze(1).to(DEVICE)
        # Use the model to predict the classes in response to the sequences
        with torch.no_grad():
            cls_est = model.forward(tensor)
        # Compute the hit scores on the estimates (numpy arrays expected)
        items += acc.accuracy_sequences(
            cls_est.cpu().numpy(), cls_ref.cpu().numpy(), conf_mat)
    return items
#----------------------------------------------------------------------------#

#-------------------------   Model Testing (Hits)   -------------------------#
def test_model_accuracy_mivia(
    model, dataset_params, parsing_params, index,
    bounds=(-1., +1.), conf_mat=True):
    """ Model classification accuracy on a set of MIVIA AE dataset's files

    Take a fully trained Torch model and the specs of the MIVIA AE data
    files to check accuracy on; test the data file one after the other.
    The data can either be organized into chunks or sequences of chunks.
    Rescale the data between `bounds` before processing (the original
    data limits are retrieved from the `mivia_loader` module).

    For both cases, the `hits` are the classification confusion matrices
    if `conf_mat` is True, or else the accuracy items (TPs, FPs, FNs and
    TNs, plus the amounts of 0s and 1s). See the `confusion_matrix` and
    `accuracy_items` functions from the `metrics` module for the details.

    These scores are computed for every dataset listed in `index` and
    gathered into a single list; if the data are chunks, every item of
    the list is a single dict, one per dataset; if they are sequences,
    every item is a set of two dicts, the first for the hits on chunks
    and the second one for those on sequences.

    To load data, take the dataset (XML and WAV files locations and SNR)
    and parsing parameters (window and hop sizes for the chunks, and se-
    quences if so) and the indexes of the files to use for testing, and
    pass them to the `build_dataset` function from `mivia_loader` module.

    N.B.: to save computation, the Torch model is automatically passed
        to `eval` mode (`model.eval()`) and the evaluation is performed
        with `torch.no_grad()`.

    Parameters
    ----------
    model : Torch Model
        The Torch model to evaluate; should be trained prior.
    dataset_params : 2-key dict
        The parameters of the dataset:
          - folders (2-tuple of strs): the paths to the XML & WAV files;
          - SNR (int): the (fixed) SNR of the dataset to use.
    parsing_params : 3-dict
        The parsing parameters to build the chunks, and sequences is so:
          - nb_classes (int): the number of possible classes (SoftMax);
                if omitted, retrieve it from the model output shape;
          - chk_params (2-tuple of ints): the chunks specs;
          - seq_params (2-tuple of ints): the sequences specs (do not
                provide this key set it to None for no sequence).
    index : iterable of ints
        The set of indexes of the files to use for testing; can either
        be list-alike (list or tuple) or a range.
    [OPT] bounds : 2-tuple of floats
        The lower & upper bounds of the interval into which to rescale
        data, i.e. the new min & max.
            :Default: (-1., +1.)
    [OPT] conf_mat : bool
        If True, build the confusion matrices, or the accuracy items else.
            :Default: True (build the confusion matrices)

    Returns
    -------
    hits : list of dicts
        Set of confusion matrices or of accuracy items. It is a list of
        1 dict if the hit scores are obtained on chunks, or of 2 dicts
        if obtained on sequences. In any case, every dict component is
        an array with length the number of classes. See `accuracy_chunks`
        and `accuracy_sequences` functions from `accuracy` module.

    Examples
    --------
    # The aer project `datasets` package contains an example of the MIVIA
    # dataset, composed of two subdatasets of 1 file each: files 66 for
    # "training" and 29 for "testing", both with an SNR of 15 ('3').

    >>> import torch
    >>> import aer.datasets
    >>> import aer.datasets.mivia_loader as loader
    >>> from aer.models import SincNet

    # Folder containing the XML & WAV files
    >>> ROOT = f"{aer.datasets.__path__[0]}/MIVIA_AE_example/training/"
    >>> XMLFOLDER = ROOT
    >>> SNDFOLDER = ROOT + "sounds/"

    # Dataset index and SNR (only available one in the example training set)
    >>> IND, SNR = 66, 15

    # Dataset parameters
    >>> dataset_params = {'folders': (XMLFOLDER, SNDFOLDER), 'SNR': SNR}
    >>> FRATE = 32000
    >>> NB_CLASSES = 4

    # Parsing parameters
    # Sequences variant, see the 'test_model_loss_mivia' example for chunks
    >>> frm_dur = 50e-3                   # Frame duration in seconds
    >>> chk_params = (frm_dur, frm_dur)   # Chunks settings
    >>> seq_params = (10, 1)              # Sequences settings
    >>> parsing_params = {'nb_classes': NB_CLASSES,
    ...     'chk_params': chk_params, 'seq_params': seq_params}
    >>> input_shape = (1, seq_params[0], int(FRATE*frm_dur))

    # Build the model
    >>> scl_params = {'nb_filters': 80, 'filter_length': 251, 'frate': FRATE,
    ...               'bandwidth': (50., FRATE/2), 'padding': 'same'}
    >>> model = SincNet(input_shape, NB_CLASSES, scl_params,
    ...                 reg_linear=0.3, reg_conv=(3, 0.1))

    # Build confusion matrices, not accuracy items
    >>> conf_mat = True

    # Test the model on some files
    >>> hits = test_model_accuracy_mivia(
    ...     model, dataset_params, parsing_params, (IND,),
    ...     (-1., +1.), conf_mat)

    # Test the model on a range of files (range of 1 here)
    >>> hits = test_model_accuracy_mivia(
    ...     model, dataset_params, parsing_params, range(IND, IND+1),
    ...     (-1., +1.), conf_mat)
    """

    # Pass the model to evaluation mode
    model.eval()

    # Define the rescaling limits for the MIVIA dataset
    rescale = {'limits': LIMITS, 'bounds': bounds}

    # Build the data files parameter dictionary
    params = _build_params_dict(dataset_params, parsing_params)
    if 'nb_classes' not in parsing_params.keys():
        parsing_params['nb_classes'] = model.output_shape[-1]

    # Compute the accuracy scores on sequences
    if ('seq_params' in parsing_params.keys()
        and parsing_params['seq_params'] is not None):
        return _test_sequences_mivia(model, params, index, rescale, conf_mat)

    # Or compute them on chunks
    return _test_chunks_mivia(model, params, index, rescale, conf_mat)
#----------------------------------------------------------------------------#

##############################################################################
