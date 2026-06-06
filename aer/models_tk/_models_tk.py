""" Toolkit to train & test Torch models

Authors: Dylan MOLINIE
Company: CEA Paris-Saclay, Gif-sur-Yvette, France
Contact: dylan.molinie@cea.fr
Date: June 2025
Last revised: April 2026

License: GPLv3
"""

__all__ = [
    'numpy_to_torch_dtype', 'torch_to_numpy_dtype',
    'build_model_name', 'parse_modname', 'torch_dataset',
    'train_batch', 'train_epoch', 'train_model',
    'test_model_loss', 'test_model_accuracy']

import numpy as np
import torch

from aer import accuracy as acc


##############################################################################
##                         NumPy & Torch Data Types                         ##
##############################################################################

# pylint: disable-next=line-too-long
# https://github.com/pytorch/pytorch/blob/e180ca652f8a38c479a3eff1080efe69cbc11621/torch/testing/_internal/common_utils.py#L349

# Dict of NumPy dtype -> torch dtype (when the correspondence exists)
_numpy_to_torch_dtype_dict = {
    np.bool_      : torch.bool,
    np.uint8      : torch.uint8,
    np.int8       : torch.int8,
    np.int16      : torch.int16,
    np.int32      : torch.int32,
    np.int64      : torch.int64,
    np.float16    : torch.float16,
    np.float32    : torch.float32,
    np.float64    : torch.float64,
    np.complex64  : torch.complex64,
    np.complex128 : torch.complex128
}

# Dict of torch dtype -> NumPy dtype
_torch_to_numpy_dtype_dict = {
    value : key for (key, value) in _numpy_to_torch_dtype_dict.items()}
_torch_to_numpy_dtype_dict.update({
    torch.bfloat16: np.float32,
    torch.complex32: np.complex64
})

def numpy_to_torch_dtype(np_dtype):
    """ Convert a NumPy datatype into a Torch datatype

    Taken from "torch/testing/_internal/common_utils.py".

    Parameters
    ----------
    np_dtype : NumPy class type or array dtype
        The NumPy datatype to convert into Torch datatype.

    Returns
    -------
    torch_dtype : Torch dtype
        The equivalent Torch datatype.

    Examples
    --------
    >>> import numpy as np

    >>> numpy_to_torch_dtype(np.float32)
    torch.float32

    >>> numpy_to_torch_dtype(np.arange(10).dtype)
    torch.int64
    """
    try:
        # NumPy class type (e.g. `np.float32`)
        return _numpy_to_torch_dtype_dict[np_dtype]
    except KeyError:
        # Numpy array dtype (e.g. `np.arange(5).dtype`)
        return _numpy_to_torch_dtype_dict[np_dtype.type]

def torch_to_numpy_dtype(torch_dtype):
    """ Convert a NumPy datatype into a Torch datatype

    Taken from "torch/testing/_internal/common_utils.py".

    Parameters
    ----------
    torch_dtype : Torch tensor dtype
        The Torch datatype to convert into NumPy datatype.

    Returns
    -------
    np_dtype : Numpy class type
        The equivalent NumPy datatype.

    Examples
    --------
    >>> import torch

    >>> torch_to_numpy_dtype(torch.float32)
    <class 'numpy.float32'>

    >>> torch_to_numpy_dtype(torch.arange(10).dtype)
    <class 'numpy.int64'>
    """
    # Both Torch type and tensor type are direct dtype
    return _torch_to_numpy_dtype_dict[torch_dtype]

##############################################################################



##############################################################################
##                             Model File Name                              ##
##############################################################################

def build_model_name(mod_type, rec_cel=None, dtype=None, snr=None, epoch=None):
    """ Build the name of the file to save

    Append every item one after the other, starting from the `mod_type`;
    insert an underscore '_' between items. If an item is None or empty
    string, ignore it. The items are appended in this order:
        {mod_type, rec_cel, dtype, snr, epoch}
    The resulting model name is thus:
        f'{mod_type}_{rec_cel}_{dtype}_{snr}_{epoch}'

    Parameters
    ----------
    mod_type : string
        The network base name (e.g. SincNet, DENet, etc.).
    [OPT] rec_cel : string
        The recurrent cell name.
            :Default: None
    [OPT] dtype : string
        The format of the input data (e.g. 'chks' or 'seqs').
            :Default: None
    [OPT] snr : int
        The SNR of the database.
            :Default: None
    [OPT] epoch : int
        The number of epochs used for training.
            :Default: None

    Returns
    -------
    modname : string
        The standardized model name.

    Examples
    --------
    >>> build_model_name('SincNet', None, 'chks', 10)
    'SincNet_chks_10'

    >>> build_model_name('SincNet', 'LSTM', 'chks')
    'SincNet_LSTM_chks'

    >>> build_model_name('SincNet', 'LSTM', 'chks', 10, 10)
    'SincNet_LSTM_chks_10_10'
    """
    modname = mod_type
    for item in (rec_cel, dtype, str(snr), str(epoch)):
        if item not in (None, '', 'None'):
            modname +=  "_" + item
    return modname

def parse_modname(modname):
    """ Parse the model name into more readable strings (for display)

    In a string, replace the '_' by ' $-$ ' (LaTeX format), add 'SNR'
    after 'Net' and replace 'chks' by 'Chunks' & 'seqs' by 'Sequences'.
    This function is designed to operate on the standardized model names
    generated by the `build_model_name` function and the returned string
    is typically designed to be used as a title for plots.

    Parameters
    ----------
    modname : string
        The model name, generated by the `build_model_name` function.

    Returns
    -------
    parsed_name : string
        The parsed model name.

    Examples
    --------
    >>> parse_modname(build_model_name('SincNet', None, 'chks', 10))
    'SincNet $-$ SNR 10 $-$ Chunks'

    >>> parse_modname(build_model_name('SincNet', 'LSTM', 'chks', 10))
    'SincNet $-$ SNR 10 $-$ LSTM $-$ Chunks'
    """
    modname = modname.replace("Net_", "Net_SNR ")
    modname = modname.replace("_", " $-$ ")
    modname = modname.replace("chks", "Chunks")
    modname = modname.replace("seqs", "Sequences")
    return modname

##############################################################################



##############################################################################
##                              Model Training                              ##
##############################################################################

#-------------------------   Build Torch Dataset   --------------------------#
def torch_dataset(inputs, labels, **kwargs):
    """ Wrap a data array and its labels into a Torch Dataset

    Take a Torch tensor of data and the corresponding objective labels
    and wrap them into a Torch `Dataset`, itself wrapped into an iter-
    able Torch `DataLoader`. If `inputs` and `labels` are not tensors
    (e.g. NumPy arrays), first cast them as it. Both tensors must have
    the same number of elements in their first dimension (batch).

    Parameters
    ----------
    inputs : Torch Tensor or array_like
        The data that serve as inputs for the network. If array, convert
        them into Torch Tensor prior.
    labels : Torch Tensor or array_like
        The data that serve as objectives for network training. If array,
        convert them into Torch Tensor prior.

    Other Parameters
    ----------------
    **kwargs : inline keyword arguments, optional
        The Torch's `DataLoader` class keyword arguments; cf. this class
        for details (`torch.utils.data.DataLoader`); useful keywords:
          - `batch_size` (int): mini batch size
          - `drop_last` (bool): ignore the last batch if incomplete
          - `shuffle` (bool): if the dataset should be shuffled

    Returns
    -------
    dataset : Torch `DataLoader` object
        The formatted dataset that can be used for training & testing.

    Examples
    --------
    >>> import torch

    >>> inputs = torch.rand(10, 3, 100)     # Format: N x C_{in} x W
    >>> labels = torch.rand(10, 4)          # Format: N x C_{out}

    >>> dataset = torch_dataset(inputs, labels,
    ...     batch_size=10, drop_last=True, shuffle=True)
    """
    # Check the inputs & labels, and cast them into Tensors if needed
    if not isinstance(inputs, torch.Tensor):
        inputs = torch.Tensor(inputs)
    if not isinstance(labels, torch.Tensor):
        labels = torch.Tensor(labels)
    # Build & return the Torch DataLoader object
    return torch.utils.data.DataLoader(
        torch.utils.data.TensorDataset(inputs, labels), **kwargs)
#----------------------------------------------------------------------------#

#------------------------   Train Model on Dataset   ------------------------#
def train_batch(dataset, model, loss_fct, optimizer):
    """ Train a Torch model on a batched dataset

    Take a Torch model, loss function and optimizer, and train the model
    on the provided dataset (cf. `torch_dataset` function). To this end,
    for every set (inputs, labels) comprised in `dataset`, pass `inputs`
    to the model to obtain its outputs, and pass both these outputs and
    `labels` to `loss_fct` to estimate the model loss. Then, perform the
    loss `backward` and step the optimizer to update the model weights
    (trainable parameters). For every set of the dataset, sum the loss
    and return the mean loss (sum(losses) / len(dataset)).

    N.B.: the model is set to `trainable` first (`model.train(True)`);
        set it to `eval` mode (`model.eval()`) once training is done.

    N.B.: the model is updated directly, no need to return it.

    Parameters
    ----------
    dataset : Torch DataLoader
        The iterable dataset to use for training; every item must be a
        set of a tensor of inputs to be fed into `model`, and a tensor
        of labels, that are the objective values to learn by the model
        and that are compared to the model response to the inputs.
    model : Torch Model
        The Torch model to train.
    loss_fct : Torch loss function
        The model evaluation loss function; see `torch.nn.modules.loss`.
    optimizer : Torch optimizer method
        The model weights update optimizer; see `torch.optim`. Must be
        linked to the model, e.g.: `torch.optim.SGD(model.parameters())`.

    Returns
    -------
    loss : float
        The mean loss, averaged over the whole dataset.

    Examples
    --------
    >>> import torch

    # Define a simple model
    >>> model = torch.nn.Sequential(
    ...     torch.nn.Linear(100, 1), torch.nn.Flatten()).to('cpu')

    # Define the loss function
    >>> loss_fct = torch.nn.CrossEntropyLoss()

    # Define the model optimizer
    >>> optimizer = torch.optim.SGD(model.parameters())

    # Build a simple random dataset
    >>> inputs, labels = torch.rand(10, 5, 100), torch.rand(10, 5)
    >>> dataset = torch_dataset(inputs, labels,
    ...     batch_size=2, drop_last=True, shuffle=True)

    # Train the model
    >>> loss = train_batch(dataset, model, loss_fct, optimizer)
    """

    # Make sure that the model is trainable
    model.train(True)

    # Perform training
    loss_tot = 0.
    for inputs, labels in dataset:
        # Reset the optimizer for every batch
        optimizer.zero_grad()
        outputs = model(inputs)
        # Compute the loss and its gradients
        loss = loss_fct(outputs, labels)
        loss.backward()
        # Update the model's trainable weights
        optimizer.step()
        # Accumulate the loss of all batches
        loss_tot += loss.item()

    # Return the mean loss over the dataset (mean of the mini batches)
    return loss_tot / len(dataset)
#----------------------------------------------------------------------------#

#--------------------   Train Model on Set of Datasets   --------------------#
def train_epoch(datasets, model, loss_fct, optimizer):
    """ Train a Torch model on a set of batched datasets (as an epoch)

    Take a Torch model, loss function and optimizer, and train the model
    on the datasets provided. For every dataset of `datasets`, call the
    `train_batch` function; see this function for the details; if it is
    directly a Torch `DataLoader` object, call the function only once.
    Return the list of the losses of every dataset.

    Parameters
    ----------
    datasets : (list/tuple of) Torch DataLoader object(s)
        The datasets to use for training; if list or tuple of DataLoader,
        each one is sequentially (no suffle) passed to the `train_batch`
        function; if direct DataLoader object, pass it to the function
        directly. Every item of a dataset must be a set of a tensor of
        inputs to be fed into `model` and a tensor of objective labels.
        See the `torch_dataset` function for details.
    model : Torch Model
        The Torch model to train.
    loss_fct : Torch loss function
        The model evaluation loss function; see `torch.nn.modules.loss`.
    optimizer : Torch optimizer method
        The model weights update optimizer; see `torch.optim`. Must be
        linked to the model, e.g.: `torch.optim.SGD(model.parameters())`.

    Returns
    -------
    loss : 1D array of floats
        The `loss_fct`'s returns, for every dataset, for every epoch.

    Examples
    --------
    >>> from random import shuffle
    >>> import torch
    >>> from torch.utils.tensorboard import SummaryWriter

    # Instantiate the TensorBoard writer
    >>> tb_writer = SummaryWriter("runs/model")

    # Define a simple model
    >>> model = torch.nn.Sequential(
    ...     torch.nn.Linear(100, 1), torch.nn.Flatten()).to('cpu')

    # Define the loss function
    >>> loss_fct = torch.nn.CrossEntropyLoss()

    # Define the model optimizer
    >>> optimizer = torch.optim.SGD(model.parameters())

    # Build a simple set of random datasets
    >>> datasets = []
    >>> for i in range(5):
    ...     datasets.append(torch_dataset(
    ...         torch.rand(10, 5, 100), torch.rand(10, 5),
    ...         batch_size=2, drop_last=True, shuffle=True))

    # Train the model with the datasets shuffled for every epoch
    # and save the current epoch loss in the TensorBoard
    >>> for epoch in range(5):
    ...     shuffle(datasets)
    ...     loss = train_epoch(datasets, model, loss_fct, optimizer)
    ...     tb_writer.add_scalar(f'Loss epoch #{epoch+1}', loss[-1], epoch+1)

    # Ensure every loss has been written in the TensorBoard
    >>> tb_writer.flush()
    """
    # N.B.: `model.train(True)` is called in `train_batch`

    # If `dataset` comprises several datasets (DataLoader)
    if isinstance(datasets, (list, tuple)):
        losses = np.empty(len(datasets))
        # Train the model on every dataset
        for i, dset in enumerate(datasets):
            print("Training on dataset", i)
            losses[i] = train_batch(dset, model, loss_fct, optimizer)

    # If it is directly a single dataset
    else:
        print("Training")
        losses = train_batch(datasets, model, loss_fct, optimizer)

    return losses
#----------------------------------------------------------------------------#

#----------------------------   Model Training   ----------------------------#
# pylint: disable-next=too-many-arguments, too-many-positional-arguments
def train_model(datasets, model, loss_fct, optimizer, epochs=1, seed=0):
    """ Train a model on a set of batched Torch datasets `epochs` times

    Take a Torch model, loss function and optimizer, and train the model
    on the provided datasets; repeat training `epochs` times. For every
    epoch, shuffle the list of datasets and call the `train_epoch` func-
    tion; see this function for the details. Return the losses for every
    dataset, for every epoch.

    Parameters
    ----------
    datasets : list of Torch DataLoader objects
        The datasets to use for training; the list is shuffled at every
        epoch. Every dataset must be a set of a tensor of inputs for the
        `model` and a tensor of objective labels. See the `torch_dataset`
        function for details.
    model : Torch Model
        The Torch model to train.
    loss_fct : Torch loss function
        The model evaluation loss function; see `torch.nn.modules.loss`.
    optimizer : Torch optimizer method
        The model weights update optimizer; see `torch.optim`. Must be
        linked to the model, e.g.: `torch.optim.SGD(model.parameters())`.
    [OPT] epochs : int
        The number of training epochs, i.e. the number of times all the
        datasets are used to train the model.
            :Default: 1
    [OPT] seed : int
        The NumPy randomizer seed; see `np.random.default_rng`.
            :Default: 0

    Returns
    -------
    loss : 1D or 2D arrays of floats
        The `loss_fct`'s returns, for every dataset, for every epoch.

    Examples
    --------
    >>> import torch

    # Define a simple model
    >>> model = torch.nn.Sequential(
    ...     torch.nn.Linear(100, 1), torch.nn.Flatten()).to('cpu')

    # Define the loss function
    >>> loss_fct = torch.nn.CrossEntropyLoss()

    # Define the model optimizer
    >>> optimizer = torch.optim.SGD(model.parameters())

    # Build a simple set of random datasets
    >>> datasets = []
    >>> for i in range(5):
    ...     datasets.append(torch_dataset(
    ...         torch.rand(10, 5, 100), torch.rand(10, 5),
    ...         batch_size=2, drop_last=True, shuffle=True))

    # Train the model with the datasets
    >>> losses = train_model(datasets, model, loss_fct, optimizer, 10)
    """
    # N.B.: `model.train(True)` is called in `train_batch`

    # Check that the `datasets` is a mutable list of DataLoaders
    if isinstance(datasets, torch.utils.data.dataloader.DataLoader):
        datasets = [datasets]
    elif isinstance(datasets, tuple):
        datasets = list(datasets)

    # Train the model on the datasets
    rng = np.random.default_rng(seed)       # RNG to shuffle the datasets
    losses = np.empty((epochs, len(datasets)))
    for i in range(epochs):
        print("Training epoch", i)
        rng.shuffle(datasets)
        losses[i] = train_epoch(datasets, model, loss_fct, optimizer)

    return np.squeeze(losses)
#----------------------------------------------------------------------------#

##############################################################################



##############################################################################
##                             Model Evaluation                             ##
##############################################################################

#---------------------------   Loss Evaluation   ----------------------------#
def test_model_loss(datasets, model, loss_fct):
    """ Loss of a model on a set of batched Torch datasets

    Take a Torch model and a loss function, and evaluate the model loss
    on every dataset provided. Return the losses for every dataset.

    Parameters
    ----------
    datasets : list of Torch DataLoader objects
        The datasets to use for loss evaluation. Every dataset must be a
        set of a tensor of inputs to be fed into `model` and a tensor of
        objective labels. See `torch_dataset` function for details.
    model : Torch Model
        The Torch model to evaluate; should be trained prior.
    loss_fct : Torch loss function
        The model evaluation loss function; see `torch.nn.modules.loss`.

    Returns
    -------
    loss : 1D array of floats
        The `loss_fct`'s returns for every dataset.

    Examples
    --------
    >>> import torch

    # Define a simple model
    >>> model = torch.nn.Sequential(
    ...     torch.nn.Linear(100, 1), torch.nn.Flatten()).to('cpu')

    # Define the loss function
    >>> loss_fct = torch.nn.CrossEntropyLoss()

    # Build a simple set of random datasets
    >>> datasets = []
    >>> for i in range(5):
    ...     datasets.append(torch_dataset(
    ...         torch.rand(10, 5, 100), torch.rand(10, 5),
    ...         batch_size=2, drop_last=True, shuffle=True))

    # Evaluate the loss of the model on the datasets
    >>> losses = test_model_loss(datasets, model, loss_fct)
    """

    # Check that the `datasets` is a mutable list of DataLoaders
    if isinstance(datasets, torch.utils.data.dataloader.DataLoader):
        datasets = [datasets]
    elif isinstance(datasets, tuple):
        datasets = list(datasets)

    # Pass the model to `eval` mode to save computation
    model.eval()

    # Load and test accuracy on every file
    losses = np.empty(len(datasets))
    for i, dset in enumerate(datasets):
        print("Computing loss on dataset", i)
        # Evaluate the model's loss on the data
        with torch.no_grad():
            losses[i] = loss_fct(
                model(dset.dataset.tensors[0]),
                dset.dataset.tensors[1]).item()

    return losses
#----------------------------------------------------------------------------#

#----------------------   Hits & Confusion Matrices   -----------------------#
def test_model_accuracy(datasets, model, conf_mat=True, sequences=False):
    """ Classification accuracy of a model on a set of batched Torch datasets

    Take a Torch model and evaluate its classification accuracy on every
    dataset provided. Return these scores for any dataset. The functions
    used for model evaluation are: `accuracy_chunks` if data chunks, and
    `accuracy_sequences` if data sequences, both from `accuracy` module;
    see these two functions for the details.

    Parameters
    ----------
    datasets : list of Torch DataLoader objects
        The datasets to use for loss evaluation. Every dataset must be a
        set of a tensor of inputs to be fed into `model` and a tensor of
        objective labels. See `torch_dataset` function for details.
    model : Torch Model
        The Torch model to evaluate; should be trained prior.
    [OPT] conf_mat : bool
        If True, build the confusion matrices, or the accuracy items else.
            :Default: True
    [OPT] sequences : bool
        If the data are sequences of chunks (True) or chunks (False).
            :Default: False

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
    >>> import torch

    # Define a simple model
    >>> model = torch.nn.Sequential(
    ...     torch.nn.Linear(100, 1), torch.nn.Flatten(-2)).to('cpu')

    # Build a simple set of random datasets
    >>> datasets = []
    >>> for i in range(5):
    ...     datasets.append(torch_dataset(
    ...         torch.rand(10, 3, 5, 100), torch.rand(10, 3, 5),
    ...         batch_size=2, drop_last=True, shuffle=True))

    # Evaluate the classification accuracy on the datasets
    >>> matrix = test_model_accuracy(datasets, model, True, sequences=True)
    """
    # Pass the model to evaluation mode
    model.eval()

    # Select the correct accuracy function depending on the data format
    if sequences:
        fct_acc = acc.accuracy_sequences
    else:
        fct_acc = acc.accuracy_chunks

    # Compute the accuracy scores on every dataset
    items = []
    for i, dset in enumerate(datasets):
        print("Computing classification accuracy on dataset", i)
        data, cls_ref = dset.dataset.tensors
        # Use the model to predict the classes in response to the chunks
        with torch.no_grad():
            cls_est = model.forward(data)
        # Compute the hit scores on the estimates (numpy arrays expected)
        items += fct_acc(
            cls_est.cpu().numpy(), cls_ref.cpu().numpy(), conf_mat)

    return items
#----------------------------------------------------------------------------#

##############################################################################
