""" Original Sinc Filterbank Network (SincNet)

Authors: Dylan MOLINIE
Company: CEA Paris-Saclay, Gif-sur-Yvette, France
Contact: dylan.molinie@cea.fr
Date: March 2025
Last revised: March 2026

License: GPLv3
"""
# pylint: disable=duplicate-code

__all__ = ['SincNet']

from torch import nn

from aer.tools import check_keys
from aer.layers._layers import conv_output_length, SwapDims
from aer.models.__basenet import _BaseNet
from aer.layers._sinclayer import SincLayer

# Default parameters for the convolution layers
_CONV_PARAMS_REF = {
    'out_channels': 60, 'kernel_size': 5, 'stride': 1, 'padding': 'valid'}
# Default parameters for SincLayer
_SCL_PARAMS_REF = {
    'nb_filters': 60, 'filter_length': 251, 'frate': 32000,
    'bandwidth': (None, None), 'padding': 'same'}


##############################################################################
##                         Sinc Filterbank Network                          ##
##############################################################################

class SincNet(_BaseNet):
    """ Original Sinc Filterbank Network

    Take the shape of an input tensor, instantiate a Torch model and add
    the layers of `SincNet` in a sequential order.

    SincNet is a DL-oriented solution that aims to identify & classify
    the sounds of interest in an audio stream. It relies on SincLayer,
    a trainable bank of band-pass filters conceived on sinc functions,
    whose training consists of identifying the filters lower and upper
    cutoff frequencies. Then, the outputs of this filterbank are passed
    into two convolutional and three dense layers to train the model to
    classify the events of interest. The model output is provided by a
    SoftMax layer, providing the "probabilities" for a chunk to belong
    to any of the `nb_classes` possible classes.

    As usual in audio stream processing and classification, the audio
    stream should be decomposed into slices (chunks) upstream.

    The input data can either be a 2D set of 1D chunks, or a 3D set of
    2D sequences of 1D chunks of data. In the former case, the network
    uses 1D layers (modules); else, it uses 2D layers, but operating
    only on the chunks, so that the sequences are dealt independently.

    N.B.: this function is greatly based on that proposed by the SincNet
        original authors (cf. https://github.com/mravanelli/SincNet).

         ―――――――――――――――――――――――――――――――――――――――――
        |  SincNet          |  Filters=60, L=251  |
         ―――――――――――――――――――――――――――――――――――――――――
        |           Time-local features           |
         ―――――――――――――――――――――――――――――――――――――――――
        |  1D Convolution   |  Filters=60, L=5    |
        |  Max Pooling      |  L=3                |
        |  Norm+LeakyReLU   |  LayerNorm          |
        |  Spatial Dropout  |  Rate: 0.1          |
        |  1D Convolution   |  Filters=60, L=5    |
        |  Max Pooling      |  L=3                |
        |  Norm+LeakyReLU   |  LayerNorm          |
        |  Spatial Dropout  |  Rate: 0.1          |
         ―――――――――――――――――――――――――――――――――――――――――
        |          Time-global features           |
         ―――――――――――――――――――――――――――――――――――――――――
        |  FC / Linear      |  Units: 2048        |
        |  Norm+LeakyReLU   |  BatchNorm          |
        |  Dropout          |  Rate: 0.3          |
        |  FC / Linear      |  Units: 1024        |
        |  Norm+LeakyReLU   |  BatchNorm          |
        |  Dropout          |  Rate: 0.3          |
        |  FC / Linear      |  Units: 512         |
        |  Norm+LeakyReLU   |  BatchNorm          |
        |  Dropout          |  Rate: 0.3          |
         ―――――――――――――――――――――――――――――――――――――――――
        |  SoftMax          |  Classifier         |
         ―――――――――――――――――――――――――――――――――――――――――

    Attributes
    ----------
    model : Torch Model class object
        The trainable Torch-based convolutional model.
    input_shape : tuple of ints
        The unbatched shape of the input (C(H)W data format).
    nb_classes : int
        The number of output classes.
    config : dict
        The configuration parameters of the model
    + Those inherited from torch.nn.Module.

    Constructor
    -----------
    __init__(input_shape, nb_classes,
             scl_params=None, reg_scl=(3, 0.1),
             conv_params=None, reg_conv=(3, 0.1),
             nb_neurons_fc=2048, reg_linear=0.3,
             rec_cell='', nb_neurons_rec=1024)

    Methods
    -------
    forward(x)
        Forward the input `x` to the model and retrieve its response.
    + Those inherited from torch.nn.Module.

    Examples
    --------
    #--- See class `ConvNet` example for details on the model options
    #--- See class `DENet` for an example with 3D sequences of chunks

    >>> import numpy as np
    >>> import torch
    >>> import aer.datasets
    >>> import aer.models_tk as mtk
    >>> from aer.datasets.mivia_loader import build_dataset

    #--- Load the data from file
    # The aer project `datasets` package contains an example of the MIVIA
    # dataset, composed of two subdatasets of 1 file each: files 66 for
    # "training" and 29 for "testing", both with an SNR of 15 ('3').

    # Dataset parameters
    >>> ROOT = f"{aer.datasets.__path__[0]}/MIVIA_AE_example/training/"
    >>> XMLFOLDER = ROOT
    >>> SNDFOLDER = ROOT + "sounds/"

    # Extension of the audio file -- Read file "00066_3.wav"
    >>> IND, SNR = 66, 15

    # Chunk parameters
    >>> frm_duration = 100e-3               # Frame duration in seconds
    >>> hop_duration = 50e-3                # Hop duration in seconds
    >>> nb_classes = 4                      # Number of possible classes

    # Chunks settings
    >>> chk_params = (frm_duration, hop_duration)

    # Read the audio file and build the chunks of data & classes
    >>> specs, data, classes = build_dataset(
    ...     (XMLFOLDER, SNDFOLDER), (IND, SNR), nb_classes, chk_params)

    # Turn the data & classes into Torch Tensor
    >>> data = torch.Tensor(data).unsqueeze(1)  # NCW format, with C=1
    >>> classes = torch.Tensor(classes)

    # Wrap the input data & the output classes into a Torch DataLoader
    >>> dataset = mtk.torch_dataset(
    ...     data, classes, batch_size=4, shuffle=True, drop_last=True)

    #--- Train & use the network

    # Retrieve the data parameters
    >>> # nb_classes = classes.shape[-1]    # Number of classes
    >>> # nb_frames = data.shape[0]         # Nb of frames (chunks)
    >>> # len_frames = data.shape[1]        # Nb of samples per chunk
    >>> input_shape = data.shape[1:]        # Unbatched data format: CW

    # Convolutional layer parameters
    >>> conv_params = {'out_channels': 60, 'kernel_size': 5,
    ...                'stride': 1, 'padding': 'valid'}

    # SincLayer parameters
    >>> scl_params = {'nb_filters': 60, 'filter_length': 251, 'frate': specs[0],
    ...               'bandwidth': (50., specs[0]/2), 'padding': 'same'}

    # Build the model
    >>> model = SincNet(input_shape, nb_classes,
    ...                 scl_params=scl_params, reg_scl=(3, 0.1),
    ...                 conv_params=(2, conv_params), reg_conv=(3, 0.1),
    ...                 nb_neurons_fc=[2048, 1024, 512], reg_linear=0.3,
    ...                 rec_cell='')

    # Define the loss function
    >>> loss_fct = torch.nn.CrossEntropyLoss()

    # Define the model optimizer
    >>> optimizer = torch.optim.SGD(model.parameters())

    # The the model
    >>> loss = mtk.train_batch(dataset, model, loss_fct, optimizer)

    # Pass the model in evaluation mode (no more training)
    >>> model.eval()

    # Predict only one chunk
    >>> y = model.forward(data[0])

    # Predict several chunks at once
    >>> y = model.forward(data[-10:])
    """

    #------------------------   SincNet for Chunks   ------------------------#
    def _build_sincnet_chks(self):
        """ Chunks variant
        (2D data, organized as `1 channel x chunk_length`) """

#        self._size = self._input_shape[-1]

        # Filter the input data with the SincLayer filterbank
        scl_params = self.config['scl_params']
        self.model.add_module(
            'sinclayer', SincLayer(**scl_params))
        self._size = conv_output_length(
            self._size, scl_params['filter_length'], scl_params['padding'])
        self._size = self._reg_conv_1d(
            1, self._size, *self.config['reg_scl'])

        # Convolutional layers
        channels = scl_params['nb_filters']
        for i, (params, reg_conv) in enumerate(
            zip(self.config['conv_params'], self.config['reg_conv']), 2):
            channels = self._add_conv_layer_chks(i, channels, params, reg_conv)

        # Apply the recurrent layer if any specified
        # (remove one dim. by concatenating the channels of the chunks)
        if 'gru' in self.config['rec_cell'] or 'lstm' in self.config['rec_cell']:
            self._add_rec_cell(sequences=False)
        else:
            self.model.add_module('flatten', nn.Flatten())
            self._size *= channels

        # Final classification stage: reduce the number of outputs with a
        # cascade of FC layers, whose numbers of outputs gradually decrease
        for i, (nb_neurons, reg_linear) in enumerate(
            zip(self.config['nb_neurons_fc'], self.config['reg_linear']), 1):
            self._add_fc_layer_chks(i, nb_neurons, reg_linear)

        # Output the classes (activation)
        self._add_activation_layer(nb_outputs=self.nb_classes)
    #------------------------------------------------------------------------#

    #----------------------   SincNet for Sequences   -----------------------#
    def _build_sincnet_seqs(self):
        """ Sequence-of-chunk variant
        (3D data, organized as `1 channel x nb_chunks x chunk_length`) """

#        self._size = self._input_shape[-1]

        # Filter the input data with the SincLayer filterbank
        scl_params = self.config['scl_params']
        self.model.add_module(
            'sinclayer', SincLayer(**scl_params))
        self._size = conv_output_length(
            self._size, scl_params['filter_length'], scl_params['padding'])
        self._size = self._reg_conv_2d(
            1, self._size, *self.config['reg_scl'])

        # Convolutional layers
        channels = scl_params['nb_filters']
        for i, (params, reg_conv) in enumerate(
            zip(self.config['conv_params'], self.config['reg_conv']), 2):
            channels = self._add_conv_layer_seqs(i, channels, params, reg_conv)

        # Reverse the sequences and channels dimensions, and flatten
        # the channels (concatenate them for every sequence chunk)
        self.model.add_module('swapdims', SwapDims(1, 2))
        self.model.add_module('flatten', nn.Flatten(2, 3))
        self._size *= channels

        # Apply the recurrent layer if any specified
        # (expects 3 dimensions exactly with sequences)
        if 'gru' in self.config['rec_cell'] or 'lstm' in self.config['rec_cell']:
            self._add_rec_cell(sequences=True)

        # Final classification stage: reduce the number of outputs with a
        # cascade of FC layers, whose numbers of outputs gradually decrease
        for i, (nb_neurons, reg_linear) in enumerate(
            zip(self.config['nb_neurons_fc'], self.config['reg_linear']), 1):
            self._add_fc_layer_seqs(i, nb_neurons, reg_linear)

        # Output the classes (activation)
        self._add_activation_layer(nb_outputs=self.nb_classes)
    #------------------------------------------------------------------------#

    #---------------------------   Constructor   ----------------------------#
    # pylint: disable-next=too-many-arguments, too-many-positional-arguments
    def __init__(self, input_shape, nb_classes,
                 scl_params=None, reg_scl=(3, 0.1),
                 conv_params=None, reg_conv=(3, 0.1),
                 nb_neurons_fc=2048, reg_linear=0.3,
                 rec_cell='', nb_neurons_rec=1024):
        """ Instantiate a SincNet object (constructor)

        Instantiate a Torch model consisting of:
          - a sinc filterbank (SincLayer);
          - convolutional layers and their regularization stage;
          - an optional recurrent cell;
          - linear (FC) layers and their regularization stage.
        See the `SincLayer` class from `layers` module. Note that the
        tensor's dimensions are automatically flattened before the fi-
        nal linear layers.

        Except the SincLayer, flatten and activation layers (linear for
        projection + SoftMax), the above-mentioned layers can be skipped
        by passing empty lists [] to `conv_params` and `nb_neurons_fc`
        and an empty string '' to `rec_cell`; any combination possible.
        SincLayer is always added as very first layer.

        For both the convolutional & linear layers, their parameters can
        be passed as arguments as lists (cf. `Parameters`); the numbers
        of items in each list define the number of layers. For instance,
        if `conv_params` is a list of 2 dictionaries, the model will fea-
        ture two convolutional layers, the first with the first dict of
        the list, and the second with the second dict of the list. Idem
        for `nb_neurons_fc`, that is a list of ints that define the number
        of output neurons of every linear layer. For both lists, if a
        single value (dict for `conv_params` and int for `nb_neurons_fc`)
        is provided, only one layer of this kind is added.

        The same set of parameters for every convolutional layer can be
        used by passing a 2-tuple to `conv_params`, consisting of 1 int
        (the number of layers) and the dictionary of parameters:
            (<int: number_of_layers>, <dict: parameters>)

        A regularization is applied after SincLayer and every convolu-
        tional and linear layer:
          - Convolutional: MaxPool, LayerNorm, LeakyReLU and Dropout;
          - Linear: BatchNorm, LeakyReLU and Dropout.
        For SincLayer, the regularization parameters can be passed using
        the `reg_scl` argument. For the convolutional and linear layers,
        the regularization parameters can be specified layer by layer by
        using lists; if single values are passed, they are used for any
        regularization of a kind. See the `Parameters` section below for
        details about these arguments.

        The recurrent cell is optional and can be avoided by leaving the
        `rec_cell` argument empty; otherwise, either a bidirectional LSTM
        or GRU can be added, with `nb_neurons_rec` output neurons (this
        value is doubled as the recurrent cell is bidirectional here).

        Parameters
        ----------
        input_shape : 2- or 3-tuple of ints
            The unbatched C(H)W shape of the input tensor, organized as:
                (in_channels=1, nb_data_per_chunk)
            if chunks, or as:
                (in_channels=1, nb_chunks_per_seq, nb_data_per_chunk)
            if sequences of chunks.
            N.B.: `in_channels` must always be 1 with this network.
        nb_classes : int
            The number of objective classes to learn (output dimension).
        [OPT] scl_params : dict
            The parameters for SincLayer; see the `SincLayer` class from
            the `layers` module for details; the main arguments are:
              - nb_filters (int): the number of filters of the bank.
              - filter_length (int): the length of the filters.
              - frate (float): the sampling frequency.
              - bandwidth (2-tuple of floats): the filterbank bandwidth,
                    as a 2-tuple organized as (freq_min, freq_max).
              - padding (str): the padding to use for the convolutions;
                    must be among {'same', 'valid'}.
                :Default: if omitted (None), use the default values:
                    {'nb_filters': 60, 'filter_length': 251,
                     'frate': 32000, 'bandwidth': (None, None),
                     'padding': 'same'}
        [OPT] reg_scl : 2-tuple of 1 int & 1 float
            The parameters for the regularization done after SincLayer;
            the 1st item is the pooling kernel size (int) and the 2nd
            the dropout rate (float). Set the kernel size to 0 to skip
            pooling, and dropout to 0. to skip dropout. See `reg_conv`
            parameter below for additional details.
                :Default: (3, 0.1)
        [OPT] conv_params : (list of) dict(s) or 2-tuple of 1 int & 1 dict
            The list of parameters dicts for the convolutional layers
            (see `torch.nn.Conv1d` & `torch.nn.Conv2d`), one per layer.
            If None, use the default values; if lone dict, wrap it into
            a list. If list of dicts, use them directly. Every dict pro-
            vided is checked, and any missing items (keys) is fulfilled
            with the default ones, if needed. One can also use the same
            dict of parameters (or None for the default values) for sev-
            eral convolutional layers by providing a 2-tuple of 1 int &
            1 dict; in such a case, the dict is repeated that number of
            times in a list. Pass an empty list [] to skip these layers.
                :Default: if omitted (None), use the default values:
                    {'out_channels': 60, 'kernel_size': 5,
                     'stride': 1, 'padding': 'valid'}
        [OPT] reg_conv : (list of) 2-tuples of 1 int & 1 float
            The parameters for the regularization applied after convolu-
            tional layers, one set per layer. The first item is the poo-
            ling size (int) and the second is the dropout rate (float);
            if dealing with 4D data sequences, the kernel size should be
            a 2-tuple of ints with the first set to 1 (kernel height);
            this is checked and corrected if needed if single int and 3D
            input shape. For any layer, set the kernel size to 0 to skip
            pooling, and dropout to 0. to skip dropout. If single tuple,
            wrap it into a list, and use it for every regularization.
                :Default: (3, 0.1)
        [OPT] nb_neurons_fc : (list of) int(s)
            The numbers of neurons for the linear layers, one per layer.
            If single int, wrap it into a list. Pass an empty list [] to
            skip linear layers (except activation).
                :Default: 2048
        [OPT] reg_linear : (list of) float(s)
            The dropout rates for the regularization applied after the
            linear layers, one value per layer; for any layer, set the
            value to 0. to skip dropout. If single float, wrap it into
            a list, and use it for every regularization.
                :Default: 0.3
        [OPT] rec_cell : string
            The recurrent cell to add; if the string contains 'gru', add
            a bidirectional GRU; if it contains 'lstm', add a bidirecti-
            onal LSTM; add no recurrent cell else. Not case sensitive.
                :Default: '' (no recurrent cell)
        [OPT] nb_neurons_rec : int
            The number of neurons for the recurrent cell; it is doubled
            as it is a bidirectional cell.
                :Default: 1024

        Returns
        -------
        sincnet : Torch Sequential Model class object
            The trainable SincNet model. Can be trained later on.
        """
        ##chunks of 200 ms, with 10 ms overlap
        ##RMSprop (lr=0.001, alpha=0.95, eps=10^-7, minibatches=128)

        # Instantiate the model and set the attributes
        super().__init__(
            input_shape, nb_classes, conv_params, _CONV_PARAMS_REF, reg_conv,
            nb_neurons_fc, reg_linear, rec_cell, nb_neurons_rec)

        # Add the parameters of SincLayer to the `config` attribute
        self.config['scl_params'] = check_keys(scl_params, _SCL_PARAMS_REF)
        self.config['reg_scl'] = self._check_reg_params(reg_scl, input_shape)

        # Build the model according to the input data shape
        if len(input_shape) == 2:
            self._build_sincnet_chks()
        else:
            self._build_sincnet_seqs()
    #------------------------------------------------------------------------#

##############################################################################
