""" Torch-based sound recognition-oriented models

Authors: Dylan MOLINIE
Company: CEA Paris-Saclay, Gif-sur-Yvette, France
Contact: dylan.molinie@cea.fr
Date: March 2025
Last revised: March 2026

License: GPLv3
"""
# pylint: disable=duplicate-code

__all__ = ['ConvNet']

from torch import nn

from aer.layers._layers import SwapDims
from aer.models.__basenet import _BaseNet

# Default parameters for the convolution layers
_CONV_PARAMS_REF = {
    'out_channels': 60, 'kernel_size': 5, 'stride': 1, 'padding': 'valid'}


##############################################################################
##                          Convolutional Network                           ##
##############################################################################

class ConvNet(_BaseNet):
    """ Standard layers-only network based on the SincNet architecture

    Take the shape of an input tensor, instantiate a Torch model and add
    the layers of `ConvNet` in a sequential order.

    As usual in audio stream processing and classification, the audio
    stream should be decomposed into slices (chunks) upstream.

    The input data can either be a 2D set of 1D chunks, or a 3D set of
    2D sequences of 1D chunks of data. In the former case, the network
    uses 1D layers (modules); else, it uses 2D layers, but operating
    only on the chunks, so that the sequences are dealt independently.

    N.B.: this network is essentially a simpler version of SincNet (see
        `SincNet` class for details); indeed, SincLayer is replaced by
        a simple standard convolutional layer with similar specs.

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
    + Those inherited from `torch.nn.Module`.

    Constructor
    -----------
    __init__(input_shape, nb_classes,
             conv_params=None, reg_conv=(3, 0.1),
             nb_neurons_fc=2048, reg_linear=0.1,
             rec_cell='', nb_neurons_rec=1024)

    Methods
    -------
    forward(x)
        Forward the input `x` to the model and retrieve its response.
    + Those inherited from `torch.nn.Module`.

    Examples
    --------
    >>> import torch
    >>> from torchinfo import summary

    # Define batch size
    >>> batch = 5

    #--- Chunks (2D) variant -- Format: CW, with C=1
    >>> input_shape = (1, 3200)

    #--- Sequences (3D) variant -- Format: CHW, with C=1
    >>> input_shape = (1, 10, 1600)

    # Generate testing data
    >>> data = torch.arange(batch*np.prod(input_shape), dtype=torch.float32)
    >>> data = data.reshape(batch, *input_shape)

    # Set the model parameters
    >>> nb_classes = 4

    #--- Parameters for the convolutional layers
    # Skip convolutional layers
    >>> conv_params = []
    # 1 single layer
    >>> conv_params = {
    ...     'out_channels': 40, 'kernel_size': 10,
    ...     'stride': 1, 'padding': 'valid'}
    # The default convolutional layer repeated several times
    >>> conv_params = (3, None)
    # The same specific convolutional layer repeated several times
    >>> conv_params = {
    ...     'out_channels': 40, 'kernel_size': 10,
    ...     'stride': 1, 'padding': 'valid'}
    >>> conv_params = (3, conv_params)
    # Several convolutional layers with different parameters
    >>> conv_params1 = {'out_channels': 40, 'kernel_size': 10}
    >>> conv_params2 = {'out_channels': 60, 'kernel_size': 8}
    >>> conv_params3 = {'out_channels': 20, 'kernel_size': 5}
    >>> conv_params = (conv_params1, conv_params2, conv_params3)

    #--- Regularization for the convolutional layers
    # The same regularization for all convolutional layers
    >>> reg_conv = (3, 0.1)
    # Different regularization
    >>> reg_conv = ((3, 0.1), (2, 0.3), (2, 0.1))

    #--- Parameters for the linear layers
    # Skip linear layers
    >>> nb_neurons_fc = []
    # 1 single layer
    >>> nb_neurons_fc = 2048    # or (2048,)
    # Several layers
    >>> nb_neurons_fc = (2048, 1024, 512)

    #--- Regularization for the linear layers
    # The same regularization for all convolutional layers
    >>> reg_linear = 0.0
    # Different regularization
    >>> reg_linear = (0.1, 0.0, 0.1)

    # Parameters for the recurrent cell
    # Type of the cell
    >>> rec_cell = 'LSTM'
    # Number of neurons of the cell (doubled as it is bidirectional)
    >>> nb_neurons_rec = 1024

    # Instantiate the model
    >>> model = ConvNet(
    ...     input_shape, nb_classes,
    ...     conv_params, reg_conv,
    ...     nb_neurons_fc, reg_linear,
    ...     rec_cell, nb_neurons_rec)
    >>> summary(model, input_size=input_shape)

    # Forward the testing data to the model
    >>> model.eval()
    >>> output = model.forward(data)
    """

    #------------------------   ConvNet for Chunks   ------------------------#
    def _build_convnet_chks(self, channels=1):
        """ Chunks variant
        (2D data, organized as `1 channel x chunk_length`) """

#        self._size = self._input_shape[-1]

        # Convolutional layers
        for i, (params, reg_conv) in enumerate(
            zip(self.config['conv_params'], self.config['reg_conv']), 1):
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

    #----------------------   ConvNet for Sequences   -----------------------#
    def _build_convnet_seqs(self, channels=1):
        """ Sequence-of-chunk variant
        (3D data, organized as `1 channel x nb_chunks x chunk_length`) """

#        self._size = self._input_shape[-1]

        # Convolutional layers
        for i, (params, reg_conv) in enumerate(
            zip(self.config['conv_params'], self.config['reg_conv']), 1):
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
                 conv_params=None, reg_conv=(3, 0.1),
                 nb_neurons_fc=2048, reg_linear=0.1,
                 rec_cell='', nb_neurons_rec=1024):
        """ Instantiate a ConvNet object (constructor)

        Instantiate a Torch model, possibly consisting of:
          - convolutional layers and their regularization stage;
          - an optional recurrent cell;
          - linear (FC) layers and their regularization stage.
        Note that the tensor's dimensions are automatically flattened
        before the final linear layers.

        Except the flatten and activation layers (linear for projection
        + SoftMax), any of the above-mentioned layers can be skipped by
        passing empty lists [] to `conv_params` and `nb_neurons_fc` and
        an empty string '' to `rec_cell`; any combination is possible.

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

        A regularization is applied after every convolutional and linear
        layer:
          - Convolutional: MaxPool, LayerNorm, LeakyReLU and Dropout;
          - Linear: BatchNorm, LeakyReLU and Dropout.
        The regularization parameters can be specified layer by layer by
        using lists; if single values are passed, they are used for any
        regularization of a kind. Skip the MaxPool and Dropout layers by
        setting the pooling kernel to 0, and dropout rate to 0..

        The recurrent cell is optional and can be avoided by leaving the
        `rec_cell` argument empty; otherwise, either a bidirectional LSTM
        or GRU can be added, with `nb_neurons_rec` output neurons (this
        value is doubled as the recurrent cell is bidirectional here).

        Parameters
        ----------
        input_shape : 2- or 3-tuple of ints
            The unbatched C(H)W shape of the input tensor, organized as:
                (in_channels, nb_data_per_chunk)
            if chunks, or as:
                (in_channels, nb_chunks_per_seq, nb_data_per_chunk)
            if sequences of chunks.
        nb_classes : int
            The number of objective classes to learn (output dimension).
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
                :Default: 0.1
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
        convnet : Torch Sequential Model class object
            The trainable convolutional model. Can be trained later on.
        """

        # Instantiate the model and set the attributes
        super().__init__(
            input_shape, nb_classes, conv_params, _CONV_PARAMS_REF, reg_conv,
            nb_neurons_fc, reg_linear, rec_cell, nb_neurons_rec)

        # Build the model according to the input data shape
        if len(input_shape) == 2:
            self._build_convnet_chks(channels=1)
        else:
            self._build_convnet_seqs(channels=1)
    #------------------------------------------------------------------------#

##############################################################################
