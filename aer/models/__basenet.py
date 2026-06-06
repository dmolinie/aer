""" Base class for Torch-based AER models

Authors: Dylan MOLINIE
Company: CEA Paris-Saclay, Gif-sur-Yvette, France
Contact: dylan.molinie@cea.fr
Date: February 2026
Last revised: March 2026

License: GPLv3
"""

__all__ = ['_BaseNet']

from torch import nn

from aer.tools import check_keys, make_iter
from aer.layers._layers import conv_output_length, ExtractTensor


##############################################################################
##                              Miscellaneous                               ##
##############################################################################

def _check_reg_params(params, input_shape):
    """ Check the format of a conv. regularization parameters set.
    See the `_BaseNet` method with the same name for details """
    # If shape (<int>, <float>)
    if isinstance(params, (list, tuple)) and isinstance(params[-1], (int, float)):
        config = params
    # If shape [(<int>, <float>)]
    else:
        config = params[0]
    # If sequences of data, check that the pooling kernel is 2D
    if len(input_shape) > 2 and isinstance(config[0], int):
        config = ((1, config[0]), config[1])
    return config

##############################################################################



##############################################################################
##                     Base Class For Sound Recognition                     ##
##############################################################################

class _BaseNet(nn.Module):
    """ Base class for the AER models

    Class that allows to instantiate a Torch model and add layers deemed
    accurate for audio event recognition, namely convolutional, linear
    (fully connected) and recurrent (GRU or LSTM) layers. Methods allow
    to add regularization after either convolutional and FC layers.

    The input data can either be 3D chunks or 4D sequences of chunks of
    data; when necessary, the methods are implemented in both variants.

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
    __init__(input_shape, nb_classes, conv_params, reg_conv,
             nb_neurons_fc, reg_linear, rec_cell, nb_neurons_rec)

    Methods
    -------
    forward(x)
        Forward the input `x` to the model and retrieve its response.
    + Those inherited from torch.nn.Module.
    """

    #---------------------------   Constructor   ----------------------------#
    # pylint: disable-next=too-many-arguments, too-many-positional-arguments
    def __init__(self, input_shape, nb_classes,
                 conv_params, conv_params_ref, reg_conv,
                 nb_neurons_fc, reg_linear,
                 rec_cell, nb_neurons_rec):
        """ Instantiate a _BaseNet object (constructor)

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
        conv_params : (list of) dict(s)
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
        conv_params_ref : dict
            The default set of parameters for the convolutional layers;
            every dict of `conv_params` is compared to it. Example:
                {'out_channels': <int>, 'kernel_size': <int>,
                 'stride': <int>, 'padding': <str>}
        reg_conv : (list of) 2-tuples of 1 int & 1 float
            The parameters for the regularization applied after convolu-
            tional layers, one set per layer. The first item is the poo-
            ling size (int) and the second is the dropout rate (float);
            if dealing with 4D data sequences, the kernel size should be
            a 2-tuple of ints with the first set to 1 (kernel height);
            this is checked and corrected if needed if single int and 3D
            input shape. For any layer, set the kernel size to 0 to skip
            pooling, and dropout to 0. to skip dropout. If single tuple,
            wrap it into a list, and use it for every regularization.
        nb_neurons_fc : (list of) int(s)
            The numbers of neurons for the linear layers, one per layer.
            If single int, wrap it into a list. Pass an empty list [] to
            skip linear layers (except activation).
        reg_linear : (list of) float(s)
            The dropout rates for the regularization applied after the
            linear layers, one value per layer; for any layer, set the
            value to 0. to skip dropout. If single float, wrap it into
            a list, and use it for every regularization.
        rec_cell : string
            The recurrent cell to add; if the string contains 'gru', add
            a bidirectional GRU; if it contains 'lstm', add a bidirecti-
            onal LSTM; add no recurrent cell else. Not case sensitive.
        nb_neurons_rec : int
            The number of neurons for the recurrent cell; it is doubled
            as it is a bidirectional cell.

        Returns
        -------
        convnet : Torch Sequential Model class object
            The convolutional model. Can be trained later on.
        """

        # Constructor of the Torch's Model class
        super().__init__()

        # Store the model input and output shapes in class attributes
        self._input_shape = input_shape
        self._nb_classes = nb_classes

        # Instantiate the model
        self._model = nn.Sequential()

        # Parameters for the convolutional and regularization layers
        conv_params, reg_conv = self._check_conv_params(
            conv_params, reg_conv, conv_params_ref.copy(), input_shape)

        # Parameters for the FC (Linear) and regularization layers
        nb_neurons_fc, reg_linear = self._check_fc_params(
            nb_neurons_fc, reg_linear)

        # Store the model's (static) configuration parameters in an attribute
        self._config = {
            'conv_params': conv_params,
            'reg_conv': reg_conv,
            'nb_neurons_fc': nb_neurons_fc,
            'reg_linear': reg_linear,
            'rec_cell': rec_cell.lower(),
            'nb_neurons_rec': nb_neurons_rec}

        # Store the number of features outputted by the last layer called
        self._size = input_shape[-1]
    #------------------------------------------------------------------------#

    #---------------------   Check the Parameter Sets   ---------------------#
    @staticmethod
    def _check_reg_params(params, input_shape):
        """ Check the format of a conv. regularization parameters set

        If `params` is a tuple/list of 2 numerals, return this tuple;
        else, extract this last tuple of it and return it. The tuple
        is assumed to have shape: (<int: pooling>, <float: dropout>).

        If `input_shape` is 3D (i.e. data are sequences), replace the
        pooling kernel by a 2-tuple of 2 ints, with the first set to 1:
        ((1, <int: pooling>), <float: dropout>).

        Parameters
        ----------
        params : 2-tuple of 1 int and 1 float, possibly wrapped into a list
            The pooling kernel size (int) and the dropout rate (float).
        input_shape : 2- or 3-tuple of ints
            The shape of the unbatched input (either CW or CHW).

        Returns
        -------
        reg_conv : 2-tuple
            If `input_shape` is 2D, has shape (<int>, <float>); else,
            has shape ((1, <int>), <float>).
        """
        return _check_reg_params(params, input_shape)

    @staticmethod
    def _check_conv_params(conv_params, reg_conv, conv_params_ref, input_shape):
        """ Check the sets of parameters for the convolutional layers
        and their following regularization layers

        Take a list of parameters sets to be used for the convolutional
        layers and compare every set to a reference one (to fill their
        possible missing values). The number of dictionaries in the list
        defines the number of convolutional layers, one per layer.

        The returned list depends on the `conv_params` argument:
          - if `None`, wrap `conv_params_ref` into a list;
          - if it is a single dictionary, wrap it into a list;
          - if it is a tuple consisting of 1 int & 1 dict, repeat this
            dictionary this number of times in a list;
          - else, compare each of its dictionaries to `conv_params_ref`
            and wrap the checked sets into a list.
        Finally, return that list.

        The regularization parameters set is expected to be a list of
        2-tuples, each consisting of 1 int (maxpooling kernel size) and
        1 float (dropout rate). If `reg_conv` is a lone 2-tuple, repeat
        it as many times as there are dicts in (fulfilled) `conv_params`;
        else, use `reg_conv` directly.

        For both `conv_params` and `reg_conv`, if dealing with 3D unba-
        tched data sequences (NCHW), the layers will require 2D kernel
        sizes and strides. As such, if the length of `input_shape` is
        greater than 2 (3D unbatched data), for every `kernel_size` and
        `stride` keys of any dict of `conv_params`, check that they are
        2D, with the H kernel size set to 1 (since the sequences should
        be groups of 1D data); do the same with the maxpooling kernel of
        the regularization layers sets of parameters.

        Parameters
        ----------
        conv_params : (list of) dict(s)
            The sets of parameters for the convolutional layers to check
            and compare to `conv_params_ref`. See the `torch.nn.Conv1d`
            and `torch.nn.Conv2d` layers for details.
        reg_conv : (list of) 2-tuples of 1 int & 1 float
            The parameters for the regularization layers, as a list of
            2-tuples consisting of the kernel size and the dropout rate.
            If the pooling kernel size is 0, and/or dropout rate is 0.,
            do not use pooling and/or dropout after the corresponding
            convolutional layer.
        conv_params_ref : (list of) dict(s)
            The default set of parameters for the convolutional layers,
            whose values are used to fulfill the minimal missing values
            in the dicts of `conv_params`, if so. Typical shape is:
                {'out_channels': <int>, 'kernel_size': <int>,
                 'stride': <int>, 'padding': <str>}
        input_shape : 2- or 3-tuple of ints
            The shape of the unbatched input (either CW or CHW).

        Returns
        -------
        conv_params : list of dict
            The fulfilled list of dictionaries for the convolutional
            layers, one set of parameters per layer, and whose values
            are correctly formatted for sequences of data, if needed.
        reg_conv : list of 2-tuples
            The list of parameters for the regularization layers, one
            per convolutional layer, and whose values are formatted for
            sequences of data, if needed.
        """
        # pylint: disable=too-many-branches

        # Check the parameters for the convolutional layers
        if conv_params is None:
            # If no set of parameters provided, use the default one
            conv_params = [conv_params_ref]
        elif isinstance(conv_params, dict):
            # If only one set of parameters, check it and wrap it into a list
            conv_params = [check_keys(conv_params, conv_params_ref)]
        elif len(conv_params) == 2 and isinstance(conv_params[0], int):
            # If a number and one set, duplicate this set this number of times
            params = check_keys(conv_params[1], conv_params_ref)
            conv_params = [params for _ in range(conv_params[0])]
        elif len(conv_params) == 2 and isinstance(conv_params[1], int):
            # Idem previous 'elif', but with the set first and number last
            params = check_keys(conv_params[0], conv_params_ref)
            conv_params = [params for _ in range(conv_params[1])]
        else:
            # Else, check every dict of parameters
            conv_params = [
                check_keys(params, conv_params_ref) for params in conv_params]

        # Set a 2D kernel size, stride and reg. pooling if sequences (with V=1)
        if len(input_shape) > 2:
            for i, params in enumerate(conv_params):
                if isinstance(params['kernel_size'], int):
                    conv_params[i]['kernel_size'] = (1, params['kernel_size'])
                if isinstance(params['stride'], int):
                    conv_params[i]['stride'] = (1, params['stride'])

        # If only one tuple of parameters, use it for all conv reg. layers
        # The tuple are assumed to have shape: (pool, dropout)
        if len(reg_conv) == 1 or isinstance(reg_conv[0], (int, float)):
            reg_conv = _check_reg_params(reg_conv, input_shape)
            reg_conv = [reg_conv for _ in range(len(conv_params))]
        else:
            reg_conv = [_check_reg_params(reg, input_shape) for reg in reg_conv]

        # Duplicate the last reg. parameters if there are more conv. layers
        while len(conv_params) > len(reg_conv):
            reg_conv.append(reg_conv[-1])

        # Remove the superfluous reg. parameters if there are less conv. layers
        while len(conv_params) < len(reg_conv):
            reg_conv.pop()

        return conv_params, reg_conv

    @staticmethod
    def _check_fc_params(nb_neurons_fc, reg_linear):
        """ Check the parameters for the linear (FC) layers and their
        following regularization layers

        Take the numbers of neurons for the linear (FC) layers: if it is
        single int, wrap it into a list. Take also the dropout rate for
        the regularization layers: if it is a float, wrap it into a list
        and repeat this value the same number of times than the length
        of the `nb_neurons_fc` list; else, use `reg_linear` directly.

        Every pair of items from `nb_neurons_fc` and `reg_linear` is sup-
        posed to be used to add a single linear (FC) layer to the model;
        there are as many FC layers as there are such pairs.

        Parameters
        ----------
        nb_neurons_fc : (list of) int(s)
            The numbers of neurons for the linear (FC) layers, one per
            layer. If single int, wrap it into a list.
        reg_linear : (list of) float(s)
            The dropout rate for every linear layer; if single float,
            wrap it into a list and repeat it as many times as there
            are items in `nb_neurons_fc`. If 0., do not add a dropout
            layer after the corresponding FC layer.

        Returns
        -------
        nb_neurons_fc : list of ints
            The number of neurons for every Linear (FC) layer.
        reg_linear : list of floats
            The dropout rate for every regularization layer.
        """

        # Number of neurons for the FC layers (wrap it into a list if int)
        nb_neurons_fc = list(make_iter(nb_neurons_fc))

        # If only one int (the dropout rate) as reg. parameter for the FC
        # layers, use it for all FC reg. layers
        if isinstance(reg_linear, float):
            reg_linear = [reg_linear for _ in range(len(nb_neurons_fc))]
        else:
            reg_linear = list(reg_linear)

        # Duplicate the last reg. parameters if there are more linear layers
        while len(nb_neurons_fc) > len(reg_linear):
            reg_linear.append(reg_linear[-1])

        # Remove the superfluous reg. parameters if there are less linear layers
        while len(nb_neurons_fc) < len(reg_linear):
            reg_linear.pop()

        return nb_neurons_fc, reg_linear
    #------------------------------------------------------------------------#

    #----------------------------   Properties   ----------------------------#
    @property
    def input_shape(self):
        """ Return the shape of the input """
        return self._input_shape

    @property
    def nb_classes(self):
        """ Return the number of objective classes """
        return self._nb_classes

    @property
    def config(self):
        """ Return the configuration parameters of the model """
        return self._config

    @property
    def model(self):
        """ Return the whole model """
        return self._model
    #------------------------------------------------------------------------#

    #---------------------   Add Convolutional Layers   ---------------------#
    def _add_conv_layer_chks(self, i, in_channels, params, reg_conv):
        """ Add a convolutional layer (chunk variant)

        Take the parameters for the convolution (`torch.nn.Conv1d`) and
        add a convolutional layer to the class model (`model` attribute)
        as well as a regularization layer (see `_reg_conv_1d`).

        Parameters
        ----------
        i : int
            Index to identify the layer, which is named 'conv_{i}'.
        in_channels : int
            The number of input channels for the convolution.
        params : dict
            The parameters for the convolution; see the constructor.
        reg_conv : 2-tuple of 1 int & 1 float
            The tuple of parameters for regularization, organized as:
            (<int: kernel_size>, <float: dropout_rate>). If the kernel
            size is 0 and/or the dropout rate is 0., skip the pooling
            and/or dropout layers.

        Returns
        -------
        nb_out_channels : int
            The number of channels of the convolutional layer output.
        """
        # Add the convolutional layer
        self._model.add_module(f'conv_{i}', nn.Conv1d(
            in_channels=in_channels, **params))
        # Compute the size of the output
        self._size = conv_output_length(
            self._size, params['kernel_size'],
            params['padding'], params['stride'])
        # Add the regularization layer
        self._size = self._reg_conv_1d(i, self._size, *reg_conv)
        return params['out_channels']

    def _add_conv_layer_seqs(self, i, in_channels, params, reg_conv):
        """ Add a convolutional layer (sequence variant)

        Take the parameters for the convolution (`torch.nn.Conv2d`) and
        add a convolutional layer to the class model (`model` attribute)
        as well as a regularization layer (see `_reg_conv_2d`).

        Parameters
        ----------
        i : int
            Index to identify the layer, which is named 'conv_{i}'.
        in_channels : int
            The number of input channels for the convolution.
        params : dict
            The parameters for the convolution; see the constructor.
        reg_conv : 2-tuple of 1 2-tuple of ints & 1 float
            The tuple of parameters for regularization, organized as:
            ((1, <int: kernel_size>), <float: dropout_rate>). If the
            kernel size is 0 and/or the dropout rate is 0., skip the
            pooling and/or dropout layers.

        Returns
        -------
        nb_out_channels : int
            The number of channels of the convolutional layer output.
        """
        # Add the convolutional layer
        self._model.add_module(f'conv_{i}', nn.Conv2d(
            in_channels=in_channels, **params))
        # Compute the size of the output
        self._size = conv_output_length(
            self._size, params['kernel_size'][1],
            params['padding'], params['stride'][1])
        # Add the regularization l
        self._size = self._reg_conv_2d(i, self._size, *reg_conv)
        return params['out_channels']
    #------------------------------------------------------------------------#

    #------------------------   Add Recurrent Cell   ------------------------#
    def _add_rec_cell(self, sequences):
        """ Add a recurrent layer to the model

        If the `config['rec_cell']` attr. contains the 'lstm' string,
        add a bidirectional LSTM; add a bidirectional GRU else. Torch
        recurrent cells output tensor with shape (output, (h_n, c_n)),
        where `output` is the last layer output features, `h_n` & `c_n`
        are the final hidden and cell states for each element in the
        sequence; as such, add a layer to extract only `output` tensor
        (see the `ExtractTensor` class from the `layers` package).

        If `sequence` is True, return `output` directly; else, return
        only the last dimension of it by using another `ExtractTensor`
        layer to reduce the output dimension (for chunks in particular),
        so that the recurrent cell output can be fed into linear layers
        (no need for such an extraction with 4D sequences of data).

        Parameters
        ----------
        sequences : bool
            If True, return the previous states of the recurrent cells
            (the sequences); else, return only the last state, so that
            the tensor dimension is reduced (the chunks).

        Returns
        -------
        None : directly add the layers.
        """
        # Add the recurrent cell
        if 'lstm' in self._config['rec_cell']:
            self._model.add_module('lstm_cell', nn.LSTM(
                self._size, self._config['nb_neurons_rec'], bidirectional=True))
        else:
            self._model.add_module('bgru_cell', nn.GRU(
                self._size, self._config['nb_neurons_rec'], bidirectional=True))
        self._size = 2 * self._config['nb_neurons_rec']

        # Torch recurrent cells return 2-tuple: (output, (h_n, c_n)), with
        # 'output' the output features from the last layer, thus extract it
        self._model.add_module('extract_values', ExtractTensor(0))
        # If 3D chunks, extract the last output of the cell to flatten the tensor
        if not sequences:
            self._model.add_module('extract_last', ExtractTensor(slice(None), -1))
    #------------------------------------------------------------------------#

    #--------------------   Add Fully Connected Layers   --------------------#
    def _add_fc_layer_chks(self, i, nb_neurons, reg_linear):
        """ Add a fully connected layer (chunk variant)

        Take the number of neurons of the linear (FC) layer and add one
        to the model and a regularization stage (see `_reg_linear`).

        Parameters
        ----------
        i : int
            Index to identify the layer, which is named 'linear_{i}'.
        nb_neurons : int
            The number of output neurons for the linear layer.
        reg_linear : float
            The dropout rate for the regularization. If 0., skip it.

        Returns
        -------
        None : directly add the layers to the model (attribute).
        """
        # Add the linear layer
        self._model.add_module(f'linear_{i}', nn.Linear(self._size, nb_neurons))
        self._size = nb_neurons     # Must be updated between the two layers
        # Add the regularization layer
        self._reg_linear(i, self._size, reg_linear)

    def _add_fc_layer_seqs(self, i, nb_neurons, reg_linear):
        """ Add a fully connected layer (sequence variant)

        Take the number of neurons of the linear (FC) layer and add one
        to the model and a regularization stage (see `_reg_linear`).

        Parameters
        ----------
        i : int
            Index to identify the layer, which is named 'linear_{i}'.
        nb_neurons : int
            The number of output neurons for the linear layer.
        reg_linear : float
            The dropout rate for the regularization. If 0., skip it.

        Returns
        -------
        None : directly add the layers to the model (attribute).
        """
        # Add the linear layer
        self._model.add_module(f'linear_{i}', nn.Linear(self._size, nb_neurons))
        self._size = nb_neurons     # Must be updated between the two layers
        # Add the regularization layer
        self._reg_linear(i, self._input_shape[1], reg_linear)
    #------------------------------------------------------------------------#

    #----------------------   Add Activation Layers   -----------------------#
    def _add_activation_layer(self, nb_outputs):
        """ Add the activation/output layers (all variants)

        Take the number of outputs of the model and add a final linear
        layer to project the data into this space. Add also a SoftMax
        layer to obtain probabilities of belonging to every class.

        Parameters
        ----------
        nb_outputs : int
            The number of outputs neurons.

        Returns
        -------
        None : directly add the layers to the model (attribute).
        """
        self._model.add_module('linear_out', nn.Linear(self._size, nb_outputs))
        self._model.add_module('activation', nn.Softmax(-1))
    #------------------------------------------------------------------------#

    #------------------   Reg. For Convolutional Layers   -------------------#
    def _reg_conv_1d(self, i, length, pooling=0, dropout=0.):
        """ Regularization for the convolutional layers (chunks)

        Add a MaxPool, LayerNorm, LeakyReLU and 1D Dropout layers to the
        model (attribute). If `pooling` is 0, do not add a maxpool layer.
        If `dropout` is 0., do not apply dropout.

        Parameters
        ----------
        i : int
            Index to identify the layers.
        length : int
            The number of features of the data (last dimension).
        [OPT] pooling : int
            The maxpool kernel size. If 0, skip this layer.
                :Default: 0
        [OPT] dropout : float
            The dropout rate. If 0., skip this layer.
                :Default: 0.

        Returns
        -------
        length : int
            The number of features after the layers. Changed only if
            the pooling kernel size is greater than 0.
        """
        # Add the MaxPool layer
        if pooling != 0:
            self._model.add_module(f'maxpool{i}', nn.MaxPool1d(kernel_size=pooling))
            length //= pooling      # Size of the output if MaxPool
        # Apply a LayerNorm to the last dimension of the data (W)
        self._model.add_module(f'layernorm{i}', nn.LayerNorm(length))
        # Add a LeakyReLU activation
        self._model.add_module(f'leakyrelu{i}', nn.LeakyReLU())
        # Apply some dropout, if so
        if dropout != 0.:
            self._model.add_module(f'dropout{i}', nn.Dropout1d(p=dropout))
        return length

    def _reg_conv_2d(self, i, length, pooling=(1, 0), dropout=0.):
        """ Regularization for the convolutional layers (sequences)

        Add a MaxPool, LayerNorm, LeakyReLU and 2D Dropout layers to the
        model (attribute). If the last value of `pooling` is 0, do not
        add a maxpool layer. If `dropout` is 0., do not apply dropout.

        Parameters
        ----------
        i : int
            Index to identify the layers.
        length : int
            The number of features of the data (last dimension).
        [OPT] pooling : 2-tuple of ints
            The maxpool 2D kernel size. Since data should be sequences
            of chunks of 1D data, the vertical kernel size should be 1.
            Skip pooling if the second value is 0.
                :Default: (1, 0)
        [OPT] dropout : float
            The dropout rate. If 0., skip this layer.
                :Default: 0.

        Returns
        -------
        length : int
            The number of features after the layers. Changed only if
            the pooling kernel size is greater than 0.
        """
        # Add the MaxPool layer
        if pooling[1] != 0:
            self._model.add_module(f'maxpool{i}', nn.MaxPool2d(kernel_size=pooling))
            length //= pooling[1]   # Size of the output if MaxPool
        # Apply a LayerNorm to the last dimension of the data (W)
        self._model.add_module(f'layernorm{i}', nn.LayerNorm(length))
        # Add a LeakyReLU activation
        self._model.add_module(f'leakyrelu{i}', nn.LeakyReLU())
        # Apply some dropout, if so
        if dropout != 0.:
            self._model.add_module(f'dropout{i}', nn.Dropout2d(p=dropout))
        return length
    #------------------------------------------------------------------------#

    #-------------------   Reg. For Linear (FC) Layers   --------------------#
    def _reg_linear(self, i, channels, dropout=0.):
        """ Regularization for the linear layers (all variants)

        Add a BatchNorm, LeakyReLU and Dropout layers to the model.
        If dropout is 0., do not apply dropout.

        Parameters
        ----------
        i : int
            Index to identify the layers.
        channels : int
            The number of channels of the input tensor (for BatchNorm).
        [OPT] dropout : float
            The dropout rate. If 0., skip this layer.
                :Default: 0.

        Returns
        -------
        None : directly add the layers to the model (attribute).
        """
        # Add the BatchNorm & LeakyReLU layers
        self._model.add_module(f'batchnorm{i}', nn.BatchNorm1d(channels))
        self._model.add_module(f'leakyrelu{i}', nn.LeakyReLU())
        # Apply some dropout, if so
        if dropout != 0.:
            self._model.add_module(f'dropout{i}', nn.Dropout(p=dropout))
    #------------------------------------------------------------------------#

    #--------------------------   Forward Method   --------------------------#
    def forward(self, x):
        """ Forward a tensor to the network

        Parameters
        ----------
        x : Torch Tensor (NC(H)W format)
            The data to forward to the network.

        Returns
        -------
        output : Torch Tensor
            The network's output in response to input `x`.
        """
        # Unbatched chunk, unbatched sequence or unsqueezed unbatched chk/seq
        # Check `len(x) == 1` as the network input must have 1 channel only
        if len(x) == 1 and x.ndim in (2, 3) and x.shape[1] != 1:
            return self._model(x.unsqueeze(0))
        return self._model(x)
    #------------------------------------------------------------------------#

##############################################################################
