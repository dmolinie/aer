""" Denoising-Enhancement Layers

Authors: Dylan MOLINIE
Company: CEA Paris-Saclay, Gif-sur-Yvette, France
Contact: dylan.molinie@cea.fr
Date: February 2025
Last revised: March 2026

License: GPLv3

The DELayer was proposed by A. Greco, A. Roberto, A. Saggese & M. Vento
in "DENet: a deep architecture for audio surveillance applications", in
Neural Computing and Applications, 33(17):11273-11284, September 2021.

The implementation below is a fully documented, completely reworked and
cleaned-up version of the original authors' own implementation, freely
accessible at https://github.com/MiviaLab/DENet.

This implementation runs with Torch 2.9.1 in Python 3.12, and runs on
Ubuntu 24.04.
"""

__all__ = ['DELayer', 'attention_model']

import torch
from torch import nn

from aer.layers._layers import SwapDims, conv_output_length


##############################################################################
##                       Default DELayer Inner Model                        ##
##############################################################################

#------------------------   Default DELayer Model   -------------------------#
def attention_model(input_shape, out_channels=1):
    """ Instantiate the default Denoising-Enhancement inner model

    Instantiate the DELayer original attention model, composed of two
    parts: a first succession of convolutional layers, then followed
    by a set of fully connected layers.

    The first, convolutional part is defined as:
         ―――――――――――――――――――――――――――――――――――――――――――
        |  1D Convolution  |  Filters=30, L=7, S=2  |
        |  1D Convolution  |  Filters=30, L=7, S=3  |
        |  1D Convolution  |  Filters=10, L=7, S=3  |
         ―――――――――――――――――――――――――――――――――――――――――――
    Then, the output of this is flattened and fed into a set of fully
    connected layers, whose architecture depends on `out_channels`. If
    it is 1, as defined in the original DELayer, the layers are:
         ―――――――――――――――――――――――――――――――――――――――――――
        |  FC / Linear     |  Units: 128            |
        |  FC / Linear     |  Units: 64             |
        |  FC / Linear     |  Units: 1              |
         ―――――――――――――――――――――――――――――――――――――――――――
    Else, if `out_channels` is higher than 1, only one layer is used,
    composed of `out_channels` outputs:
         ―――――――――――――――――――――――――――――――――――――――――――
        |  FC / Linear     |  Units=`out_channels`  |
         ―――――――――――――――――――――――――――――――――――――――――――

    N.B.: the data format must be NC(H)W to comply with the PyTorch format.

    Parameters
    ----------
    input_shape : 2- or 3-tuple of ints
        The unbatched C(H)W shape of the input tensor, required to comp-
        ute the number of input neurons for the Linear layers,and select
        the right model (`chunks` or `sequences` variant). Organized as:
            (in_channels=1, nb_data_per_chunk)
        if chunks, or as:
            (in_channels=1, nb_chunks_per_seq, nb_data_per_chunk)
        if sequences of chunks.
    [OPT] out_channels : strictly positive integer
        The number of channels of the output (i.e. of neurons in the
        output layer), as specified in the above description.
            :Default: 1

    Returns
    -------
    inner_model : Torch (Sequential) model
        The default inner model for DELayer.

    Examples
    --------
    >>> from torchinfo import summary

    # Model with only one output channel
    >>> input_shape = (80, 3200)
    >>> inner_model = attention_model(input_shape, 1)

    # Model with the same number of outputs channels than inputs'
    >>> inner_model = attention_model(input_shape, input_shape[0])

    >>> summary(inner_model)
    =================================================================
    Layer (type:depth-idx)                   Param #
    =================================================================
    Sequential                               --
    ├─Conv1d: 1-1                            16,830
    ├─Conv1d: 1-2                            6,330
    ├─Conv1d: 1-3                            2,110
    ├─Flatten: 1-4                           --
    ├─Linear: 1-5                            140,080
    ├─ReLU: 1-6                              --
    =================================================================
    Total params: 165,350
    Trainable params: 165,350
    Non-trainable params: 0
    =================================================================
    """

    # Convolution & Linear layers common parameters
    kernel_size = 7
    padding = 'valid'
    lin_bias = True

    # Instantiate the model (sequential)
    model = torch.nn.Sequential()

    # Add the convolutional layers
    if len(input_shape) == 2:
        # Chunks -- 1D convolutions with W='kernel_size'
        model.add_module('del_conv_1', nn.Conv1d(
            input_shape[0], 30, kernel_size, stride=2, padding=padding))
        model.add_module('del_conv_2', nn.Conv1d(
            30, 30, kernel_size, stride=3, padding=padding))
        model.add_module('del_conv_3', nn.Conv1d(
            30, 10, kernel_size, stride=3, padding=padding))
    else:
        # Sequences of chunks -- 2D convolutions with H=1 and W='kernel_size'
        model.add_module('del_conv_1', nn.Conv2d(
            input_shape[0], 30, (1, kernel_size), stride=(1, 2), padding=padding))
        model.add_module('del_conv_2', nn.Conv2d(
            30, 30, (1, kernel_size), stride=(1, 3), padding=padding))
        model.add_module('del_conv_3', nn.Conv2d(
            30, 10, (1, kernel_size), stride=(1, 3), padding=padding))

    # Compute the successive shapes of the input
    size = conv_output_length(input_shape[-1], kernel_size, padding, stride=2)
    size = conv_output_length(size, kernel_size, padding, stride=3)
    size = conv_output_length(size, kernel_size, padding, stride=3)

    # Flatten the samples
    if len(input_shape) == 3:   # Sequences of chunks
        # Exchange channel & sequence positions to allow channel flattening
        model.add_module('del_swapdims_1', SwapDims(1, 2))
    model.add_module('del_flatten', nn.Flatten(-2, -1))
    size *= 10                  # Times the nb of outputs of 'del_conv_3'

    # Add the dense layers
    if out_channels != 1:       # ND output space
        model.add_module('del_linear_1', nn.Linear(size, out_channels, lin_bias))
    else:                       # 1D output space
        model.add_module('del_linear_1', nn.Linear(size, 128, lin_bias))
        model.add_module('del_linear_2', nn.Linear(128, 64, lin_bias))
        model.add_module('del_linear_3', nn.Linear(64, 1, lin_bias))

    # Final activation function
    model.add_module('del_activation', nn.ReLU())

    # Reestablish the channels & sequences positions
    if len(input_shape) == 3:   # Sequences of chunks
        model.add_module('del_swapdims_2', SwapDims(1, 2))

    return model
#----------------------------------------------------------------------------#

##############################################################################



##############################################################################
##                       Denoising-Enhancement Layer                        ##
##############################################################################

class DELayer(torch.nn.Module):
    """ Torch-based Denoising-Enhancement Layer class

    Layer that is thought to be part of a larger model as an attention
    branch, i.e. a layer that operates in parallel of the main pipeline
    and whose outputs act on those of the main branch.

    In practice, take an input tensor and multiply it by a set of weigh-
    ting coefficients so as to reduce the background noise and increase
    the sounds of interest. To this end, take a Torch Model, use it to
    provide a set of outputs, and use them to weight the input tensor.
    A normalization (SoftMax) and a regularization (dropout) are added
    to the inner model to improve accuracy and generalization capability;
    also, give the possibility to fuse the channels of the outputs (i.e.
    the "refined inputs") to reduce the output space and save computation.

    When used in a wider model, the DELayer attention branch should be
    used in a sequential manner, since it takes a tensor and applies to
    it its weighting coefficients to refine the input, which results in
    a "refined inputs" that is returned as the DELayer outputs. As such,
    the training of the inner model is performed alongside that of the
    other layers of the overall model.

    Attributes
    ----------
    inner_model : Torch Sequential Model, getter only
        Attention branch inner model.
    sum_out_channels : bool, getter only
        If the channels of the output should be summed.
    dropout : float, getter only
        Dropout regularization rate, if any.
    nd_inner_model : bool, getter only
        If 1D (F) or ND (T) inner model output space.

    Constructor
    -----------
    __init__(inner_model, sum_out_channels=True, dropout=0.0, nd_inner_model=True)

    Methods
    -------
    forward(x)
        Apply the Denoising-Enhancement attention mechanism.

    Examples
    --------
    >>> import numpy as np
    >>> import torch
    >>> from aer.layers import Squeeze, SincLayer, DELayer
    >>> from aer.models_tk import torch_dataset, train_batch

    # Dataset parameters (chunks)
    >>> batch_size = 15         # Number of datasets
    >>> nb_frames = 10          # Number of signal frames (chunks)
    >>> len_frames = 400        # Number of samples in each chunk
    >>> nb_out_classes = 5      # Number of objective classes

    #----- Chunk variant (1D input data)
    # Shape of the input (NCW) -- Exactly 1 channel for SincLayer
    >>> input_shape = (batch_size, channels:=1, len_frames)
    # Shape of the output
    >>> output_shape = (batch_size, nb_out_classes)
    #-------------------------

    #----- Sequence variant (2D input data)
    # Shape of the input (NCHW) -- Exactly 1 channel for SincLayer
    >>> input_shape = (batch_size, channels:=1, nb_frames, len_frames)
    # Shape of the output
    >>> output_shape = (batch_size, nb_frames, nb_out_classes)
    #-------------------------

    # SincLayer parameters
    >>> nb_filters = 12         # Number of filters in the bank
    >>> filt_lgt = 251          # Length (in samples) of the filters
    >>> frate = 16000           # Sampling frequency

    # DELayer parameters
    >>> sum_channels = True     # If output channels should be summed
    >>> dropout = 0.1           # Dropout regularization rate, if any
    >>> nd_model = True         # If inner model output space is ND
    >>> del_shape = (nb_filters, *input_shape[2:])  # Inner model shape

    # Instantiate the model with SincLayer & DELayer
    # `Squeeze(1)` since `sum_channels` is True (thus 1-channel output)
    >>> model = torch.nn.Sequential(
    ...     SincLayer(nb_filters, filt_lgt, frate),
    ...     DELayer(del_shape, sum_channels, dropout, nd_model),
    ...     Squeeze(1),
    ...     torch.nn.Linear(len_frames, nb_out_classes))

    # Define the loss function & and model optimizer
    >>> loss_fct = torch.nn.CrossEntropyLoss()
    >>> optimizer = torch.optim.SGD(model.parameters())

    # Generate the data samples (NCW data format)
    >>> inputs = torch.arange(np.prod(input_shape), dtype=torch.float32)
    >>> inputs = inputs.reshape(input_shape)

    # Generate the objective data
    >>> outputs = torch.arange(np.prod(output_shape), dtype=torch.float32)
    >>> outputs = outputs.reshape(output_shape)

    # Wrap the inputs & outputs into a Torch dataset
    >>> dataset = torch_dataset(inputs, outputs, batch_size=1, shuffle=True)

    # Train the model
    >>> train_batch(dataset, model, loss_fct, optimizer)

    # Use the model for prediction
    >>> pred_out = model.forward(inputs)
    """

    #---------------------------   Constructor   ----------------------------#
    def __init__(self, inner_model, sum_out_channels=True,
                 dropout=0.0, nd_inner_model=True):
        """ Instantiate a DELayer object (constructor)

        Parameters
        ----------
        inner_model : Torch Model or tuple of 2-3 ints
            The attention branch model, as a Torch Sequential object;
            one can use the default model (that originally proposed by
            the DENet authors) by passing the unbatched shape of the
            input tensor (cf. the `attention_model` function).
        [OPT] sum_out_channels : bool
            If the channels of the inner model's outputs should be sum-
            med (True) or left separated (False). If so, the summation
            is done after the regularization and the input weighting.
            It has no effect with 1-channel input tensors.
                :Default: True (single channeled output)
        [OPT] dropout : float
            The dropout rate. If zero, no dropout is applied; else, a
            dropout regularization is done before weighting the input
            tensor components by the inner model's outputs.
                :Default: 0.0 (no dropout)
        [OPT] nd_inner_model : bool
            If the inner model output space is 1- or N-dimensional, with
            N the number of input channels. If True, the inner model is
            assumed to output N values; if False, it is supposed to out-
            put only one value. Only used to define the default inner
            model, i.e. if `inner_model` is a tuple of ints and not a
            Torch Sequential model.
                :Default: True

        Returns
        -------
        delayer : DELayer class object
            The correctly instantiated DELayer object.

        Examples
        --------
        >>> input_shape = (80, 3200)    # Format: CW

        # With the default inner model
        >>> delayer = DELayer(input_shape, True, 0.5, False)
        >>> delayer = DELayer(input_shape, False, 0.0, True)

        # With a 1-output model as input
        >>> inner_model = attention_model(input_shape, 1)
        >>> delayer = DELayer(inner_model, False, 0., False)

        # With a N-output model as input
        >>> out_channels = input_shape[0]
        >>> inner_model = attention_model(input_shape, out_channels)
        >>> delayer = DELayer(inner_model, False, 0., out_channels!=1)
        """

        super().__init__()

        # Layer parameters
        self._dropout = float(dropout)              # Dropout rate
        self._nd_inner_model = nd_inner_model       # If 1D or ND inner model

        # Number of outputs
        self._sum_channels = sum_out_channels       # Sum or not the outputs7

        # If no inner model provided, use the default one as attention model
        if isinstance(inner_model, torch.nn.Sequential):
            self._inner_model = inner_model
        else:
            # Dimension of the inner model output space (wrt `nd_inner_model`)
            channels = inner_model[0] if self._nd_inner_model else 1
            # Build the default inner attention model
            self._inner_model = attention_model(inner_model, channels)
    #------------------------------------------------------------------------#

    #----------------------------   Properties   ----------------------------#
    @property
    def sum_out_channels(self):
        """ Get the sum channels option """
        return self._sum_channels

    @property
    def dropout(self):
        """ Get the dropout rate """
        return self._dropout

    @property
    def nd_inner_model(self):
        """ Get if 1D (F) or ND (T) inner model output space """
        return self._nd_inner_model

    @property
    def inner_model(self):
        """ Get the attention branch inner model """
        return self._inner_model
    #------------------------------------------------------------------------#

    #--------------------------   Call the Layer   --------------------------#
    def forward(self, x):
        """ Apply the Denoising-Enhancement attention mechanism

        Take an input tensor `x`, pass it to the attention branch inner
        model and use its outputs in response to `x` as weights. Refine
        these weights by passing them into a SoftMax layer and apply a
        dropout regularization if the `dropout` attribute is non zero;
        use the refined coefficients to denoise and enhance the inputs
        `x` by multiplying them with each other and return the so-built
        tensor `x_refined`.

        If the `sum_out_channels` attribute is True, sum the channels of
        the refined tensor; leave them as they are else.

        Parameters
        ----------
        x : Torch tensor
            The input tensor with shape NC(H)W.

        Returns
        -------
        x_refined : Torch tensor
            The denoised-enhanced (input) tensor.

        Examples
        --------
        >>> import numpy as np

        # Dataset parameters (chunks)
        >>> nb_frames = 100             # Number of signal frames (chunks)
        >>> len_frames = 400            # Number of samples in each chunk

        # Shape of the input (NCW) -- More than 1 channel for DELayer
        >>> input_shape = (nb_frames, channels:=10, len_frames)

        # DELayer parameters
        >>> sum_channels = True         # If output channels should be summed
        >>> dropout = 0.1               # Dropout regularization rate, if any
        >>> nd_inn_model = True         # If inner model output space is ND

        # DELayer inner model (with unbatched input shape)
        >>> inn_model = attention_model(
        ...     input_shape[1:], input_shape[1] if nd_inn_model else 1)

        # Generate the data samples
        >>> data = torch.arange(np.prod(input_shape), dtype=torch.float32)
        >>> data = data.reshape(input_shape)

        # Instantiate build, and use the DELayer enhancer
        >>> delayer = DELayer(inn_model, sum_channels, dropout)
        >>> output = delayer.forward(data)

        # Check the output shape
        >>> output.shape
        torch.Size([100, 1, 400])
        """

        # Feed the inner model with the whole input tensor
        inn_mod_out = self._inner_model(x)

        # Normalize the weighting coefficients
#        weights = torch.softmax(inn_mod_out)    # SoftMax is slower and seems
        weights = inn_mod_out                    # less accurate

        # Perform a dropout regularization, if any
        if self._dropout != 0.0:
            weights = nn.functional.dropout(weights, self._dropout)

        # Apply the DE weighting coefficients to the input tensor `x`
        x_refined = torch.multiply(x, weights.unsqueeze(-1))

        # If `sum_out_channels`, sum the channels of the model outputs
        if self._sum_channels:
            x_refined = torch.sum(x_refined, axis=1, keepdims=True)

        return x_refined
    #------------------------------------------------------------------------#

##############################################################################
