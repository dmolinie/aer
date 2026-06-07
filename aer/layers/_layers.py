""" Miscellaneous Torch-based layers

Authors: Dylan MOLINIE
Company: CEA Paris-Saclay, Gif-sur-Yvette, France
Contact: dylan.molinie@cea.fr
Date: March 2025
Last revised: March 2026

License: GPLv3
"""
# pylint: disable=too-few-public-methods

__all__ = [
    'conv_output_length',
    'Reshape', 'SwapDims', 'Squeeze', 'Unsqueeze',
    'PrintShape', 'ExtractTensor', 'LayerNorm',]

import torch


##############################################################################
##                         Miscellaneous Functions                          ##
##############################################################################

#----------------------   Convolution Output Length   -----------------------#
def conv_output_length(inp_length, kernel_size, padding, stride=1):
    """ Compute the output length of a convolution given an input length

    Take the length (in samples) of an input vector/tensor, the size of
    the kernel (filter) & the convolution params (padding, stride), and
    compute the length (in samples) of the output of the convolution.

    Parameters
    ----------
    inp_length : int
        The length of the input.
    kernel_size : int
        The size of the convolution kernel (filter).
    padding : str
        The convolution padding; should be one among:
            {'same', 'valid', 'full', 'causal'}.
    [OPT] stride : int
        The convolution stride (~kernel window step).
            :Default: 1

    Returns
    -------
    out_length
        The output length (integer).

    Examples
    --------
    >>> import numpy as np
    >>> data = np.arange(100)
    >>> filter = np.arange(50)
    >>> conv_output_length(len(data), len(filter), 'same', 1)   # 100
    >>> conv_output_length(len(data), len(filter), 'full', 1)   # 149
    >>> conv_output_length(len(data), len(filter), 'same', 3)   # 34
    >>> conv_output_length(len(data), len(filter), 'full', 3)   # 50
    """

    # Check the 'padding' type
    if padding in ('same', 'causal'):
        out_length = inp_length
    elif padding == 'full':
        out_length = inp_length + kernel_size - 1
    else:    # 'valid'
        out_length = inp_length - kernel_size + 1

    # Compute the length of the convolution output
    return (out_length + stride - 1) // stride
#----------------------------------------------------------------------------#

##############################################################################



##############################################################################
##                          Tensor Methods Wrapper                          ##
##############################################################################

#--------------------------   Print Tensor Shape   --------------------------#
class Reshape(torch.nn.Module):
    """ Reshape a tensor on the flow, and return the it

    Attributes
    ----------
    new_shape : array_like of ints
        The tensor's new shape.
    + Those inherited from `Torch.nn.Module`.

    Constructor
    -----------
    __init__(new_shape)

    Methods
    -------
    forward(x)
        Reshape the tensor.
    + Those inherited from `Torch.nn.Module`.

    Examples
    --------
    >>> import torch

    # Generate dummy data
    >>> inputs = torch.rand(10, 5, 100)

    # Build the layer
    >>> model = Reshape((10, 5, 10, 10))

    # Reshape the tensor on the flow
    >>> outputs = model.forward(inputs)

    # Check the input & output tensors shapes
    >>> print('Inputs:', inputs.shape, '-- Outputs:', outputs.shape)
    Inputs: torch.Size([10, 5, 100]) -- Outputs: torch.Size([10, 5, 10, 10])
    """

    def __init__(self, new_shape):
        """ Instantiate a Reshape object (constructor)

        Initialize the layer and store the tensor new shape as an attribute.

        Parameters
        ----------
        new_shape : array_like of ints
            The new shape for any input tensor (would be NC(H)W format).
        """
        super().__init__()
        self.shape = new_shape

    def forward(self, x):
        """ Reshape an input tensor `x` and return it

        Parameters
        ----------
        x : Torch tensor (NC(H)W format)
            The on-the-flow tensor to reshape.

        Returns
        -------
        x : Torch tensor
            The reshaped tensor.
        """
        return x.reshape(self.shape)
#----------------------------------------------------------------------------#

#-------------------------   Dimension Swap Layer   -------------------------#
class SwapDims(torch.nn.Module):
    """ Swap two dimensions of a tensor on the flow

    Attributes
    ----------
    dim0 : int
        The first dimension to swap.
    dim1 : int
        The second dimension to swap.
    + Those inherited from `Torch.nn.Module`.

    Constructor
    -----------
    __init__(dim0, dim1)

    Methods
    -------
    forward(x)
        Apply on-the-flow dimension swapping on the tensor.
    + Those inherited from `Torch.nn.Module`.

    Examples
    --------
    >>> import torch

    # Build the layer
    >>> model = SwapDims(1, 2)

    # Create some testing data and pass them to the model
    >>> inputs = torch.rand(10, 5, 100)
    >>> outputs = model.forward(inputs)

    # Compare the inputs & outputs shapes
    >>> print('Inputs:', inputs.shape, '-- Outputs:', outputs.shape)
    Inputs: torch.Size([10, 5, 100]) -- Outputs: torch.Size([10, 100, 5])
    """

    def __init__(self, dim0, dim1):
        """ Instantiate a SwapDims object (constructor)

        Parameters
        ----------
        dim0 : int
            The first dimension to swap.
        dim1 : int
            The second dimension to swap.
        """
        super().__init__()
        self.dim0 = dim0
        self.dim1 = dim1

    def forward(self, x):
        """ Swap the tensor's two dimensions specified in constructor

        Parameters
        ----------
        x : Torch tensor (NC(H)W format)
            The on-the-flow tensor for which to swap dimensions.

        Returns
        -------
        x_swap : Torch tensor
            The tensor with swapped dimensions.
        """
        return x.swapdims(self.dim0, self.dim1)
#----------------------------------------------------------------------------#

#----------------------------   Squeeze Tensor   ----------------------------#
class Squeeze(torch.nn.Module):
    """ Squeeze (remove 1-item axis) an on-the-flow tensor along a dimension

    Attributes
    ----------
    dim : int
        The dimension to squeeze.
    + Those inherited from `Torch.nn.Module`.

    Constructor
    -----------
    __init__(*dim)

    Methods
    -------
    forward(x)
        Squeeze the tensor.
    + Those inherited from `Torch.nn.Module`.

    Examples
    --------
    >>> import torch

    # Build the layer
    >>> model = Squeeze(1)

    # Create some testing data and pass them to the model
    >>> inputs = torch.rand(10, 1, 100)
    >>> outputs = model.forward(inputs)

    # Compare the inputs & outputs shapes
    >>> print('Inputs:', inputs.shape, '-- Outputs:', outputs.shape)
    Inputs: torch.Size([10, 1, 100]) -- Outputs: torch.Size([10, 100])
    """
    def __init__(self, *dim):
        """ Instantiate a Squeeze object (constructor)

        Parameters
        ----------
        [OPT] dim : int
            The dimension to squeeze. If omitted, apply squeezing to any
            dimensions.
                :Default: None
        """
        super().__init__()
        self.dim = dim

    def forward(self, x):
        """ Squeeze the tensor

        Parameters
        ----------
        x : Torch tensor (NC(H)W format)
            The on-the-flow tensor to squeeze.

        Returns
        -------
        x_sqz : Torch tensor
            The squeezed input tensor.
        """
        return x.squeeze(self.dim)
#----------------------------------------------------------------------------#

#---------------------------   Unsqueeze Tensor   ---------------------------#
class Unsqueeze(torch.nn.Module):
    """ Unsqueeze an on-the-flow tensor along a dimension

    Attributes
    ----------
    dim : int
        The dimension to unsqueeze.
    + Those inherited from `Torch.nn.Module`.

    Constructor
    -----------
    __init__(dim)

    Methods
    -------
    forward(x)
        Unsqueeze the tensor.
    + Those inherited from `Torch.nn.Module`.

    Examples
    --------
    >>> import torch

    # Build the layer
    >>> model = Unsqueeze(1)

    # Create some testing data and pass them to the model
    >>> inputs = torch.rand(10, 5, 100)
    >>> outputs = model.forward(inputs)

    # Compare the inputs & outputs shapes
    >>> print('Inputs:', inputs.shape, '-- Outputs:', outputs.shape)
    Inputs: torch.Size([10, 5, 100]) -- Outputs: torch.Size([10, 1, 5, 100])
    """
    def __init__(self, dim):
        """ Instantiate an Unsqueeze object (constructor)

        Parameters
        ----------
        dim : int
            The dimension to unsqueeze.
        """
        super().__init__()
        self.dim = dim

    def forward(self, x):
        """ Unsqueeze the tensor

        Parameters
        ----------
        x : Torch tensor (NC(H)W format)
            The on-the-flow tensor to unsqueeze.

        Returns
        -------
        x_unsqz : Torch tensor
            The unsqueezed input tensor.
        """
        return x.unsqueeze(self.dim)
#----------------------------------------------------------------------------#

##############################################################################



##############################################################################
##                           Miscellaneous Layers                           ##
##############################################################################

#--------------------------   Print Tensor Shape   --------------------------#
class PrintShape(torch.nn.Module):
    """ Print the shape of a tensor on the flow, and return the tensor;
    designed to be used for testing and debugging

    Attributes
    ----------
    None (except those inherited from `Torch.nn.Module`)

    Constructor
    -----------
    __init__()

    Methods
    -------
    forward(x)
        Print the shape of the tensor, and return it.
    + Those inherited from `Torch.nn.Module`.

    Examples
    --------
    >>> import torch

    # Build the layer
    >>> model = PrintShape()
    >>> inputs = torch.rand(10, 5, 100)

    # Print the shape of the tensor on the flow
    >>> outputs = model.forward(inputs)
    Tensor shape: torch.Size([10, 5, 100])

    # Check that the input tensor is returned by the layer
    >>> torch.all(inputs == outputs)
    tensor(True)
    """

    def forward(self, x):
        """ Print the shape of the Tensor and return the tensor itself

        Parameters
        ----------
        x : Torch tensor (NC(H)W format)
            The on-the-flow tensor for which to print the shape.

        Returns
        -------
        x : Torch tensor
            The input tensor (to continue the flow).
        """
        print('Tensor shape:', x.shape)
        return x
#----------------------------------------------------------------------------#

#--------------------------   Extract Sub-Tensor   --------------------------#
class ExtractTensor(torch.nn.Module):
    """ Extract a sub-tensor from an on-the-flow tensor

    N.B.: recurrent cells (GRU, LSTM, etc.) return a 2-tuple, organized
        as: (output, (h_n, c_n)), where:
          - output is the output features from the last layer
          - h_n is the final hidden state for each element in the sequence
          - c_n is final cell state for each element in the sequence
        As such, to retrieve the output tensor from a recurrent cell,
        one should extract the first element of the tuple returned by it;
        this can be done with `ExtractTensor(0)`.

    Attributes
    ----------
    axes : int, slice or list/tuple of ints or slices
        The axes to slice from the input tensor.

    Constructor
    -----------
    __init__(*axes)

    Methods
    -------
    forward(x)
        Extract the sub-tensor.
    + Those inherited from `Torch.nn.Module`.

    Examples
    --------
    >>> import torch

    # Build the layer
    >>> model = ExtractTensor(slice(None), -1)

    # Create some testing data and pass them to the model
    >>> inputs = torch.rand(10, 5, 100)
    >>> outputs = model.forward(inputs)

    # Compare the inputs & outputs shapes
    >>> print('Inputs:', inputs.shape, '-- Outputs:', outputs.shape)
    Inputs: torch.Size([10, 5, 100]) -- Outputs: torch.Size([10, 100])
    """

    def __init__(self, *axes):
        """ Instantiate a ExtractTensor object (constructor)

        Parameters
        ----------
        axes : int, slice or list/tuple of ints or slices
            The input tensor axes to slice; use `slice` built-in object
            to slice the tensor; use `slice(None)` to slice a full axis
            (equivalent to `:`).
        """
        super().__init__()
        self.axes = axes

    def forward(self, x):
        """ Extract the sub-tensor by extracting the axes specified in
        the constructor from input `x`

        Parameters
        ----------
        x : Torch tensor (NC(H)W format)
            The on-the-flow tensor for which to extract the sub-tensor.

        Returns
        -------
        x_sub : Torch tensor
            The sub-tensor, as `x[self.axis]`.
        """
        # If `axes` is a direct int or slice
        if isinstance(self.axes, (int, slice)):
            return x[self.axes]
        # If `axes` is an 1-int/slice iterable
        if len(self.axes) == 1:
            return x[self.axes[0]]
        # If `axes` has several ints/slices
        return x[*self.axes]
#----------------------------------------------------------------------------#

#--------------------   Trainable Layer Normalization   ---------------------#
class LayerNorm(torch.nn.Module):
    """ Torch Layer Normalization (LayerNorm)

    The normalization in Recurrent Neural Networks as implemented below
    was proposed by J. Ba, J. Kiros, G. Hinton in "Layer Normalization",
    and accessible at https://arxiv.org/abs/1607.06450.

    Provide a Torch-based normalization layer; for an input tensor `x`,
    by referring to `m` and `s` as its mean and standard deviation, the
    normalization operation is defined as:
        x_norm = g * (x - m) / (s + eps) + b
    where `g` is the normalization scale and `b` is the bias, which are
    trainable weights that are optimized during the training process;
    `eps` is a std regularization term that avoids dividing by 0.

    The normalization is performed over the channels, i.e. the mean and
    standard deviation are computed along the channel dimension.

    Attributes
    ----------
    normalized_shape : int or list of ints
        The tensor's dimensions to normalize.
    eps : float
        The standard deviation regularization term.
    gamma : torch (trainable) Parameters
        Trainable channel scaling weights.
    beta : torch (trainable) Parameters
        Trainable channel normalization biases.
    + Those inherited from `Torch.nn.Module`.

    Constructor
    -----------
    __init__(normalized_shape, eps=1e-6)

    Methods
    -------
    forward(x)
        Apply normalization to the input data.
    + Those inherited from `Torch.nn.Module`.

    Examples
    --------
    >>> import torch

    # Build the layer
    >>> input_shape = (10, 5, 100)
    >>> model = LayerNorm(input_shape, 1e-6)

    # Create some testing data and pass them to the model
    >>> inputs = torch.rand(input_shape)
    >>> outputs = model.forward(inputs)

    # Compare the inputs & outputs shapes
    >>> print('Inputs:', inputs.mean(), '-- Outputs:', outputs.mean())
    Inputs: tensor(0.4983) -- Outputs: tensor(2.3651e-08)
    """

    def __init__(self, normalized_shape, eps=1e-5):
        """ Instantiate a LayerNorm object (constructor)

        Initialize the `gamma` scaling term to 1 and the `beta` bias
        term to 0, and add these terms to the trainable parameters.
        Both are the same length as the tensor to normalize.

        Parameters
        ----------
        normalized_shape : int or list or torch.Size
            (torch naming) The axes to normalize.
        [OPT] eps : float
            The standard deviation regularization term.
                :Default: 1e-5
        """
        super().__init__()
        self.eps = eps
        self.normalized_shape = normalized_shape
        self.gamma = torch.nn.Parameter(torch.ones(normalized_shape))
        self.beta = torch.nn.Parameter(torch.zeros(normalized_shape))
        self.dims = []

    def _get_dims(self, input_shape):
        """ Get the axes of the `normalized_shape` dimensions """
        self.dims = [len(input_shape)-i for i, (dim_x, shape)
                     in enumerate(zip(input_shape, self.normalized_shape))
                     if dim_x == shape].reverse()

    def forward(self, x):
        """ Normalize the input tensor on the flow

        Parameters
        ----------
        x : Torch tensor (NC(H)W format)
            The on-the-flow tensor to normalize.

        Returns
        -------
        x_norm : Torch tensor
            The input tensor normalized along the axes.
        """
        if len(self.dims) == 0:
            self._get_dims(x.shape)
        mean = x.mean(self.dims, keepdim=True)
        var = x.var(self.dims, keepdim=True, correction=0)
        return self.gamma * (x - mean) / torch.sqrt(var + self.eps) + self.beta
#----------------------------------------------------------------------------#

###############################################################################
