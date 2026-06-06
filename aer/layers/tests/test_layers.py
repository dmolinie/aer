import torch
from aer.layers._layers import *


def test_conv_output_length():
    """ Compute the output length of a convolution given an input length """
    import numpy as np
    data = np.arange(100)
    filter = np.arange(50)
    print(conv_output_length(len(data), len(filter), 'same', 1))
    print(conv_output_length(len(data), len(filter), 'full', 1))
    print(conv_output_length(len(data), len(filter), 'same', 3))
    print(conv_output_length(len(data), len(filter), 'full', 3))

def test_Reshape():
    """ Reshape a tensor on the flow, and return the it """
    # Generate dummy data
    inputs = torch.rand(10, 5, 100)
    # Build the layer
    model = Reshape((10, 5, 10, 10))
    # Reshape the tensor on the flow
    outputs = model.forward(inputs)
    # Check the input & output tensors shapes
    print('Inputs:', inputs.shape, '-- Outputs:', outputs.shape)

def test_SwapDims():
    """ Swap two dimensions of a tensor on the flow """
    # Build the layer
    model = SwapDims(1, 2)
    # Create some testing data and pass them to the model
    inputs = torch.rand(10, 5, 100)
    outputs = model.forward(inputs)
    # Compare the inputs & outputs shapes
    print('Inputs:', inputs.shape, '-- Outputs:', outputs.shape)

def test_Squeeze():
    """ Squeeze (remove 1-item axis) an on-the-flow tensor along a dimension """
    # Build the layer
    model = Squeeze(1)
    # Create some testing data and pass them to the model
    inputs = torch.rand(10, 1, 100)
    outputs = model.forward(inputs)
    # Compare the inputs & outputs shapes
    print('Inputs:', inputs.shape, '-- Outputs:', outputs.shape)

def test_Unsqueeze():
    """ Unsqueeze an on-the-flow tensor along a dimension """
    # Build the layer
    model = Unsqueeze(1)
    # Create some testing data and pass them to the model
    inputs = torch.rand(10, 5, 100)
    outputs = model.forward(inputs)
    # Compare the inputs & outputs shapes
    print('Inputs:', inputs.shape, '-- Outputs:', outputs.shape)

def test_PrintShape():
    """ Print the shape of a tensor on the flow, and return the tensor;
    designed to be used for testing and debugging """
    # Build the layer
    model = PrintShape()
    inputs = torch.rand(10, 5, 100)
    # Print the shape of the tensor on the flow
    outputs = model.forward(inputs)
    # Check that the input tensor is returned by the layer
    print(torch.all(inputs == outputs))

def test_ExtractTensor():
    """ Extract a sub-tensor from an on-the-flow tensor """
    # Build the layer
    model = ExtractTensor(slice(None), -1)
    # Create some testing data and pass them to the model
    inputs = torch.rand(10, 5, 100)
    outputs = model.forward(inputs)
    # Compare the inputs & outputs shapes
    print('Inputs:', inputs.shape, '-- Outputs:', outputs.shape)

def test_LayerNorm():
    """ Torch Layer Normalization (LayerNorm) """
    # Build the layer
    input_shape = (10, 5, 100)
    model = LayerNorm(input_shape, 1e-6)
    # Create some testing data and pass them to the model
    inputs = torch.rand(input_shape)
    outputs = model.forward(inputs)
    # Compare the inputs & outputs shapes
    print('Inputs:', inputs.mean(), '-- Outputs:', outputs.mean())



# Launch test/example functions
test_conv_output_length()

test_Reshape()

test_SwapDims()

test_Squeeze()

test_Unsqueeze()

test_PrintShape()

test_ExtractTensor()

test_LayerNorm()

