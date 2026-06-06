import numpy as np
import torch
from aer.layers._sinclayer import *


def test_SincLayer_chks():
    """ Sinc Filterbank Layer class """
    from aer.models_tk import torch_dataset, train_batch

    # Dataset parameters (chunks)
    batch_size = 15         # Number of datasets
    nb_frames = 10          # Number of signal frames (chunks)
    len_frames = 400        # Number of samples in each chunk
    nb_out_classes = 5      # Number of objective classes

    # SincLayer parameters
    nb_filters = 12         # Number of filters in the bank
    filt_lgt = 251          # Length (in samples) of the filters
    frate = 16000           # Sampling frequency

    # Shape of the input (NCW) -- Exactly 1 channel for SincLayer
    input_shape = (batch_size, channels:=1, len_frames)
    # Shape of the output
    output_shape = (batch_size, nb_out_classes)

    # Instantiate the model with SincLayer
    model = torch.nn.Sequential(
        SincLayer(nb_filters, filt_lgt, frate),
        torch.nn.Flatten(),
        torch.nn.Linear(nb_filters*len_frames, nb_out_classes))

    # Define the loss function & and model optimizer
    loss_fct = torch.nn.CrossEntropyLoss()
    optimizer = torch.optim.SGD(model.parameters())

    # Generate the data samples (NCW data format)
    inputs = torch.arange(np.prod(input_shape), dtype=torch.float32)
    inputs = inputs.reshape(input_shape)

    # Generate the objective data
    outputs = torch.arange(np.prod(output_shape), dtype=torch.float32)
    outputs = outputs.reshape(output_shape)

    # Wrap the inputs & outputs into a Torch dataset
    dataset = torch_dataset(inputs, outputs, batch_size=1, shuffle=True)

    # Train the model
    train_batch(dataset, model, loss_fct, optimizer)

    # Use the model for prediction
    pred_out = model.forward(inputs)

def test_SincLayer_seqs():
    """ Sinc Filterbank Layer class """
    from aer.models_tk import torch_dataset, train_batch
    from aer.layers import SwapDims

    # Dataset parameters (chunks)
    batch_size = 15         # Number of datasets
    nb_frames = 10          # Number of signal frames (chunks)
    len_frames = 400        # Number of samples in each chunk
    nb_out_classes = 5      # Number of objective classes

    # SincLayer parameters
    nb_filters = 12         # Number of filters in the bank
    filt_lgt = 251          # Length (in samples) of the filters
    frate = 16000           # Sampling frequency

    # Shape of the input (NCHW) -- Exactly 1 channel for SincLayer
    input_shape = (batch_size, channels:=1, nb_frames, len_frames)
    # Shape of the output
    output_shape = (batch_size, nb_frames, nb_out_classes)

    # Instantiate the model with SincLayer
    model = torch.nn.Sequential(
        SincLayer(nb_filters, filt_lgt, frate),
        SwapDims(1, 2),
        torch.nn.Flatten(2, 3),
        torch.nn.Linear(nb_filters*len_frames, nb_out_classes))

    # Define the loss function & and model optimizer
    loss_fct = torch.nn.CrossEntropyLoss()
    optimizer = torch.optim.SGD(model.parameters())

    # Generate the data samples (NCW data format)
    inputs = torch.arange(np.prod(input_shape), dtype=torch.float32)
    inputs = inputs.reshape(input_shape)

    # Generate the objective data
    outputs = torch.arange(np.prod(output_shape), dtype=torch.float32)
    outputs = outputs.reshape(output_shape)

    # Wrap the inputs & outputs into a Torch dataset
    dataset = torch_dataset(inputs, outputs, batch_size=1, shuffle=True)

    # Train the model
    train_batch(dataset, model, loss_fct, optimizer)

    # Use the model for prediction
    pred_out = model.forward(inputs)

def test_SincLayer_init():
    """ Instantiate a SincLayer object (constructor) and add the
    cutoff frequencies to its trainable parameters """
    sinclayer = SincLayer(10, 251, 16000)
    sinclayer = SincLayer(10, 251, 16000, (0., 100.))

def test_SincLayer_rescale():
    """ Rescale the frequencies provided as argument """
    nb_frames, len_frames = 100, 400                    # Input params
    nb_filters, filt_lgt, frate = 10, 251, 16000        # Layer params
    input_shape = (nb_frames, channels:=1, len_frames)  # Input shape
    sinclayer = SincLayer(nb_filters, filt_lgt, frate)
    low_co = sinclayer.rescale(sinclayer.low_co)
    upp_co = sinclayer.rescale(sinclayer.upp_co)

def test_SincLayer_forward():
    """ Apply filtering to an input signal """
    # Dataset parameters (chunks)
    nb_frames = 100             # Number of signal frames (chunks)
    len_frames = 400            # Number of samples in each chunk

    # Shape of the input (NCW) -- Exactly 1 channel for SincLayer
    inp_shape = (nb_frames, channels:=1, len_frames)

    # SincLayer parameters
    nb_filters = 10             # Number of filters in the bank
    filt_lgt = 251              # Length (in samples) of the filters
    frate = 16000               # Sampling frequency

    # Generate the data samples (NCW data format)
    inputs = torch.arange(np.prod(inp_shape), dtype=torch.float32)
    inputs = inputs.reshape(inp_shape)

    # Instantiate and use the SincLayer filter
    sinclayer = SincLayer(nb_filters, filt_lgt, frate).to('cpu')
    outputs = sinclayer.forward(inputs)

    #----- Check the effect of the padding strategy
    # Default padding strategy ('same')
    sinclayer.padding = 'same'
    print(sinclayer.forward(inputs).shape)

    # Change the padding strategy ('same' as default)
    sinclayer.padding = 'valid'
    print(sinclayer.forward(inputs).shape)



# Launch test/example functions
test_SincLayer_chks()

test_SincLayer_seqs()

test_SincLayer_init()

test_SincLayer_rescale()

test_SincLayer_forward()

