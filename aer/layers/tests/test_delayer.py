import numpy as np
import torch
from aer.layers._layers import Squeeze
from aer.layers._sinclayer import *
from aer.layers._delayer import *


def test_attention_model():
    """ Attention model """
    from torchinfo import summary
    # Model with only one output channel
    input_shape = (80, 3200)
    inner_model = attention_model(input_shape, 1)
    # Model with the same number of outputs channels than inputs'
    inner_model = attention_model(input_shape, input_shape[0])
    summary(inner_model)

def test_DELayer_chks():
    """ Torch-based Denoising-Enhancement Layer class """
    from aer.models_tk import torch_dataset, train_batch

    # Dataset parameters (chunks)
    batch_size = 15         # Number of datasets
    nb_frames = 10          # Number of signal frames (chunks)
    len_frames = 400        # Number of samples in each chunk
    nb_out_classes = 5      # Number of objective classes

    # Shape of the input (NCW) -- Exactly 1 channel for SincLayer
    input_shape = (batch_size, channels:=1, len_frames)
    # Shape of the output
    output_shape = (batch_size, nb_out_classes)

    # SincLayer parameters
    nb_filters = 12         # Number of filters in the bank
    filt_lgt = 251          # Length (in samples) of the filters
    frate = 16000           # Sampling frequency

    # DELayer parameters
    sum_channels = True     # If output channels should be summed
    dropout = 0.1           # Dropout regularization rate, if any
    nd_model = True         # If inner model output space is ND
    del_shape = (nb_filters, *input_shape[2:])  # Inner model shape

    # Instantiate the model with SincLayer & DELayer
    # `Squeeze(1)` since `sum_channels` is True (thus 1-channel output)
    model = torch.nn.Sequential(
        SincLayer(nb_filters, filt_lgt, frate),
        DELayer(del_shape, sum_channels, dropout, nd_model),
        Squeeze(1),
        torch.nn.Linear(len_frames, nb_out_classes))

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

def test_DELayer_seqs():
    """ Torch-based Denoising-Enhancement Layer class """
    from aer.models_tk import torch_dataset, train_batch

    # Dataset parameters (chunks)
    batch_size = 15         # Number of datasets
    nb_frames = 10          # Number of signal frames (chunks)
    len_frames = 400        # Number of samples in each chunk
    nb_out_classes = 5      # Number of objective classes

    # Shape of the input (NCHW) -- Exactly 1 channel for SincLayer
    input_shape = (batch_size, channels:=1, nb_frames, len_frames)
    # Shape of the output
    output_shape = (batch_size, nb_frames, nb_out_classes)

    # SincLayer parameters
    nb_filters = 12         # Number of filters in the bank
    filt_lgt = 251          # Length (in samples) of the filters
    frate = 16000           # Sampling frequency

    # DELayer parameters
    sum_channels = True     # If output channels should be summed
    dropout = 0.1           # Dropout regularization rate, if any
    nd_model = True         # If inner model output space is ND
    del_shape = (nb_filters, *input_shape[2:])  # Inner model shape

    # Instantiate the model with SincLayer & DELayer
    # `Squeeze(1)` since `sum_channels` is True (thus 1-channel output)
    model = torch.nn.Sequential(
        SincLayer(nb_filters, filt_lgt, frate),
        DELayer(del_shape, sum_channels, dropout, nd_model),
        Squeeze(1),
        torch.nn.Linear(len_frames, nb_out_classes))

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

def test_DELayer_init():
    """ Instantiate a DELayer object (constructor) """
    input_shape = (80, 3200)    # Format: CW
    # With the default inner model
    delayer = DELayer(input_shape, True, 0.5, False)
    delayer = DELayer(input_shape, False, 0.0, True)
    # With a 1-output model as input
    inner_model = attention_model(input_shape, 1)
    delayer = DELayer(inner_model, False, 0., False)
    # With a N-output model as input
    out_channels = input_shape[0]
    inner_model = attention_model(input_shape, out_channels)
    delayer = DELayer(inner_model, False, 0., out_channels!=1)

def test_DELayer_forward():
    """ Apply the Denoising-Enhancement attention mechanism """
    # Dataset parameters (chunks)
    nb_frames = 100             # Number of signal frames (chunks)
    len_frames = 400            # Number of samples in each chunk
    # Shape of the input (NCW) -- More than 1 channel for DELayer
    input_shape = (nb_frames, channels:=10, len_frames)
    # DELayer parameters
    sum_channels = True         # If output channels should be summed
    dropout = 0.1               # Dropout regularization rate, if any
    nd_inn_model = True         # If inner model output space is ND
    # DELayer inner model (with unbatched input shape)
    inn_model = attention_model(
        input_shape[1:], input_shape[1] if nd_inn_model else 1)
    # Generate the data samples
    data = torch.arange(np.prod(input_shape), dtype=torch.float32)
    data = data.reshape(input_shape)
    # Instantiate build, and use the DELayer enhancer
    delayer = DELayer(inn_model, sum_channels, dropout)
    output = delayer.forward(data)
    # Check the output shape
    print(output.shape)



# Launch test/example functions
test_attention_model()

test_DELayer_chks()

test_DELayer_seqs()

test_DELayer_init()

test_DELayer_forward()

