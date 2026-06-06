import numpy as np
import torch
import aer.datasets
import aer.models_tk as mtk
from aer.datasets.mivia_loader import build_dataset
from aer.models._sincnet import *
from aer.models._denet import *


def overall_sincnet():
    """ Didactic example of SincNet use """

    # Dataset parameters
    ROOT = f"{aer.datasets.__path__[0]}/MIVIA_AE_example/training/"
    XMLFOLDER = ROOT
    SNDFOLDER = ROOT + "sounds/"

    # Extension of the audio file -- Read file "00066_3.wav"
    IND, SNR = 66, 15

    # Chunk parameters
    frm_duration = 100e-3               # Frame duration in seconds
    hop_duration = 50e-3                # Hop duration in seconds
    nb_classes = 4                      # Number of possible classes

    # Chunks settings
    chk_params = (frm_duration, hop_duration)

    # Read the audio file and build the chunks of data & classes
    specs, data, classes = build_dataset(
        (XMLFOLDER, SNDFOLDER), (IND, SNR), nb_classes, chk_params)

    # Turn the data & classes into Torch Tensor
    data = torch.Tensor(data).unsqueeze(1)  # NCW format, with C=1
    classes = torch.Tensor(classes)

    # Wrap the input data & the output classes into a Torch DataLoader
    dataset = mtk.torch_dataset(
        data, classes, batch_size=4, shuffle=True, drop_last=True)

    #--- Train & use the network

    # Retrieve the data parameters
    # nb_classes = classes.shape[-1]    # Number of classes
    # nb_frames = data.shape[0]         # Nb of frames (chunks)
    # len_frames = data.shape[1]        # Nb of samples per chunk
    input_shape = data.shape[1:]        # Unbatched data format: CW

    # Convolutional layer parameters
    conv_params = {'out_channels': 60, 'kernel_size': 5,
                   'stride': 1, 'padding': 'valid'}

    # SincLayer parameters
    scl_params = {'nb_filters': 60, 'filter_length': 251, 'frate': specs[0],
                  'bandwidth': (50., specs[0]/2), 'padding': 'same'}

    # Build the model
    model = SincNet(input_shape, nb_classes,
                    scl_params=scl_params, reg_scl=(3, 0.1),
                    conv_params=(2, conv_params), reg_conv=(3, 0.1),
                    nb_neurons_fc=[2048, 1024, 512], reg_linear=0.3,
                    rec_cell='')

    # Define the loss function
    loss_fct = torch.nn.CrossEntropyLoss()

    # Define the model optimizer
    optimizer = torch.optim.SGD(model.parameters())

    # The the model
    loss = mtk.train_batch(dataset, model, loss_fct, optimizer)

    # Pass the model in evaluation mode (no more training)
    model.eval()

    # Predict only one chunk
    y = model.forward(data[0])

    # Predict several chunks at once
    y = model.forward(data[-10:])


def overall_denet():
    """ Didactic example of DENet use """

    # Dataset parameters
    ROOT = f"{aer.datasets.__path__[0]}/MIVIA_AE_example/training/"
    XMLFOLDER = ROOT
    SNDFOLDER = ROOT + "sounds/"

    # Extension of the audio file -- Read file "00066_3.wav"
    IND, SNR = 66, 15

    # Chunk parameters
    frm_duration = 50e-3                # Frame duration in seconds
    nb_classes = 4                      # Number of possible classes

    # Chunks & Sequences settings
    chk_params = (frm_duration, frm_duration)
    seq_params = (10, 1)

    # Read the audio file and build the chunks of data & classes
    specs, data, classes = build_dataset(
        (XMLFOLDER, SNDFOLDER), (IND, SNR), nb_classes, chk_params, seq_params)

    # Turn the data & classes into Torch Tensor
    data = torch.Tensor(data).unsqueeze(1)  # NCHW format, with C=1 here
    classes = torch.Tensor(classes)

    # Wrap the input data & the output classes into a Torch DataLoader
    dataset = mtk.torch_dataset(
        data, classes, batch_size=4, shuffle=True, drop_last=True)

    #--- Train & use the network

    # Retrieve the data parameters
    # nb_classes = classes.shape[-1]    # Number of classes
    # nb_frames = data.shape[2]         # Nb of frames (chunks)
    # len_frames = data.shape[-1]       # Nb of samples per chunk
    input_shape = data.shape[1:]        # Unbatched data format: CHW

    # Convolutional layer parameters
    conv_params = {'out_channels': 60, 'kernel_size': 5,
                   'stride': 1, 'padding': 'valid'}

    # SincLayer parameters
    scl_params = {'nb_filters': 60, 'filter_length': 251, 'frate': specs[0],
                  'bandwidth': (50., specs[0]/2), 'padding': 'same'}

    # DELayer parameters
    del_params = {'sum_out_channels': False, 'dropout': 0.3,
                  'nd_inner_model': False}

    # Build the model
    model = DENet(input_shape, nb_classes,
                  scl_params=scl_params, reg_scl=(3, 0.1),
                  del_params=del_params, reg_del=(3, 0.1),
                  conv_params=(2, conv_params), reg_conv=(3, 0.1),
                  nb_neurons_fc=[2048, 1024, 512], reg_linear=0.3,
                  rec_cell='gru', nb_neurons_rec=1024)

    # Define the loss function
    loss_fct = torch.nn.CrossEntropyLoss()

    # Define the model optimizer
    optimizer = torch.optim.SGD(model.parameters())

    # The the model
    loss = mtk.train_batch(dataset, model, loss_fct, optimizer)

    # Pass the model in evaluation mode (no more training)
    model.eval()

    # Predict only one chunk
    y = model.forward(data[0])

    # Predict several chunks at once
    y = model.forward(data[-10:])



# Launch test/example functions
overall_sincnet()

overall_denet()

