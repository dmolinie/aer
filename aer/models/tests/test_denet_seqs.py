import numpy as np
import torch
from torchinfo import summary
from aer.models._denet import *


def test_DENet_no_conv_no_fc_no_rec_seqs():
    """ Default parameters -- sequences """
    #--- Sequences (3D) variant -- Format: CHW, with C=1
    batch = 5
    input_shape = (1, 10, 1600)
    # Generate testing data
    data = torch.arange(batch*np.prod(input_shape), dtype=torch.float32)
    data = data.reshape(batch, *input_shape)
    # Instantiate the model
    # No layers
    model = DENet(input_shape, nb_classes=4,
                  conv_params=[], nb_neurons_fc=[], rec_cell='')
    summary(model, input_size=input_shape)
    # Only conv layers (default used)
    model = DENet(input_shape, nb_classes=4,
                  conv_params=None, nb_neurons_fc=[], rec_cell='')
    summary(model, input_size=input_shape)
    # Only FC layers
    model = DENet(input_shape, nb_classes=4,
                  conv_params=[], nb_neurons_fc=[2048], rec_cell='')
    summary(model, input_size=input_shape)
    # Only rec. cell
    model = DENet(input_shape, nb_classes=4,
                  conv_params=[], nb_neurons_fc=[], rec_cell='gru')
    summary(model, input_size=input_shape)
    # Conv layer & FC layer
    model = DENet(input_shape, nb_classes=4,
                  conv_params=None, nb_neurons_fc=[2048], rec_cell='')
    summary(model, input_size=input_shape)
    # Conv layer & rec. cell
    model = DENet(input_shape, nb_classes=4,
                  conv_params=None, nb_neurons_fc=[], rec_cell='gru')
    summary(model, input_size=input_shape)
    # FC layer & rec. cell
    model = DENet(input_shape, nb_classes=4,
                  conv_params=[], nb_neurons_fc=[2048], rec_cell='gru')
    summary(model, input_size=input_shape)
    # Conv layer & FC layer & rec. cell
    model = DENet(input_shape, nb_classes=4,
                  conv_params=None, nb_neurons_fc=[2048], rec_cell='gru')
    summary(model, input_size=input_shape)
    # Forward the testing data to the model
    model.eval()
    output = model.forward(data)
    print(output.shape)

def test_DENet_default_seqs():
    """ Default parameters -- sequences """
    #--- Sequences (3D) variant -- Format: CHW, with C=1
    batch = 5
    input_shape = (1, 10, 1600)
    # Generate testing data
    data = torch.arange(batch*np.prod(input_shape), dtype=torch.float32)
    data = data.reshape(batch, *input_shape)
    # Instantiate the model
    model = DENet(input_shape, nb_classes=4)
    summary(model, input_size=input_shape)
    # Forward the testing data to the model
    model.eval()
    output = model.forward(data)
    print(output.shape)

def test_DENet_1cv_1fc_seqs():
    """ 1 convolutional layer & 1 FC layer & 1 LSTM -- sequences """
    #--- Sequences (3D) variant -- Format: CHW, with C=1
    batch = 5
    input_shape = (1, 10, 1600)
    # Generate testing data
    data = torch.arange(batch*np.prod(input_shape), dtype=torch.float32)
    data = data.reshape(batch, *input_shape)
    # Parameters
    scl_params = {'nb_filters': 80, 'filter_length': 251, 'frate': 32000,
                  'bandwidth': (0., 16000.), 'padding': 'same'}
    del_params = {'inner_model': None, 'nd_inner_model': False,
                  'sum_out_channels': True, 'dropout': 0.1}
    # Instantiate the model
    model = DENet(input_shape, nb_classes=4,
        scl_params=scl_params, del_params=del_params,
        conv_params=[], reg_conv=(2, 0.2),
        nb_neurons_fc=1024, reg_linear=0.3,
        rec_cell='LSTM', nb_neurons_rec=512)
    summary(model, input_size=input_shape)
    # Forward the testing data to the model
    model.eval()
    output = model.forward(data)
    print(output.shape)

def test_DENet_multiconv_seqs():
    """ 3 same convolutional layers with the same reg. parameters & an LSTM"""
    #--- Sequences (3D) variant -- Format: CHW, with C=1
    batch = 5
    input_shape = (1, 10, 1600)
    # Generate testing data
    data = torch.arange(batch*np.prod(input_shape), dtype=torch.float32)
    data = data.reshape(batch, *input_shape)
    # Parameters
    scl_params = {'nb_filters': 80, 'filter_length': 251, 'frate': 32000,
                  'bandwidth': (0., 16000.), 'padding': 'same'}
    del_params = {'inner_model': None, 'nd_inner_model': True,
                  'sum_out_channels': False, 'dropout': 0.1}
    cv_params = {
        'out_channels': 60, 'kernel_size': 5, 'stride': 1, 'padding': 'valid'}
    # Instantiate the model
    model = DENet(input_shape, nb_classes=4,
        scl_params=scl_params, del_params=del_params,
        conv_params=(2, cv_params),
        reg_conv=[(3, 0.1)],
        rec_cell='LSTM')
    summary(model, input_size=input_shape)
    # Forward the testing data to the model
    model.eval()
    output = model.forward(data)
    print(output.shape)

def test_DENet_multifc_seqs():
    """ 3 different FC layers with the same reg. parameters & a BGRU """
    #--- Chunks (2D) variant -- Format: CW, with C=1
    batch = 5
    input_shape = (1, 10, 1600)
    # Generate testing data
    data = torch.arange(batch*np.prod(input_shape), dtype=torch.float32)
    data = data.reshape(batch, *input_shape)
    # Parameters
    scl_params = {'nb_filters': 80, 'filter_length': 251, 'frate': 32000,
                  'bandwidth': (0., 16000.), 'padding': 'same'}
    del_params = {'inner_model': None, 'nd_inner_model': True,
                  'sum_out_channels': False, 'dropout': 0.1}
    cv_params = {
        'out_channels': 60, 'kernel_size': 6, 'stride': 1, 'padding': 'valid'}
    # Instantiate the model
    model = DENet(input_shape, nb_classes=4,
        scl_params=scl_params, del_params=del_params,
        conv_params=(cv_params, 3),
        nb_neurons_fc=(1024, 512), reg_linear = [0.1],
        rec_cell='BGRU', nb_neurons_rec=512)
    summary(model, input_size=input_shape)
    # Forward the testing data to the model
    model.eval()
    output = model.forward(data)
    print(output.shape)



# Launch test/example functions
test_DENet_no_conv_no_fc_no_rec_seqs()

test_DENet_default_seqs()

test_DENet_1cv_1fc_seqs()

test_DENet_multiconv_seqs()

test_DENet_multifc_seqs()

