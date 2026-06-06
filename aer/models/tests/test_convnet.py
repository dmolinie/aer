import numpy as np
import torch
from torchinfo import summary
from aer.models._convnet import *


def test_ConvNet_no_conv_no_fc_no_rec_chks():
    """ Default parameters -- chunks """
    #--- Chunks (2D) variant -- Format: CW, with C=1
    batch = 5
    input_shape = (1, 3200)
    # Generate testing data
    data = torch.arange(batch*np.prod(input_shape), dtype=torch.float32)
    data = data.reshape(batch, *input_shape)
    # Instantiate the model
    # No layers
    model = ConvNet(input_shape, nb_classes=4,
                    conv_params=[], nb_neurons_fc=[], rec_cell='')
    summary(model, input_size=input_shape)
    # Only conv layers (default used)
    model = ConvNet(input_shape, nb_classes=4,
                    conv_params=None, nb_neurons_fc=[], rec_cell='')
    summary(model, input_size=input_shape)
    # Only FC layers
    model = ConvNet(input_shape, nb_classes=4,
                    conv_params=[], nb_neurons_fc=[2048], rec_cell='')
    summary(model, input_size=input_shape)
    # Only rec. cell
    model = ConvNet(input_shape, nb_classes=4,
                    conv_params=[], nb_neurons_fc=[], rec_cell='gru')
    summary(model, input_size=input_shape)
    # Conv layer & FC layer
    model = ConvNet(input_shape, nb_classes=4,
                    conv_params=None, nb_neurons_fc=[2048], rec_cell='')
    summary(model, input_size=input_shape)
    # Conv layer & rec. cell
    model = ConvNet(input_shape, nb_classes=4,
                    conv_params=None, nb_neurons_fc=[], rec_cell='gru')
    summary(model, input_size=input_shape)
    # FC layer & rec. cell
    model = ConvNet(input_shape, nb_classes=4,
                    conv_params=[], nb_neurons_fc=[2048], rec_cell='gru')
    summary(model, input_size=input_shape)
    # Conv layer & FC layer & rec. cell
    model = ConvNet(input_shape, nb_classes=4,
                    conv_params=None, nb_neurons_fc=[2048], rec_cell='gru')
    summary(model, input_size=input_shape)
    # Forward the testing data to the model
    model.eval()
    output = model.forward(data)
    print(output.shape)

def test_ConvNet_no_conv_no_fc_no_rec_seqs():
    """ Default parameters -- sequences """
    #--- Sequences (3D) variant -- Format: CHW, with C=1
    batch = 5
    input_shape = (1, 10, 1600)
    # Generate testing data
    data = torch.arange(batch*np.prod(input_shape), dtype=torch.float32)
    data = data.reshape(batch, *input_shape)
    # Instantiate the model
    # No layers
    model = ConvNet(input_shape, nb_classes=4,
                    conv_params=[], nb_neurons_fc=[], rec_cell='')
    summary(model, input_size=input_shape)
    # Only conv layers (default used)
    model = ConvNet(input_shape, nb_classes=4,
                    conv_params=None, nb_neurons_fc=[], rec_cell='')
    summary(model, input_size=input_shape)
    # Only FC layers
    model = ConvNet(input_shape, nb_classes=4,
                    conv_params=[], nb_neurons_fc=[2048], rec_cell='')
    summary(model, input_size=input_shape)
    # Only rec. cell
    model = ConvNet(input_shape, nb_classes=4,
                    conv_params=[], nb_neurons_fc=[], rec_cell='gru')
    summary(model, input_size=input_shape)
    # Conv layer & FC layer
    model = ConvNet(input_shape, nb_classes=4,
                    conv_params=None, nb_neurons_fc=[2048], rec_cell='')
    summary(model, input_size=input_shape)
    # Conv layer & rec. cell
    model = ConvNet(input_shape, nb_classes=4,
                    conv_params=None, nb_neurons_fc=[], rec_cell='gru')
    summary(model, input_size=input_shape)
    # FC layer & rec. cell
    model = ConvNet(input_shape, nb_classes=4,
                    conv_params=[], nb_neurons_fc=[2048], rec_cell='gru')
    summary(model, input_size=input_shape)
    # Conv layer & FC layer & rec. cell
    model = ConvNet(input_shape, nb_classes=4,
                    conv_params=None, nb_neurons_fc=[2048], rec_cell='gru')
    summary(model, input_size=input_shape)
    # Forward the testing data to the model
    model.eval()
    output = model.forward(data)
    print(output.shape)

def test_ConvNet_default_chks():
    """ Default parameters -- chunks """
    #--- Chunks (2D) variant -- Format: CW, with C=1
    batch = 5
    input_shape = (1, 3200)
    # Generate testing data
    data = torch.arange(batch*np.prod(input_shape), dtype=torch.float32)
    data = data.reshape(batch, *input_shape)
    # Instantiate the model
    model = ConvNet(input_shape, nb_classes=4)
    summary(model, input_size=input_shape)
    # Forward the testing data to the model
    model.eval()
    output = model.forward(data)
    print(output.shape)

def test_ConvNet_default_seqs():
    """ Default parameters -- sequences """
    #--- Sequences (3D) variant -- Format: CHW, with C=1
    batch = 5
    input_shape = (1, 10, 1600)
    # Generate testing data
    data = torch.arange(batch*np.prod(input_shape), dtype=torch.float32)
    data = data.reshape(batch, *input_shape)
    # Instantiate the model
    model = ConvNet(input_shape, nb_classes=4)
    summary(model, input_size=input_shape)
    # Forward the testing data to the model
    model.eval()
    output = model.forward(data)
    print(output.shape)

def test_ConvNet_1cv_1fc_chks():
    """ 1 convolutional layer & 1 FC layer & 1 BGRU-- chunks """
    #--- Chunks (2D) variant -- Format: CW, with C=1
    batch = 5
    input_shape = (1, 3200)
    # Generate testing data
    data = torch.arange(batch*np.prod(input_shape), dtype=torch.float32)
    data = data.reshape(batch, *input_shape)
    # Parameters
    conv_params = {
        'out_channels': 40, 'kernel_size': 10, 'stride': 1, 'padding': 'valid'}
    # Instantiate the model
    model = ConvNet(input_shape, nb_classes=4,
        conv_params=conv_params, reg_conv=(2, 0.2),
        nb_neurons_fc=1024, reg_linear=0.3,
        rec_cell='GRU', nb_neurons_rec=512)
    summary(model, input_size=input_shape)
    # Forward the testing data to the model
    model.eval()
    output = model.forward(data)
    print(output.shape)

def test_ConvNet_1cv_1fc_seqs():
    """ 1 convolutional layer & 1 FC layer & 1 LSTM -- sequences """
    #--- Sequences (3D) variant -- Format: CHW, with C=1
    batch = 5
    input_shape = (1, 10, 1600)
    # Generate testing data
    data = torch.arange(batch*np.prod(input_shape), dtype=torch.float32)
    data = data.reshape(batch, *input_shape)
    # Parameters
    conv_params = {
        'out_channels': 40, 'kernel_size': 10, 'stride': 1, 'padding': 'same'}
    # Instantiate the model
    model = ConvNet(input_shape, nb_classes=4,
        conv_params=conv_params, reg_conv=(2, 0.2),
        nb_neurons_fc=1024, reg_linear=0.3,
        rec_cell='LSTM', nb_neurons_rec=512)
    summary(model, input_size=input_shape)
    # Forward the testing data to the model
    model.eval()
    output = model.forward(data)
    print(output.shape)

def test_ConvNet_multiconv_chks():
    """ 3 different convolutional layers with different reg. parameters """
    #--- Chunks (2D) variant -- Format: CW, with C=1
    batch = 5
    input_shape = (1, 3200)
    # Generate testing data
    data = torch.arange(batch*np.prod(input_shape), dtype=torch.float32)
    data = data.reshape(batch, *input_shape)
    # Parameters
    cv_params1 = {
        'out_channels': 60, 'kernel_size': 6, 'stride': 2, 'padding': 'valid'}
    cv_params2 = {
        'out_channels': 40, 'kernel_size': 5, 'stride': 1, 'padding': 'same'}
    cv_params3 = {
        'out_channels': 20, 'kernel_size': 3, 'stride': 1, 'padding': 'valid'}
    reg_conv1 = (3, 0.1)
    reg_conv2 = (5, 0.1)
    reg_conv3 = (3, 0.0)
    # Instantiate the model
    model = ConvNet(input_shape, nb_classes=4,
        conv_params=(cv_params1, cv_params2, cv_params3),
        reg_conv=(reg_conv1, reg_conv2, reg_conv3))
    summary(model, input_size=input_shape)
    # Forward the testing data to the model
    model.eval()
    output = model.forward(data)
    print(output.shape)

def test_ConvNet_multiconv_seqs():
    """ 3 same convolutional layers with the same reg. parameters & an LSTM"""
    #--- Sequences (3D) variant -- Format: CHW, with C=1
    batch = 5
    input_shape = (1, 10, 1600)
    # Generate testing data
    data = torch.arange(batch*np.prod(input_shape), dtype=torch.float32)
    data = data.reshape(batch, *input_shape)
    # Parameters
    cv_params = {
        'out_channels': 60, 'kernel_size': 5, 'stride': 1, 'padding': 'valid'}
    # Instantiate the model
    model = ConvNet(input_shape, nb_classes=4,
        conv_params=(3, cv_params),
        reg_conv=[(3, 0.1)],
        rec_cell='LSTM')
    summary(model, input_size=input_shape)
    # Forward the testing data to the model
    model.eval()
    output = model.forward(data)
    print(output.shape)

def test_ConvNet_multifc_chks():
    """ 3 different FC layers with different reg. parameters """
    #--- Chunks (2D) variant -- Format: CW, with C=1
    batch = 5
    input_shape = (1, 3200)
    # Generate testing data
    data = torch.arange(batch*np.prod(input_shape), dtype=torch.float32)
    data = data.reshape(batch, *input_shape)
    # Parameters
    cv_params = {
        'out_channels': 60, 'kernel_size': 6, 'stride': 2, 'padding': 'valid'}
    nb_neurons_fc = (2048, 1024, 512)
    reg_linear = (0.1, 0.5, 0.0)
    # Instantiate the model
    model = ConvNet(input_shape, nb_classes=4,
        conv_params=cv_params,
        nb_neurons_fc=nb_neurons_fc,
        reg_linear=reg_linear)
    summary(model, input_size=input_shape)
    # Forward the testing data to the model
    model.eval()
    output = model.forward(data)
    print(output.shape)

def test_ConvNet_multifc_seqs():
    """ 3 different FC layers with the same reg. parameters & a BGRU """
    #--- Chunks (2D) variant -- Format: CW, with C=1
    batch = 5
    input_shape = (1, 10, 1600)
    # Generate testing data
    data = torch.arange(batch*np.prod(input_shape), dtype=torch.float32)
    data = data.reshape(batch, *input_shape)
    # Parameters
    cv_params = {
        'out_channels': 60, 'kernel_size': 6, 'stride': 1, 'padding': 'valid'}
    # Instantiate the model
    model = ConvNet(input_shape, nb_classes=4,
        conv_params=(cv_params, 3),
        nb_neurons_fc=(1024, 512), reg_linear = [0.1],
        rec_cell='BGRU', nb_neurons_rec=512)
    summary(model, input_size=input_shape)
    # Forward the testing data to the model
    model.eval()
    output = model.forward(data)
    print(output.shape)



# Launch test/example functions
test_ConvNet_no_conv_no_fc_no_rec_chks()

test_ConvNet_no_conv_no_fc_no_rec_seqs()

test_ConvNet_default_chks()

test_ConvNet_default_seqs()

test_ConvNet_1cv_1fc_chks()

test_ConvNet_1cv_1fc_seqs()

test_ConvNet_multiconv_chks()

test_ConvNet_multiconv_seqs()

test_ConvNet_multifc_chks()

test_ConvNet_multifc_seqs()

