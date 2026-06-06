import numpy as np
import torch
from aer.models_tk._models_tk import *


def test_numpy_to_torch_dtype():
    """ Convert a NumPy datatype into a Torch datatype """
    print(numpy_to_torch_dtype(np.float32))
    print(numpy_to_torch_dtype(np.arange(10).dtype))

def test_torch_to_numpy_dtype():
    """ Convert a NumPy datatype into a Torch datatype """
    print(torch_to_numpy_dtype(torch.float32))
    print(torch_to_numpy_dtype(torch.arange(10).dtype))

def test_build_model_name():
    """ Build the name of the file to save """
    print(build_model_name('SincNet', None, 'chks', 10))
    print(build_model_name('SincNet', 'LSTM', 'chks'))
    print(build_model_name('SincNet', 'LSTM', 'chks', 10, 10))

def test_parse_modname():
    """ Parse the model name into more readable strings (for display) """
    print(parse_modname(build_model_name('SincNet', None, 'chks', 10)))
    print(parse_modname(build_model_name('SincNet', 'LSTM', 'chks', 10)))

def test_torch_dataset():
    """ Wrap a data array and its labels into a Torch Dataset """
    inputs = torch.rand(10, 3, 100)     # Format: N x C_{in} x W
    labels = torch.rand(10, 4)          # Format: N x C_{out}
    dataset = torch_dataset(inputs, labels,
        batch_size=10, drop_last=True, shuffle=True)

def test_train_batch():
    """ Train a Torch model on a batched dataset """
    # Define a simple model
    model = torch.nn.Sequential(
        torch.nn.Linear(100, 1), torch.nn.Flatten()).to('cpu')
    # Define the loss function
    loss_fct = torch.nn.CrossEntropyLoss()
    # Define the model optimizer
    optimizer = torch.optim.SGD(model.parameters())
    # Build a simple random dataset
    inputs, labels = torch.rand(10, 5, 100), torch.rand(10, 5)
    dataset = torch_dataset(inputs, labels,
        batch_size=2, drop_last=True, shuffle=True)
    # Train the model
    loss = train_batch(dataset, model, loss_fct, optimizer)

def test_train_epoch():
    """ Train a Torch model on a set of batched datasets (as an epoch) """
    from random import shuffle
    from torch.utils.tensorboard import SummaryWriter
    # Instantiate the TensorBoard writer
    tb_writer = SummaryWriter("runs/model")
    # Define a simple model
    model = torch.nn.Sequential(
        torch.nn.Linear(100, 1), torch.nn.Flatten()).to('cpu')
    # Define the loss function
    loss_fct = torch.nn.CrossEntropyLoss()
    # Define the model optimizer
    optimizer = torch.optim.SGD(model.parameters())
    # Build a simple set of random datasets
    datasets = []
    for i in range(5):
        datasets.append(torch_dataset(
            torch.rand(10, 5, 100), torch.rand(10, 5),
            batch_size=2, drop_last=True, shuffle=True))
    # Train the model with the datasets shuffled for every epoch
    # and save the current epoch loss in the TensorBoard
    for epoch in range(5):
        shuffle(datasets)
        loss = train_epoch(datasets, model, loss_fct, optimizer)
        tb_writer.add_scalar(f'Loss epoch #{epoch+1}', loss[-1], epoch+1)
    # Ensure every loss has been written in the TensorBoard
    tb_writer.flush()

def test_train_model():
    """ Train a model on a set of Torch datasets `epochs` times """
    # Define a simple model
    model = torch.nn.Sequential(
        torch.nn.Linear(100, 1), torch.nn.Flatten()).to('cpu')
    # Define the loss function
    loss_fct = torch.nn.CrossEntropyLoss()
    # Define the model optimizer
    optimizer = torch.optim.SGD(model.parameters())
    # Build a simple set of random datasets
    datasets = []
    for i in range(5):
        datasets.append(torch_dataset(
            torch.rand(10, 5, 100), torch.rand(10, 5),
            batch_size=2, drop_last=True, shuffle=True))
    # Train the model with the datasets
    losses = train_model(datasets, model, loss_fct, optimizer, 10)

def test_test_model_loss():
    """ Loss of a model on a set of Torch datasets """
    # Define a simple model
    model = torch.nn.Sequential(
        torch.nn.Linear(100, 1), torch.nn.Flatten()).to('cpu')
    # Define the loss function
    loss_fct = torch.nn.CrossEntropyLoss()
    # Build a simple set of random datasets
    datasets = []
    for i in range(5):
        datasets.append(torch_dataset(
            torch.rand(10, 5, 100), torch.rand(10, 5),
            batch_size=2, drop_last=True, shuffle=True))
    # Train the model with the datasets
    losses = test_model_loss(datasets, model, loss_fct)

def test_test_model_accuracy():
    """ Classification accuracy of a model on a set of Torch datasets """
    #--- Chunks
    # Define a simple model
    model = torch.nn.Sequential(
        torch.nn.Linear(100, 1), torch.nn.Flatten()).to('cpu')
    # Build a simple set of random datasets
    datasets = []
    for i in range(5):
        datasets.append(torch_dataset(
            torch.rand(10, 5, 100), torch.rand(10, 5),
            batch_size=2, drop_last=True, shuffle=True))
    # Train the model with the datasets
    matrix = test_model_accuracy(datasets, model, True, False)
    #--- Sequences
    # Build a simple set of random datasets
    # Define a simple model
    model = torch.nn.Sequential(
        torch.nn.Linear(100, 1), torch.nn.Flatten(-2)).to('cpu')
    datasets = []
    for i in range(5):
        datasets.append(torch_dataset(
            torch.rand(10, 3, 5, 100), torch.rand(10, 3, 5),
            batch_size=2, drop_last=True, shuffle=True))
    # Evaluate the classification accuracy on the datasets
    matrix = test_model_accuracy(datasets, model, True, sequences=True)



# Launch test/example functions
test_numpy_to_torch_dtype()

test_torch_to_numpy_dtype()

test_build_model_name()

test_parse_modname()

test_torch_dataset()

test_train_batch()

test_train_epoch()

test_train_model()

test_test_model_loss()

test_test_model_accuracy()

