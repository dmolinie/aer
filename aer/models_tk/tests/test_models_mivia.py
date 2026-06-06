import numpy as np
import torch
from aer.models_tk._models_mivia import *


def test_train_model_mivia_chks():
    """ Train a model on a set of files from the MIVIA AE dataset -- Chunks """
    import torch
    import aer.datasets
    import aer.datasets.mivia_loader as loader
    from aer.models import SincNet

    # Folder containing the XML & WAV files
    ROOT = f"{aer.datasets.__path__[0]}/MIVIA_AE_example/training/"
    XMLFOLDER = ROOT
    SNDFOLDER = ROOT + "sounds/"

    # Dataset index and SNR (only available one in the example training set)
    IND, SNR = 66, 15

    # Dataset parameters
    dataset_params = {'folders': (XMLFOLDER, SNDFOLDER), 'SNR': SNR}
    FRATE = 32000
    NB_CLASSES = 4

    # Parsing parameters
    #--- Chunks only variant
    frm_dur = 100e-3                  # Frame duration in seconds
    hop_dur = 50e-3                   # Hop duration in seconds
    chk_params = (frm_dur, hop_dur)   # Chunks settings
    parsing_params = {'nb_classes': NB_CLASSES,
        'chk_params': chk_params, 'seq_params': None}
    input_shape = (1, int(FRATE*frm_dur))

    # Build the model
    scl_params = {'nb_filters': 80, 'filter_length': 251, 'frate': FRATE,
                  'bandwidth': (50., FRATE/2), 'padding': 'same'}
    model = SincNet(input_shape, NB_CLASSES, scl_params,
                    reg_linear=0.3, reg_conv=(3, 0.1))

    # Define the loss function
    loss_fct = torch.nn.CrossEntropyLoss()

    # Define the optimizer to use for model training
    optimizer = torch.optim.SGD(model.parameters(), lr=0.001, momentum=0.9)

    # Train the model on some files only (3 times the same here)
    loss = train_model_mivia(model, loss_fct, optimizer,
       dataset_params, parsing_params, (IND, IND, IND),
       batch_size=4, drop_last=True, shuffle=True)

    # Train the model on a range of files (range of 1 here)
    loss = train_model_mivia(model, loss_fct, optimizer,
        dataset_params, parsing_params, range(IND, IND+1),
        batch_size=4, drop_last=True, shuffle=True)

    # Train the model on a range of files with specific training parameters
    loss = train_model_mivia(model, loss_fct, optimizer,
        dataset_params, parsing_params, range(IND, IND+1),
        epochs=2, batch_size=4, drop_last=True, shuffle=True)


def test_train_model_mivia_seqs():
    """ Train a model on a set of files from the MIVIA AE dataset -- Sequences """
    import torch
    import aer.datasets
    import aer.datasets.mivia_loader as loader
    from aer.models import SincNet

    # Folder containing the XML & WAV files
    ROOT = f"{aer.datasets.__path__[0]}/MIVIA_AE_example/training/"
    XMLFOLDER = ROOT
    SNDFOLDER = ROOT + "sounds/"

    # Dataset index and SNR (only available one in the example training set)
    IND, SNR = 66, 15

    # Dataset parameters
    dataset_params = {'folders': (XMLFOLDER, SNDFOLDER), 'SNR': SNR}
    FRATE = 32000
    NB_CLASSES = 4

    # Parsing parameters
    frm_dur = 50e-3                   # Frame duration in seconds
    chk_params = (frm_dur, frm_dur)   # Chunks settings
    seq_params = (10, 1)              # Sequences settings
    parsing_params = {'nb_classes': NB_CLASSES,
        'chk_params': chk_params, 'seq_params': seq_params}
    input_shape = (1, seq_params[0], int(FRATE*frm_dur))

    # Build the model
    scl_params = {'nb_filters': 80, 'filter_length': 251, 'frate': FRATE,
                  'bandwidth': (50., FRATE/2), 'padding': 'same'}
    model = SincNet(input_shape, NB_CLASSES, scl_params,
                    reg_linear=0.3, reg_conv=(3, 0.1))

    # Define the loss function
    loss_fct = torch.nn.CrossEntropyLoss()

    # Define the optimizer to use for model training
    optimizer = torch.optim.SGD(model.parameters(), lr=0.001, momentum=0.9)

    # Train the model on some files only (3 times the same here)
    loss = train_model_mivia(model, loss_fct, optimizer,
       dataset_params, parsing_params, (IND, IND, IND),
       batch_size=4, drop_last=True, shuffle=True)

    # Train the model on a range of files (range of 1 here)
    loss = train_model_mivia(model, loss_fct, optimizer,
        dataset_params, parsing_params, range(IND, IND+1),
        batch_size=4, drop_last=True, shuffle=True)

    # Train the model on a range of files with specific training parameters
    loss = train_model_mivia(model, loss_fct, optimizer,
        dataset_params, parsing_params, range(IND, IND+1),
        epochs=2, batch_size=4, drop_last=True, shuffle=True)

def test_test_model_loss_mivia():
    """ Evaluate model loss on a set of files from the MIVIA AE dataset """
    import aer.datasets
    import aer.datasets.mivia_loader as loader
    from aer.models import SincNet

    # Folder containing the XML & WAV files
    ROOT = f"{aer.datasets.__path__[0]}/MIVIA_AE_example/training/"
    XMLFOLDER = ROOT
    SNDFOLDER = ROOT + "sounds/"

    # Dataset index and SNR (only available one in the example training set)
    IND, SNR = 66, 15

    # Dataset parameters
    dataset_params = {'folders': (XMLFOLDER, SNDFOLDER), 'SNR': SNR}
    FRATE = 32000
    NB_CLASSES = 4

    # Parsing parameters
    # Chunks variant, see the 'test_model_accuracy_mivia' example for sequences
    frm_dur = 100e-3                  # Frame duration in seconds
    hop_dur = 50e-3                   # Hop duration in seconds
    chk_params = (frm_dur, hop_dur)   # Chunks settings
    parsing_params = {'nb_classes': NB_CLASSES,
        'chk_params': chk_params, 'seq_params': None}
    input_shape = (1, int(FRATE*frm_dur))

    # Build the model
    scl_params = {'nb_filters': 80, 'filter_length': 251, 'frate': FRATE,
                  'bandwidth': (50., FRATE/2), 'padding': 'same'}
    model = SincNet(input_shape, NB_CLASSES, scl_params,
                    reg_linear=0.3, reg_conv=(3, 0.1))

    # Define the loss function
    loss_fct = torch.nn.CrossEntropyLoss()

    # Test the model on some files
    loss = test_model_loss_mivia(
        model, loss_fct, dataset_params, parsing_params, (IND,))

    # Test the model on a range of files
    loss = test_model_loss_mivia(
        model, loss_fct, dataset_params, parsing_params, range(IND, IND+1))

def test_test_model_accuracy_mivia():
    """ Model classification accuracy on a set of MIVIA AE dataset's files """
    import aer.datasets
    import aer.datasets.mivia_loader as loader
    from aer.models import SincNet

    # Folder containing the XML & WAV files
    ROOT = f"{aer.datasets.__path__[0]}/MIVIA_AE_example/training/"
    XMLFOLDER = ROOT
    SNDFOLDER = ROOT + "sounds/"

    # Dataset index and SNR (only available one in the example training set)
    IND, SNR = 66, 15

    # Dataset parameters
    dataset_params = {'folders': (XMLFOLDER, SNDFOLDER), 'SNR': SNR}
    FRATE = 32000
    NB_CLASSES = 4

    # Parsing parameters
    # Sequences variant, see the 'test_model_loss_mivia' example for chunks
    frm_dur = 50e-3                   # Frame duration in seconds
    chk_params = (frm_dur, frm_dur)   # Chunks settings
    seq_params = (10, 1)              # Sequences settings
    parsing_params = {'nb_classes': NB_CLASSES,
        'chk_params': chk_params, 'seq_params': seq_params}
    input_shape = (1, seq_params[0], int(FRATE*frm_dur))

    # Build the model
    scl_params = {'nb_filters': 80, 'filter_length': 251, 'frate': FRATE,
                  'bandwidth': (50., FRATE/2), 'padding': 'same'}
    model = SincNet(input_shape, NB_CLASSES, scl_params,
                    reg_linear=0.3, reg_conv=(3, 0.1))

    # Build confusion matrices, not accuracy items
    conf_mat = True

    # Test the model on some files
    hits = test_model_accuracy_mivia(
        model, dataset_params, parsing_params, (IND,),
        (-1., +1.), conf_mat)

    # Test the model on a range of files
    hits = test_model_accuracy_mivia(
        model, dataset_params, parsing_params, range(IND, IND+1),
        (-1., +1.), conf_mat)



# Launch test/example functions
#test_train_model_mivia_chks() # Long

#test_train_model_mivia_seqs() # Long

test_test_model_loss_mivia()

test_test_model_accuracy_mivia()

