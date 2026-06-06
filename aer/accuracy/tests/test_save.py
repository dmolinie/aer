from aer.accuracy._metrics import *
from aer.accuracy._save import *


def test_save():
    """ Save accuracy on sequences (confusion matrix or accuracy items) """
    import aer.datasets
    import aer.datasets.mivia_loader as loader

    # Dataset parameters
    ROOT = f"{aer.datasets.__path__[0]}/MIVIA_AE_example/training/"
    XMLFOLDER = ROOT
    SNDFOLDER = ROOT + "sounds/"

    # Dataset index and SNR (only available one in the example training set)
    IND, SNR = 66, 15

    # Chunks parameters
    FRM_DURATION = 50e-3    # Frame duration in seconds
    CHK_PARAMS = (FRM_DURATION, FRM_DURATION)

    # Number of possible classes
    NB_CLASSES = 4

    # Load the data and build the chunks
    specs, data_chks, class_chks = loader.build_dataset(
        (XMLFOLDER, SNDFOLDER), (IND, SNR), NB_CLASSES, CHK_PARAMS)

    # Generate a dummy torch model
    import torch
    model = torch.nn.Sequential(
        torch.nn.Linear(data_chks.shape[-1], NB_CLASSES),
        torch.nn.Softmax(1))

    # Test the model on some files
    conf_mat = True         # Confusion matrices, not accuracy items
    class_est = model.forward(torch.Tensor(data_chks)).detach().numpy()
    hits = accuracy_chunks(class_est, class_chks, conf_mat)

    # Save the results (list of 1 dict)
    save_hits(f"Hits/Testing_chks.csv", hits)

    # Save the results (list of 3 dicts)
    save_hits(f"Hits/Testing_chks_3.csv", hits+hits+hits)


def test_load():
    """ Classifications accuracy items on 2D arrays (class by class) """
    import aer.datasets
    import aer.datasets.mivia_loader as loader

    # Dataset parameters
    ROOT = f"{aer.datasets.__path__[0]}/MIVIA_AE_example/training/"
    XMLFOLDER = ROOT
    SNDFOLDER = ROOT + "sounds/"

    # Dataset index and SNR (only available one in the example training set)
    IND, SNR = 66, 15

    # Chunks parameters
    FRM_DURATION = 50e-3    # Frame duration in seconds
    CHK_PARAMS = (FRM_DURATION, FRM_DURATION)
    SEQ_PARAMS = (10, 1)

    # Number of possible classes
    NB_CLASSES = 4

    # Load the data and build the chunks
    specs, data_chks, class_chks = loader.build_dataset(
        (XMLFOLDER, SNDFOLDER), (IND, SNR), NB_CLASSES, CHK_PARAMS, SEQ_PARAMS)

    # Generate a dummy torch model
    import torch
    model = torch.nn.Sequential(
        torch.nn.Linear(data_chks.shape[-1], NB_CLASSES),
        torch.nn.Softmax(1))

    # Test the model on some files
    conf_mat = True         # Confusion matrices, not accuracy items
    class_est = model.forward(torch.Tensor(data_chks)).detach().numpy()
    hits = accuracy_sequences(class_est, class_chks, conf_mat)

    # Save the results (list of 1 tuple of 2 dicts)
    save_hits(f"Hits/Testing_seqs.csv", hits)

    # Save the results (list of 3 tuples of 2 dicts)
    save_hits(f"Hits/Testing_seqs_3.csv", hits+hits+hits)

    # Load the classes & hits
    classes, hits = load_hits(f"Hits/Testing_seqs.csv", True, NB_CLASSES)
    print(classes)
    print(hits)

    # Load the classes & hits
    classes, hits = load_hits(f"Hits/Testing_seqs_3.csv", True, NB_CLASSES)
    print(classes)
    print(hits)



# Launch test/example functions
test_save()

test_load()

