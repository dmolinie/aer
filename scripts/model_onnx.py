""" Script to export Torch models into ONNX computation graphs

Authors: Dylan MOLINIE
Company: CEA Paris-Saclay, Gif-sur-Yvette, France
Contact: dylan.molinie@cea.fr
Date: December 2025

License: GPLv3
"""

import argparse

import numpy as np
import torch

import aer.tools as tls
import aer.models_tk as mtk
import aer.onnx as oxtk
from aer.datasets.mivia_loader import build_dataset, LIMITS

tls.exec_file('.pystartup.py', globals(), root='~/')

# Data folders
ROOT = "data/MIVIA_AE/"
XMLFOLDER_TRN = ROOT + "training/"                  # Events XML files folder
SNDFOLDER_TRN = ROOT + "training/sounds/"           # Audio WAV files folder
XMLFOLDER_GEN = ROOT + "testing/"                   # Events XML files folder
SNDFOLDER_GEN = ROOT + "testing/sounds/"            # Audio WAV files folder
NB_CLASSES = 4                                      # Nb of possible classes


##############################################################################
##                             Model Selection                              ##
##############################################################################

# Retrieve the model variant from the command line arguments
parser = argparse.ArgumentParser()
parser.add_argument('--variant', '-var',                    # Model variant
    type=str, default='ConvNet',
    help="Model variant: 'ConvNet', 'SincNet', 'DENet'")
parser.add_argument('--recurrent', '-rec',                  # Recurrent cell
    type=str, default='NoRec',
    help="Recurrent cell, if any: 'NoRec', 'LSTM', 'GRU'")
parser.add_argument('--snr', '-snr',                        # SNR
    type=int, default=20,
    help="Data SNR: 5, 10, 15, 20, 25, 30")
parser.add_argument('--sequences', '-seq',                  # If sequences
    type=bool, action=argparse.BooleanOptionalAction, default=False,
    help="If sequences of chunks or just chunks; use '--sequences' for True "\
         + "and '--no-sequences' for False")
parser.add_argument('--epochs', '-eps',                     # Nb of epochs
    type=int, default=10,
    help="Number of epochs for training")
args = parser.parse_args()


# Choose the network variant
#VARIANT = ['ConvNet', 'SincNet', 'DENet'][0]
VARIANT = args.variant

# Set the recurrent cell
#RECURRENT = ['NoRec', 'LSTM', 'GRU'][0]
RECURRENT = args.recurrent

# Choose the input data format
#SEQUENCES = [False, True][0]
SEQUENCES = args.sequences

# Set the dataset SNR
#SNR = [5, 10, 15, 20, 25, 30][4]
SNR = args.snr

# Set the number of epochs to use for training
EPOCHS = args.epochs

# Set the data format extension
DTYPE = 'Sequences' if SEQUENCES else 'Chunks'


# Model & hits folders
FOLDER_ROOT = tls.check_folder("Models/")
FOLDER_MODS = tls.check_folder(FOLDER_ROOT+f"Models/{SNR}/{DTYPE}/")

# Build model & folders names
dtype = 'seqs' if SEQUENCES else 'chks'
bar_name = mtk.build_model_name(VARIANT, RECURRENT)
mod_name = bar_name + f"_{dtype}_{SNR}"
folder_mods = tls.check_folder(FOLDER_MODS + bar_name + '/')

##############################################################################



##############################################################################
##                         Final Model ONNX Export                          ##
##############################################################################

#def compare_res(session, model, data):
#    """ Compare the accuracy of a Torch model and its ONNX counterpart """
#    with torch.no_grad():
#        # Load and use the ONNX model
#        pred_nx = oxtk.run_onnx(session, data.detach().numpy())[0]
#        # Load and use the saved Torch model
#        pred_ks = model.forward(data).detach().numpy()
#        # Compare the ONNX & Torch models predictions
#        print("Torch and ONNX close:", np.allclose(pred_nx, pred_ks, atol=1e-1))
#    return pred_ks-pred_nx

#--------------------------   Parsing Parameters   --------------------------#
# Set the frame & hop durations
if SEQUENCES:
    FRM_DURATION = 50e-3                        # Frame duration in seconds
    HOP_DURATION = FRM_DURATION                 # Hop duration in seconds
    chk_params = (FRM_DURATION, FRM_DURATION)   # Chunks settings
    seq_params = (10, 1)                        # Sequences settings
else:
    FRM_DURATION = 100e-3                       # Frame duration in seconds
    HOP_DURATION = 50e-3                        # Hop duration in seconds
    chk_params = (FRM_DURATION, HOP_DURATION)   # Chunks settings
    seq_params = None

# Set the dataset parameters
dataset_trn = {'folders': (XMLFOLDER_TRN, SNDFOLDER_TRN), 'SNR': SNR}
dataset_gen = {'folders': (XMLFOLDER_GEN, SNDFOLDER_GEN), 'SNR': SNR}
parsing_params = {
    'nb_classes': NB_CLASSES, 'chk_params': chk_params, 'seq_params': seq_params}
#----------------------------------------------------------------------------#

#------------------------   Load Calibration Data   -------------------------#
# Data rescaling bounds
BOUNDS = (-1., +1.)

# Set the frame & hop durations
if SEQUENCES:
    # Load the data and build the sequences of chunks
    specs, data, classes = build_dataset(
        (XMLFOLDER_TRN, SNDFOLDER_TRN), (66, SNR),
        NB_CLASSES, chk_params, seq_params,
        limits=LIMITS, bounds=BOUNDS, dtype=np.float32)
else:
    # Load the data and build the chunks
    specs, data, classes = build_dataset(
        (XMLFOLDER_TRN, SNDFOLDER_TRN), (66, SNR),
        NB_CLASSES, chk_params,
        limits=LIMITS, bounds=BOUNDS, dtype=np.float32)

# Extract only the first batch (so that the ONNX graph will use only 1 batch)
# Use slice [:1] to keep the dimensions
data = data[:1]
classes = classes[:1]

# Convert the data into Torch tensors
tensor = torch.Tensor(data).unsqueeze(1)        # NC(H)W format, with C=1 here
#classes = torch.Tensor(classes)
#----------------------------------------------------------------------------#

#-------------------------   Perform ONNX Export   --------------------------#
# Load the Torch model & export it to ONNX
model = torch.load(folder_mods+mod_name+'.pth', weights_only=False)

# Beware 'dynamic_batch', as it adds additional layers to the ONNX export
print(f"Exporting model '{mod_name}' in folder '{folder_mods}'")
model_onnx = oxtk.torch2onnx(
    folder_mods+mod_name+'.onnx', model, tensor,
    dynamic_batch=False, dynamo=True, verbose=False)

## Compare Torch & ONNX models accuracy
## (ONNX must be reloaded as it is wrapped into an onnxruntime session object)
#error = compare_res(oxtk.load_onnx(folder_mods+mod_name+'.onnx'), model, tensor)

# List and save the model operators (ONNX version)
ops = [node.op_type for node in model_onnx.model.graph.all_nodes()]
tls.save_as_csv(folder_mods+mod_name+'.csv', ops)
#----------------------------------------------------------------------------#

##############################################################################
