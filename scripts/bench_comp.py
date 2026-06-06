""" Benchmark to compare the inference times and memory footprints of
different configurations of the Torch models

Authors: Dylan MOLINIE
Company: CEA Paris-Saclay, Gif-sur-Yvette, France
Contact: dylan.molinie@cea.fr
Date: April 2026

License: GPLv3
"""

import os
import time
import argparse
import csv

import numpy as np
import torch
from torchinfo import summary

import aer.tools as tls
import aer.accuracy as acc
import aer.models_tk as mtk
from aer.datasets.mivia_loader import build_dataset, LIMITS

tls.exec_file('.pystartup.py', globals(), root='~/')

# Torch parameters
torch.set_default_dtype(torch.float32)
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# Data folders
ROOT = "data/MIVIA_AE/"
XMLFOLDER_TRN = ROOT + "training/"                  # Events XML files folder
SNDFOLDER_TRN = ROOT + "training/sounds/"           # Audio WAV files folder
XMLFOLDER_GEN = ROOT + "testing/"                   # Events XML files folder
SNDFOLDER_GEN = ROOT + "testing/sounds/"            # Audio WAV files folder
NB_CLASSES = 4                                      # Nb of possible classes

# Rescaling bounds
BOUNDS = (-1., +1.)


##############################################################################
##                             Model Selection                              ##
##############################################################################

# Retrieve the model variant from the command line arguments
parser = argparse.ArgumentParser()
parser.add_argument('--benchmark', '-bench',                # Benchmark ID
    type=int, default=0,
    help="ID of the benchmark (to build a dedicated folder)")
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
parser.add_argument('--nb_conv_layers', '-cv_layers',       # Nb of convs.
    type=int, default=3,
    help="Number of convolutional layers")
parser.add_argument('--conv_filters',                       # Nb filters for convs
    type=int, default=60,
    help="Number of filters for the convolutional layers")
parser.add_argument('--nb_linear_layers', '-fc_layers',     # Nb of FC layers
    type=int, default=3,
    help="Number of FC layers")
parser.add_argument('--linear_neurons',                     # Nb neurons (1st FC
    type=int, default=2048,                                 # /2 each following)
    help="Number of neurons of the 1st FC layer; this number is divided by "\
         + "/2^i for each next FC layer (/2, /4, /8, ...)")
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

# Number of convolutional layers and their number of filters
NB_CONV_LAYERS = args.nb_conv_layers        # Nb CV layers
CONV_FILTERS = args.conv_filters            # Nb filters

# Number of FC layers and the number of neurons of the first one
# This number is /2^i for each following layer (/2, /4, /8, ...)
NB_FC_LAYERS = args.nb_linear_layers        # Nb FC layers
NB_NEURONS_FC = args.linear_neurons         # Nb neurons 1st layer

# Identifier of the benchmark
BENCH = args.benchmark


# Model & hits folders
FOLDER_ROOT = tls.check_folder(f"Models/Bench_{BENCH}/")
FOLDER_MODS = tls.check_folder(FOLDER_ROOT+f"Models/{SNR}/{DTYPE}/")
FOLDER_HITS = tls.check_folder(FOLDER_ROOT+f"Hits/{SNR}/{DTYPE}/")
FOLDER_COMP_STATS = tls.check_folder(FOLDER_ROOT+"Comparison/Stats/")
FOLDER_COMP_HITS = tls.check_folder(FOLDER_ROOT+"Comparison/Hits/")

# Build model & folders names
dtype = 'seqs' if SEQUENCES else 'chks'
bar_name = mtk.build_model_name(VARIANT, RECURRENT)
if BENCH == 1:
    bar_name += f'_{NB_CONV_LAYERS}CV_{NB_FC_LAYERS}FC'
if BENCH == 2:
    bar_name += f'_{CONV_FILTERS}Filters_{NB_NEURONS_FC}Neurons'
mod_name = bar_name + f"_{dtype}_{SNR}"
folder_mods = tls.check_folder(FOLDER_MODS + bar_name + '/')
folder_hits = tls.check_folder(FOLDER_HITS + bar_name + '/')

# Items to compare (to be written in the CSV file)
if BENCH == 0:
    ITEMS = ['Variant', 'Rec. cell']
    items = [VARIANT, RECURRENT]
elif BENCH == 1:
    ITEMS = ['# Convolutions', '# Fully Connected']
    items = [NB_CONV_LAYERS, NB_FC_LAYERS]
else:
    ITEMS = ['# CV Filters', '# Neurons (1st FC)']
    items = [CONV_FILTERS, NB_NEURONS_FC]

##############################################################################



##############################################################################
##                         Model Parameters Setting                         ##
##############################################################################

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

#----------------------   Load Data for Calibration   -----------------------#
if SEQUENCES:
    # Load the data and build the sequences of chunks
    specs, data, classes = build_dataset(
        (XMLFOLDER_TRN, SNDFOLDER_TRN), (66, SNR),
        NB_CLASSES, chk_params, seq_params,
        limits=LIMITS, bounds=BOUNDS)
    input_shape = (1, *data.shape[1:])          # Data format: NCHW
else:
    # Load the data and build the chunks
    specs, data, classes = build_dataset(
        (XMLFOLDER_TRN, SNDFOLDER_TRN), (66, SNR),
        NB_CLASSES, chk_params,
        limits=LIMITS, bounds=BOUNDS)
    input_shape = (1, data.shape[1])            # Data format: NCW

# Convert the data into batched Torch Tensor
tensors = torch.Tensor(data).unsqueeze(1)       # Batch the data
#----------------------------------------------------------------------------#

##############################################################################



##############################################################################
##                 Evaluation of an Already-Existing Model                  ##
##############################################################################

##--- Same results for all SNRs; to be executed only once

#print(mod_name)

#times = np.empty(2, float)

## Evaluate time to load the model
#tps_bfr = time.time()
#model = torch.load(folder_mods+mod_name+'.pth', weights_only=False)
#model.to(DEVICE)
#model = model.eval()
#tps_aft = time.time()
#times[0] = tps_aft - tps_bfr

## Evaluate inference time
#tps_bfr = time.time()
#with torch.no_grad():
#    res = model.forward(tensors)
#tps_aft = time.time()
#times[1] = (tps_aft - tps_bfr) / len(tensors)


## Retrieve model stats
#stats = summary(model, input_size=tensors[:1].shape, verbose=0)
#params = stats.total_mult_adds, stats.total_params
#sizes = stats.total_param_bytes, stats.total_output_bytes


## Write the headers only if the file does not exist
#csvfile = FOLDER_COMP_STATS + f"stats_{dtype}.csv"
#if not os.path.exists(csvfile):
#    with open(csvfile, 'a', encoding='utf-8') as file:
#        writer = csv.writer(file)
#        writer.writerow(
#            ['Model', *ITEMS,
#             'Time Loading [s]', 'Time Forward (1 batch) [s]',
#             '# Mult-Adds', '# Params',
#             'Size params [B]', 'Size output [B]', 'Total size [B]'])

## Write the statistics in a csv file
#with open(csvfile, 'a', encoding='utf-8') as file:
#    writer = csv.writer(file)
#    writer.writerow(
#        [mod_name, *items,
#        *times.round(10),
#        *params,
#        *sizes, sum(sizes)])

##############################################################################



##############################################################################
##                              Save Mean Hits                              ##
##############################################################################

#--- Should be executed for every SNR

def write_headers(csvfile, ITEMS, headers, classes):
    """ Write the headers only if the file does not exist """
    if not os.path.exists(csvfile):
        with open(csvfile, 'a', encoding='utf-8') as file:
            writer = csv.writer(file)
            writer.writerow(['', '', '', *headers])
            writer.writerow(['Model', *ITEMS, *classes, *classes, *classes])

def write_stats(csvfile, mod_name, items, avgs, sens, spec):
    """ Write the statistics in a csv file """
    # TPs, Sensitivity and Specificity for Chunks, and for Sequences then if so
#    results = np.ravel(np.concatenate((avgs, sens, spec), axis=-1)).round(3)
    with open(csvfile, 'a', encoding='utf-8') as file:
        writer = csv.writer(file)
        writer.writerow([mod_name, *items, *avgs, *sens, *spec])

# The classes
HEADERS = ['True Positives', 'Sensitivity', 'Specificity']
CLASSES = ['BGND', 'Glass', 'Shot', 'Scream', 'Mean']

# Reload the hits (if needed)
idx = EPOCHS - 1
cmat_trn = acc.load_hits(folder_hits+f"E{idx}_ConfMat_trn.csv", SEQUENCES)[1]
cmat_gen = acc.load_hits(folder_hits+f"E{idx}_ConfMat_gen.csv", SEQUENCES)[1]

# Get the hits (TPs, FPs, FNs & TNs) from the confusion matrices
hits_trn = [acc.conf_mat_to_acc_items(mat, True) for mat in cmat_trn]
hits_gen = [acc.conf_mat_to_acc_items(mat, True) for mat in cmat_gen]

# Compute the number of hits per class
avgs_trn = acc.compute_avgs(hits_trn, 'TP', '1s_ref')
avgs_gen = acc.compute_avgs(hits_gen, 'TP', '1s_ref')

# Retrieve the last row (which is the mean)
# Flatten (ravel) the arrays in case of sequences
res_trn = np.ravel(avgs_trn[-1]).round(3)
res_gen = np.ravel(avgs_gen[-1]).round(3)

# Compute the (mean) sensitivity per class
sens_trn = acc.sensitivity(hits_trn, False).mean(0)
sens_gen = acc.sensitivity(hits_gen, False).mean(0)
sens_trn = np.concatenate((sens_trn, sens_trn.mean(-1, keepdims=True)), axis=-1)
sens_gen = np.concatenate((sens_gen, sens_gen.mean(-1, keepdims=True)), axis=-1)

# Compute the (mean) specificity per class
spec_trn = acc.specificity(hits_trn, False).mean(0)
spec_gen = acc.specificity(hits_gen, False).mean(0)
spec_trn = np.concatenate((spec_trn, spec_trn.mean(-1, keepdims=True)), axis=-1)
spec_gen = np.concatenate((spec_gen, spec_gen.mean(-1, keepdims=True)), axis=-1)

# Append 'chks' and 'seqs' to the headers if SEQUENCES
if SEQUENCES:
    # Append suffix to headers
    heads = []
    for head in HEADERS:
        heads += [head+' (chks)', head+' (seqs)']
    # Double the classes (one for chunks, one for sequences)
    classes = CLASSES + CLASSES
else:
    heads = HEADERS
    classes = CLASSES

# Add empty strings to headers to match the columns
headers = []
for head in heads:
    headers += [head, '', '', '', '']

# The mean hits of the training datasets
csvfile = FOLDER_COMP_HITS + f'res_hits_trn_{dtype}_{SNR}.csv'
write_headers(csvfile, ITEMS, headers, classes)
write_stats(csvfile, mod_name, items,
    np.ravel(res_trn).round(3),
    np.ravel(sens_trn).round(3),
    np.ravel(spec_trn).round(3))

# The mean hits of the testing datasets
csvfile = FOLDER_COMP_HITS + f'res_hits_gen_{dtype}_{SNR}.csv'
write_headers(csvfile, ITEMS, headers, classes)
write_stats(csvfile, mod_name, items,
    np.ravel(res_gen).round(3),
    np.ravel(sens_gen).round(3),
    np.ravel(spec_gen).round(3))

##############################################################################



##############################################################################
##                    Evaluation of a Non-Existing Model                    ##
##############################################################################

#import aer.models as nets

#print(mod_name)


#times = np.empty(2, float)

##---------------------------   Model Parameters   ---------------------------#
## Convolutional, Recurrent (if any) and Linear layers
#CV_PARAMS = {'out_channels': CONV_FILTERS,
#             'kernel_size': 10, 'stride': 1, 'padding': 'valid'}
#NB_NEURONS_FC = [NB_NEURONS_FC // i for i in range(1, NB_FC_LAYERS+1)]
#NB_NEURONS_REC = 512

## Regularization layers
#REG_CONV = (3, 0.1)
#REG_LINEAR = 0.1

## Set the parameters
#if VARIANT != 'ConvNet':
#    # If SincNet, use 2 convolutional layers + 1 SincLayer
#    CONV_PARAMS = (NB_CONV_LAYERS-1, CV_PARAMS)
#    SCL_PARAMS = {'nb_filters': CONV_FILTERS,
#                  'filter_length': 251, 'padding': 'valid',
#                  'frate': specs[0], 'bandwidth': (50., specs[0]/2)}
#    REG_SCL = (3, 0.1)
#    if VARIANT == 'DENet':
#        # If DENet, add also 1 DELayer
#        DEL_PARAMS = {'sum_out_channels': False, 'dropout': 0.3,
#                      'nd_inner_model': True}
#        REG_DEL = (3, 0.1)
#else:
#    # If ConvNet, use only 3 convolutional layers
#    CONV_PARAMS = (NB_CONV_LAYERS, CV_PARAMS)
##----------------------------------------------------------------------------#

##--------------------------   Evaluate the Model   --------------------------#
## Evaluate the time to build the model
#tps_bfr = time.time()
#if VARIANT == 'ConvNet':
#    model = nets.ConvNet(
#        input_shape, NB_CLASSES,
#        conv_params=CONV_PARAMS, reg_conv=REG_CONV,
#        nb_neurons_fc = NB_NEURONS_FC, reg_linear=REG_LINEAR,
#        rec_cell=RECURRENT, nb_neurons_rec=NB_NEURONS_REC)
#elif VARIANT == 'SincNet':
#    model = nets.SincNet(
#        input_shape, NB_CLASSES,
#        scl_params=SCL_PARAMS, reg_scl=REG_SCL,
#        conv_params=CONV_PARAMS, reg_conv=REG_CONV,
#        nb_neurons_fc = NB_NEURONS_FC, reg_linear=REG_LINEAR,
#        rec_cell=RECURRENT, nb_neurons_rec=NB_NEURONS_REC)
#elif VARIANT == 'DENet':
#    model = nets.DENet(
#        input_shape, NB_CLASSES,
#        scl_params=SCL_PARAMS, reg_scl=REG_SCL,
#        del_params=DEL_PARAMS, reg_del=REG_DEL,
#        conv_params=CONV_PARAMS, reg_conv=REG_CONV,
#        nb_neurons_fc = NB_NEURONS_FC, reg_linear=REG_LINEAR,
#        rec_cell=RECURRENT, nb_neurons_rec=NB_NEURONS_REC)
#model.to(DEVICE)
#model = model.eval()
#tps_aft = time.time()
#times[0] = tps_aft - tps_bfr

## Evaluate inference time
#tps_bfr = time.time()
#with torch.no_grad():
#    res = model.forward(tensors)
#tps_aft = time.time()
#times[1] = (tps_aft - tps_bfr) / len(tensors)


## Retrieve model stats
#stats = summary(model, input_size=tensors[:1].shape, verbose=0)
#params = stats.total_mult_adds, stats.total_params
#sizes = stats.total_param_bytes, stats.total_output_bytes


## Write the headers only if the file does not exist
#csvfile = FOLDER_COMP_STATS + f"stats_{dtype}.csv"
#if not os.path.exists(csvfile):
#    with open(csvfile, 'a', encoding='utf-8') as file:
#        writer = csv.writer(file)
#        writer.writerow(
#            ['Model', *ITEMS,
#             'Time Loading [s]', 'Time Forward (1 batch) [s]',
#             '# Mult-Adds', '# Params',
#             'Size params [B]', 'Size output [B]', 'Total size [B]'])

## Write the statistics in a csv file
#with open(csvfile, 'a', encoding='utf-8') as file:
#    writer = csv.writer(file)
#    writer.writerow(
#        [mod_name, *items,
#        *times.round(10),
#        *params,
#        *sizes, sum(sizes)])
##----------------------------------------------------------------------------#

##############################################################################
