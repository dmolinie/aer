""" Benchmark to test different configurations of the Torch models

Authors: Dylan MOLINIE
Company: CEA Paris-Saclay, Gif-sur-Yvette, France
Contact: dylan.molinie@cea.fr
Date: December 2025

License: GPLv3
"""

import argparse

import numpy as np
import matplotlib.pyplot as plt

import aer.tools as tls
import aer.accuracy as acc
#import aer.models_tk as mtk    # Avoid loading Torch uselessly
from aer import plot
import aer.display as disp

tls.exec_file('.pystartup.py', globals(), root='~/')


def build_model_name(mod_type, rec_cel=None, dtype=None, snr=None, epoch=None):
    """ Copy of the 'aer.models_tk.build_model_name' function to avoid to
    uselessly load Torch at every execution of the script """
    modname = mod_type
    for item in (rec_cel, dtype, str(snr), str(epoch)):
        if item not in (None, '', 'None'):
            modname +=  "_" + item
    return modname


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
FOLDER_HITS = tls.check_folder(FOLDER_ROOT+f"Hits/{SNR}/{DTYPE}/")
FOLDER_LOSS = tls.check_folder(FOLDER_ROOT+f"Losses/{SNR}/{DTYPE}/")

# Build model & folders names
#bar_name = mtk.build_model_name(VARIANT, RECURRENT)
bar_name = build_model_name(VARIANT, RECURRENT)
if BENCH == 1:
    bar_name += f'_{NB_CONV_LAYERS}CV_{NB_FC_LAYERS}FC'
if BENCH == 2:
    bar_name += f'_{CONV_FILTERS}Filters_{NB_NEURONS_FC}Neurons'
folder_loss = tls.check_folder(FOLDER_LOSS + bar_name + '/')
folder_hits = tls.check_folder(FOLDER_HITS + bar_name + '/')

##############################################################################



##############################################################################
##                              Model Training                              ##
##############################################################################

NB_NEURONS_FC = [NB_NEURONS_FC // i for i in range(1, NB_FC_LAYERS+1)]

# The classes (for the plots)
CLASSES = ['BGND', 'Glass', 'Shot', 'Scream', 'Mean']

# Title for the plots (the same for any epoch)
plot_title = f"{VARIANT} $-$ {RECURRENT} $-$ {DTYPE} $-$ SNR {SNR}"
if BENCH > 0:
    plot_title += f" $-$ {NB_CONV_LAYERS} Conv. & {NB_FC_LAYERS} FC"\
        + f" $-$ {CONV_FILTERS} Filters & {NB_NEURONS_FC[0]} Neurons"

#----------------------------   Plot the Hits   -----------------------------#
for i in range(EPOCHS):

    # Reload the hits (if needed) (skip the classes '[0]')
    cmat_trn = acc.load_hits(folder_hits+f"E{i}_ConfMat_trn.csv", SEQUENCES)[1]
    cmat_gen = acc.load_hits(folder_hits+f"E{i}_ConfMat_gen.csv", SEQUENCES)[1]

    # Get the hits (TPs, FPs, FNs & TNs) from the confusion matrices
    hits_trn = [acc.conf_mat_to_acc_items(mat, True) for mat in cmat_trn]
    hits_gen = [acc.conf_mat_to_acc_items(mat, True) for mat in cmat_gen]

    # Compute the number of hits per classe
    avgs_trn = acc.compute_avgs(hits_trn, 'TP', '1s_ref').round(3)
    avgs_gen = acc.compute_avgs(hits_gen, 'TP', '1s_ref').round(3)

    # Plot the hits obtained on training & testing for this epoch
    fig_bars = disp.plot_hits(
        (avgs_trn, avgs_gen),
        labels=("Dataset #", "Accuracy [% Hits]"),
        titles=("Training datasets", "Testing datasets"),
        suptitle=plot_title,
        rtexts=CLASSES,
        fig_params={'figsize': (20.29, 9.41)},
        subfig_params={'hspace': 0.10},
        bar_params={'mean': True},
        label_params={'size': 23},
        title_params={'size': 23, 'pad': 0.025},
        suptitle_params={'size': 28, 'y': 1.07},
        rtext_params={'size': 15, 'stacked': False},
        bartext_params={'size': 10},
        fname=folder_hits+f"E{i}_Hits.png")
    plt.close()
#----------------------------------------------------------------------------#

#---------------------------   Plot the Losses   ----------------------------#
# Reload the training & testing losses
loss_trn = np.loadtxt(folder_loss+"loss_trn.csv", delimiter=',')
loss_gen = np.loadtxt(folder_loss+"loss_gen.csv", delimiter=',')

# Plot the losses on a single graph
fig_loss = plt.figure(figsize=(19.20, 10.80), layout='constrained')
fig_loss = plot.plot(fig_loss.gca(),
    (loss_trn.mean(1), loss_gen.mean(1)),
    labels=("# Epoch", "Loss (MSE)"),
    titles=plot_title,
    legends=("Training", "Testing"),
    fname=folder_loss+"Losses.png")
plt.close()
#----------------------------------------------------------------------------#

##############################################################################
