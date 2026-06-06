""" Script reload models hits & losses to display them

Authors: Dylan MOLINIE
Company: CEA Paris-Saclay, Gif-sur-Yvette, France
Contact: dylan.molinie@cea.fr
Date: April 2026

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
FOLDER_HITS = tls.check_folder(FOLDER_ROOT+f"Hits/{SNR}/{DTYPE}/")
FOLDER_LOSS = tls.check_folder(FOLDER_ROOT+f"Losses/{SNR}/{DTYPE}/")

# Build model & folders names
#bar_name = mtk.build_model_name(VARIANT, RECURRENT)
bar_name = build_model_name(VARIANT, RECURRENT)
folder_loss = tls.check_folder(FOLDER_LOSS + bar_name + '/')
folder_hits = tls.check_folder(FOLDER_HITS + bar_name + '/')

##############################################################################



##############################################################################
##                       Compute the Hits from Model                        ##
##############################################################################

#import torch
#import aer.models_tk as mtk

#FOLDER_MODS = tls.check_folder(FOLDER_ROOT+f"Models/{SNR}/{DTYPE}/")
#mod_name = bar_name + f"_{dtype}_{SNR}"
#folder_mods = tls.check_folder(FOLDER_MODS + bar_name + '/')
#dtype = 'seqs' if SEQUENCES else 'chks'

## Torch parameters
#torch.set_default_dtype(torch.float32)
#DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
#print("Torch device:", DEVICE)

## Data rescaling bounds
#BOUNDS = (-1., +1.)

## Dataset indexes
#idx_trn = range(1, 67)  # 67
#idx_gen = range(1, 30)  # 30

## Load the Torch model
#model = torch.load(folder_mods+mod_name+'.pth', weights_only=False)

## Compute accuracy on the training & testing data
#cmat_trn = mtk.test_model_accuracy_mivia(
#    model, dataset_trn, parsing_params, idx_trn,
#    bounds=BOUNDS, conf_mat=True)
#cmat_gen = mtk.test_model_accuracy_mivia(
#    model, dataset_gen, parsing_params, idx_gen,
#    bounds=BOUNDS, conf_mat=True)

## Save the number of hits and the hit ratios
#acc.save_hits(folder_hits+"ConfMat_trn.csv", cmat_trn)
#acc.save_hits(folder_hits+"ConfMat_gen.csv", cmat_gen)

##############################################################################



##############################################################################
##                           Display Hits & Loss                            ##
##############################################################################

# The classes (for the plots)
CLASSES = ['BGND', 'Glass', 'Shot', 'Scream', 'Mean']
# Title for the plots (the same for any epoch)
plot_title = f"{VARIANT} $-$ {RECURRENT} $-$ {DTYPE} $-$ SNR {SNR}"

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
