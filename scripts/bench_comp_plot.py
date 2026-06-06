""" Display & confront the results of the benchmarks

Authors: Dylan MOLINIE
Company: CEA Paris-Saclay, Gif-sur-Yvette, France
Contact: dylan.molinie@cea.fr
Date: April 2026

License: GPLv3
"""

import csv

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import TABLEAU_COLORS

import aer.tools as tls
from aer import plot
import aer.display as disp

tls.exec_file('.pystartup.py', globals(), root='~/')


##############################################################################
##                             Model Selection                              ##
##############################################################################

# Benchmark Identifier
BENCH = 0

# Data format
SEQUENCES = False

# Model & hits folders
FOLDER_ROOT = tls.check_folder(f"Models/Bench_{BENCH}/")
FOLDER_COMP_STATS = tls.check_folder(FOLDER_ROOT+"Comparison/Stats/")
FOLDER_COMP_HITS = tls.check_folder(FOLDER_ROOT+"Comparison/Hits/")

# Data type (for Bench 0)
dtype = 'seqs' if SEQUENCES else 'chks'

# Items to compare (to be written in the CSV file)
if BENCH == 0:
    VAR_1, VAR_2 = "", ""
    TITLE = "Complete models with & without recurrent cells"
elif BENCH == 1:
    VAR_1, VAR_2 = "Conv", "FC"
    TITLE = "ConvNet (NoRec) with 60 Filters & 2048 Neurons"
else:
    VAR_1, VAR_2 = "Filters", "Neurons"
    TITLE = "ConvNet (NoRec) with 2 Convolutions & 2 Fully Connected"
TITLE += " $-$ Sequences" if SEQUENCES else " $-$ Chunks"

##############################################################################



##############################################################################
##                     Benchmark Evaluation Comparison                      ##
##############################################################################

def read_stats(csvfile):
    """ Read & load the statistics from the csv file """
    items, times, params, sizes = [], [], [], []
    with open(csvfile, 'r', encoding='utf-8') as file:
        reader = csv.reader(file)
        next(reader)            # Skip headers
        for line in reader:
            items.append((line[1], line[2]))
            times.append(float(line[3])+float(line[4]))
            params.append((line[5], line[6]))
            sizes.append(line[-1])
    times = (1e3 * np.array(times, float)).round(2)
    params = (1e-6 * np.array(params, int)).round(2)
    sizes = (1e-6 * np.array(sizes, int)).round(2)
    return items, times, params, sizes

def plot_bars(axis, xpos, vals, textsize=10, **bar_params):
    """ Core statements for the plots: bar chart, top bar texts, dashed
    gray line at maximum value and adjust the x-margins """
    # Plot the bars
    plot.bar_core(axis, xpos, vals, mean=False, **bar_params)
    # Draw a horizontal line at the max value
    axis.axhline(np.max(vals), color='lightgray', linestyle='--', label='_nolegend_')
    # Add the values at the top of the bars
    plot.set_bartop_text(axis, vals, size=textsize)
    # Adjust the plots x-margins
    plot.set_margins(axs, x=0.025)

# Read the stats of the benchmark models
items, times, params, sizes = read_stats(FOLDER_COMP_STATS+f"stats_{dtype}.csv")

# Instantiate a figure
fig, axs = plt.subplots(3, 1,
    figsize=(19.20, 10.80), layout='constrained', sharex=True)

# Generate the x-ticks values & labels
variables = [f"{var[0]} {VAR_1}\n{var[1]} {VAR_2}" for var in items]
xpos = np.arange(1, len(variables)+1)

# Plot the bars
plot_bars(axs[0], xpos, times, 10, color=TABLEAU_COLORS, tick_label=variables)
plot_bars(axs[1], xpos, sizes, 10, color=TABLEAU_COLORS)
plot_bars(axs[2], xpos, params, 10, double_bars=True, color=TABLEAU_COLORS)

# Set the plot title & labels
fig.suptitle(TITLE, size=22)
axs[0].set_ylabel("Exec. Time [ms]", size=18)
axs[1].set_ylabel("RAM consumption [Mo]", size=18)
axs[2].set_ylabel("# Parameters (x$10^6$)", size=18)
if BENCH == 1:
    axs[2].legend(['# Mult-Adds', '# Weights'],
        bbox_to_anchor=(1., 0.9), loc='upper right')
elif BENCH == 2:
    axs[2].legend(['# Mult-Adds', '# Weights'],
        bbox_to_anchor=(0., 0.9), loc='upper left')

# Adjust the figure
plt.xticks(rotation=45, ha='center', size=12)

# Save the figure in a file
plt.savefig(FOLDER_COMP_STATS+f"Comparison_{dtype}.pdf", bbox_inches='tight')

##############################################################################



##############################################################################
##                          Plot Models Mean Hits                           ##
##############################################################################

#def plot_bars(axs, xpos, hits, textsize=9, double_bars=False, **bar_params):
#    """ Core statements for the plots: bar chart, top bar texts, dashed
#    gray line at maximum value and adjust the x-margins """
#    # Plot the bars
#    plot.bar_core(axs, xpos, hits,
#        mean=True, double_bars=double_bars, **bar_params)
#    # Plot the surrounding gray bars
#    plot.bar_core(axs, xpos, np.full_like(hits, 100),
#        double_bars=double_bars, color='w', edgecolor='lightgray', fill=False)
#    # Add the rounded values at the top of the bars
#    plot.set_bartop_text(axs, hits.astype(int), size=textsize)
#    # Adjust the plots x-margins
#    plot.set_margins(axs)#, x=0.015)

#def read_results(csvfile, sequences):
#    """ Read & load the statistics from the csv file """
#    end = 13 if sequences else 8
#    items, hits = [], []
#    with open(csvfile, 'r', encoding='utf-8') as file:
#        reader = csv.reader(file)
#        next(reader)            # Skip headers 1
#        next(reader)            # Skip headers 2
#        for line in reader:
#            items.append((line[1], line[2]))
#            hits.append(line[3:end])
#    return items, np.array(hits, float).round(2)

## Signal-to-Noise Ratio
#SNR = 20

## Read the items and the mean hits of the benchmark models
## Note that the items are the same in both files
#items, res_trn = read_results(
#    FOLDER_COMP_HITS+f"res_hits_trn_{dtype}_{SNR}.csv", SEQUENCES)
#items, res_gen = read_results(
#    FOLDER_COMP_HITS+f"res_hits_gen_{dtype}_{SNR}.csv", SEQUENCES)

## Reshape the data if sequences to comply with the 'bar_core' format
#if SEQUENCES:
#    res_trn = res_trn.reshape(len(items), -1, 2)
#    res_gen = res_gen.reshape(len(items), -1, 2)

## Instantiate the figure
#fig = plt.figure(layout='constrained', figsize=(19.20, 7.20))
#subfigs = fig.subfigures(1, 2)#, width_ratios=(1, 1))
#axs_lft = subfigs[0].subplots(3, 3, sharex=True, sharey=True)   # Left figure
#axs_rgt = subfigs[1].subplots(3, 3, sharex=True, sharey=True)   # Right figure

## Plot the results as bar charts
#labels = ['Bk', 'GB', 'Sh', 'Sc', r'$\overline{\mu}$']
#xpos = np.arange(1, len(labels)+1)
#plot_bars(axs_lft.flatten(), xpos, res_trn, 9,
#    double_bars=SEQUENCES, color=TABLEAU_COLORS, tick_label=labels)
#plot_bars(axs_rgt.flatten(), xpos, res_gen, 9,
#    double_bars=SEQUENCES, color=TABLEAU_COLORS, tick_label=labels)

## Add first items as y-axis labels
#for i, (axis_lft, axis_rgt) in enumerate(zip(axs_lft[:, 0], axs_rgt[:, 0])):
#    axis_lft.set_ylabel(f"{items[i*3][0]} {VAR_1}", size=14)
#    axis_rgt.set_ylabel(f"{items[i*3][0]} {VAR_1}", size=14)

## Add the second items as axis titles
#for i, (axis_lft, axis_rgt) in enumerate(zip(axs_lft[0], axs_rgt[0])):
#    axis_lft.set_title(f"{items[i][1]} {VAR_2}", size=14)
#    axis_rgt.set_title(f"{items[i][1]} {VAR_2}", size=14)

## Set the figure titles
#subfigs[0].suptitle("TPs on Training", size=18)
#subfigs[1].suptitle("TPs on Testing", size=18)

## Save the figure in a file
#plt.savefig(FOLDER_COMP_HITS+f'Hits_{dtype}_{SNR}.pdf', bbox_inches='tight')
#plt.close()

##############################################################################
