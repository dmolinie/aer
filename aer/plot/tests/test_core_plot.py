import numpy as np
import matplotlib.pyplot as plt
from aer.plot._core_plot import *

def test_plot_core():
    """ Plot (a set of) data on (a set of) figure' axes """
    # Generate dummy data
    tstp = np.linspace(0., 10., 100)
    sin, cos = np.sin(tstp), np.cos(tstp)
    #--- 1-plot figure
    # Plot the sine with dynamic inputs
    fig = plt.figure(figsize=(19.20, 10.80))
    plot_core(fig.gca(), sin, fmt='--', linewidth=3)    # Auto xvalues
    plot_core(fig.gca(), tstp, sin, linewidth=3)        # Explicit xvalues
    # Plot the sine & cosine with the same time scale on the same plot
    fig = plt.figure(figsize=(19.20, 10.80))
    plot_core(fig.gca(),
        tstp, (cos, sin),
        fmt='--', linewidth=3)
    # Plot the sine & cosine with the specific time scales on the same plot
    fig = plt.figure(figsize=(19.20, 10.80))
    plot_core(fig.gca(),
        (tstp, tstp), (cos, sin),
        fmt='--', linewidth=3)
    #--- N-plot figure
    # Plot the sine & cosine with the same time scale on different plots
    fig, axs = plt.subplots(2, 1, figsize=(19.20, 10.80))
    plot_core(axs,
        tstp, (sin, cos),
        fmt='--', linewidth=3)
    # Plot the sine & cosine with specific time scales on different plots
    fig, axs = plt.subplots(2, 1, figsize=(19.20, 10.80))
    plot_core(axs,
        (tstp, tstp),
        (cos, sin),
        fmt='--', linewidth=3)
    # Plot the data on an NxM-plot graph in column-major order
    fig, axs = plt.subplots(3, 2, figsize=(19.20, 10.80))
    plot_core(axs.T,        # Transpose the axes for column-major order
        (tstp, tstp),
        ((sin, cos, (sin, cos)), (sin, cos,( sin, cos))),
        fmt='--', linewidth=3)
    plt.show()

def test_bar_core():
    """ Plot data bars on a figure """
    #--- Single bar plots examples
    # Generate dummy data & add the mean
    vals = np.arange(1, 30)                         # 1D data
    vals = np.hstack((vals, vals.mean()))           # Append the mean
    # Generate simple x-ticks (same as those automatically generated)
    xpos = np.arange(1, len(vals)+1)                # +1 for the mean
    # Generate simple x-labels (optional)
    tick_labels = np.array(np.arange(1, len(xpos)+1), dtype=str)
    # 1-plot figure
    fig = plt.figure()
    bar_core(fig.gca(),
        xpos,           # optional
        vals,
        mean=True,
        # Axis' `bar` method's parameters
        tick_label=tick_labels,
        color='blue')
    # 2-plot figure
    fig, axs = plt.subplots(2, 1)
    bar_core(axs,
        xpos,           # or (xpos, xpos)
        (vals, vals),
        mean=True)
    # 3x2-plot figure in row-major order
    fig, axs = plt.subplots(3, 2)
    bar_core(axs,       # row-major
        ((vals, vals), (vals, vals), (vals, vals)), # 3 tuples of 2 arrays
        mean=True)
    # 3x2-plot figure in column-major order
    fig, axs = plt.subplots(3, 2)
    bar_core(axs.T,                                 # column-major
        ((vals, vals, vals), (vals, vals, vals)),   # 2 tuples of 3 arrays
        mean=True)
    plt.show()
    #--- Double bar plots examples
    # Generate dummy data
    vals = np.arange(1, 31).reshape(-1, 2)          # 2D data
    vals = np.vstack((vals, vals.mean(0)))          # Append the mean
    # Generate simple x-ticks (same as those automatically generated)
    xpos = np.arange(1, len(vals)+1)                # +1 for the mean
    # Generate simple x-labels (optional)
    tick_labels = np.array(np.arange(1, len(xpos)+1), dtype=str)
    # 1-plot figure
    fig, axs = plt.subplots(1, 1)
    bar_core(axs,
        xpos,   # optional
        vals,
        mean=True,
        double_bars=True,
        # Axis' `bar` method's parameters
        tick_label=tick_labels,
        color='blue',
        width=0.3)
    # 2-plot figure
    fig, axs = plt.subplots(2, 1)
    bar_core(axs,
        xpos,   # or (xpos, xpos)
        (vals, vals),
        mean=True,
        double_bars=True)
    # 2x2-plot figure with auto xpos
    fig, axs = plt.subplots(2, 2)
    bar_core(axs,
        ((vals, vals), (vals, vals)),
        mean=True,
        double_bars=True)
    # 2x2-plot figure
    fig, axs = plt.subplots(2, 2)
    bar_core(axs,
        xpos,   # or (xpos, xpos) or ((xpos, xpos), (xpos, xpos))
        ((vals, vals), (vals, vals)),
        mean=True,
        double_bars=True)
    plt.show()



# Launch test/example functions
test_plot_core()

test_bar_core()

